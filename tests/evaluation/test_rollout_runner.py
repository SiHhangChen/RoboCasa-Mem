from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.evaluation.adapter import EpisodeRuntime
from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.rollout import RolloutRunner
from bench.evaluation.suite import ArtifactConfig, EvalSuiteConfig, RolloutConfig, load_eval_suite
from bench.policy.protocol import (
    ActionContract,
    CanonicalObservation,
    CheckpointIdentity,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
)
from bench.policy.protocol.contracts import ACTION_SCHEMA
from bench.runtime.data_contract import STATE_SCHEMA
from bench.runtime.contracts import RuntimeStep, TaskDiagnostics
from bench.evaluation.subtask import SubtaskTransition


def _contract() -> PolicyContract:
    return PolicyContract(
        server_id="fake",
        policy={"family": "fake"},
        checkpoint=CheckpointIdentity("test://fake", "sha256:fake"),
        observation=ObservationContract(
            STATE_SCHEMA,
            37,
            ("agentview_left", "agentview_right", "eye_in_hand"),
            (2, 2),
        ),
        action=ActionContract(ACTION_SCHEMA, 13, 2),
        control_frequency_hz=20,
    )


class FakeClient:
    def __init__(self) -> None:
        self.contract = _contract()
        self.episode_id = None
        self.calls = 0

    def reset(self, episode_id: str) -> None:
        self.episode_id = episode_id

    def infer(self, observation, *, episode_id: str, step: int) -> InferenceResponse:
        assert episode_id == self.episode_id
        self.calls += 1
        return InferenceResponse(
            request_id=f"request-{self.calls}",
            episode_id=episode_id,
            actions=np.zeros((2, 13), dtype=np.float32),
            timing={"model_ms": 1.0},
        )


class PromptRecordingClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    def infer(self, observation, *, episode_id: str, step: int) -> InferenceResponse:
        self.prompts.append(observation.prompt)
        return super().infer(observation, episode_id=episode_id, step=step)


class FakeEnv:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return {"step": self.steps}, 1.0, False, {"success": self.steps >= 3}

    def close(self) -> None:
        pass


@dataclass
class FakeAdapter:
    task_id: str = "WX-01"

    def create_episode(self, *, seed, policy_contract, suite):
        return EpisodeRuntime(
            episode_id=f"wx01-{seed}",
            seed=seed,
            env=FakeEnv(),
            observation={"step": 0},
            prompt="return cup",
        )

    def build_observation(self, runtime):
        return CanonicalObservation(
            task_id=self.task_id,
            state=np.zeros(37, dtype=np.float32),
            images={
                "agentview_left": np.zeros((2, 2, 3), dtype=np.uint8),
                "agentview_right": np.zeros((2, 2, 3), dtype=np.uint8),
                "eye_in_hand": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            prompt=runtime.prompt,
        )

    def check_success(self, runtime, info):
        return bool(info["success"])

    def diagnostics(self, runtime):
        return {"steps": runtime.env.steps}

    def close(self, runtime):
        runtime.env.close()


class FakeSubtaskScheduler:
    def __init__(self) -> None:
        self.step = 0
        self.prompt = "subtask one"

    @property
    def current_prompt(self) -> str:
        return self.prompt

    @property
    def current_subtask(self) -> str:
        return "one" if self.step < 2 else "two"

    def after_step(self, runtime, info):
        del info
        self.step += 1
        if self.step == 1:
            self.prompt = "subtask two"
            return SubtaskTransition(
                changed=True,
                previous_subtask="one",
                current_subtask="two",
                prompt=self.prompt,
                environment_step=runtime.env.steps,
            )
        return SubtaskTransition(
            complete=self.step >= 3,
            current_subtask="two",
            prompt=self.prompt,
            environment_step=runtime.env.steps,
        )

    def to_metadata(self):
        return {"current_subtask": self.current_subtask, "step": self.step}


class SubtaskAdapter(FakeAdapter):
    def create_subtask_scheduler(self, runtime, suite):
        del runtime, suite
        return FakeSubtaskScheduler()


class ReportAdapter(FakeAdapter):
    report_stage_profile_id = "WX-01/report-v1"

    def report_stages(self, runtime):
        success = runtime.env.steps >= 3
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": self.task_id,
            "profile_id": self.report_stage_profile_id,
            "order": ["wipe_completed", "cleaning_completed", "success"],
            "stages": {
                "wipe_completed": {
                    "applicable": True,
                    "passed": runtime.env.steps >= 1,
                    "first_pass_step": 1 if runtime.env.steps >= 1 else None,
                },
                "cleaning_completed": {
                    "applicable": True,
                    "passed": runtime.env.steps >= 2,
                    "first_pass_step": 2 if runtime.env.steps >= 2 else None,
                },
                "success": {
                    "applicable": True,
                    "passed": success,
                    "first_pass_step": 3 if success else None,
                },
            },
        }


class MissingReportAdapter(FakeAdapter):
    report_stage_profile_id = "WX-01/report-v1"
    report_stages = None


class InvalidReportAdapter(FakeAdapter):
    report_stage_profile_id = "WX-01/report-v1"

    def report_stages(self, runtime):
        del runtime
        return {
            "schema_version": "membench.report_stages/v1",
            "task_id": self.task_id,
            "profile_id": self.report_stage_profile_id,
            "order": ["success"],
            "stages": {
                "success": {
                    "applicable": False,
                    "passed": None,
                    "first_pass_step": None,
                }
            },
        }


def test_runner_replans_and_stops_on_success(tmp_path: Path):
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(max_steps=10, replan_steps=2),
        artifacts=ArtifactConfig(record_video=False),
    )
    client = FakeClient()
    writer = ArtifactWriter(tmp_path / "run")
    runner = RolloutRunner(client=client, suite=suite, artifacts=writer)

    result = runner.run_episode(FakeAdapter(), 1000)

    assert result.status == "success"
    assert result.success_step == 3
    assert result.env_steps == 3
    assert result.policy_calls == 2
    assert result.policy_actions_returned == 4
    assert result.policy_actions_consumed == 3
    assert result.reward_sum == 3
    assert (writer.episodes_dir / "seed_001000" / "result.json").is_file()


def test_runner_switches_prompt_and_discards_old_action_queue():
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(max_steps=10, replan_steps=2),
        artifacts=ArtifactConfig(record_video=False),
    )
    client = PromptRecordingClient()
    result = RolloutRunner(client=client, suite=suite).run_episode(
        SubtaskAdapter(), 1000
    )

    assert result.status == "success"
    # The second action from the first chunk is discarded at the handoff, so
    # the second inference is made with the new prompt at environment step 1.
    assert client.prompts[:2] == ["subtask one", "subtask two"]
    assert result.episode_metadata["subtask_scheduler"]["step"] == 3


def test_runner_persists_declared_report_stage_profile(tmp_path: Path) -> None:
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(max_steps=10, replan_steps=2),
        artifacts=ArtifactConfig(record_video=False),
    )
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=suite.seeds)
    runner = RolloutRunner(client=FakeClient(), suite=suite, artifacts=writer)

    result = runner.run_episode(ReportAdapter(), 1000)

    assert result.status == "success"
    assert result.report_stages is not None
    assert result.report_stages["order"] == [
        "wipe_completed",
        "cleaning_completed",
        "success",
    ]
    payload = json.loads(
        (writer.episodes_dir / "seed_001000" / "result.json").read_text()
    )
    assert payload["report_stages"] == result.report_stages
    assert writer.write_summary()["report_stages"]["complete"] is True


@pytest.mark.parametrize(
    "adapter",
    (MissingReportAdapter(), InvalidReportAdapter()),
    ids=("missing", "invalid"),
)
def test_declared_report_stage_profile_requires_valid_report(adapter) -> None:
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(max_steps=10, replan_steps=2),
        artifacts=ArtifactConfig(record_video=False),
    )
    runner = RolloutRunner(client=FakeClient(), suite=suite)

    result = runner.run_episode(adapter, 1000)

    assert result.status == "error"
    assert result.success is False
    assert result.report_stages is None
    assert "report" in str(result.error)


class SemanticTerminalRuntime:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, action) -> RuntimeStep:
        del action
        self.steps += 1
        terminal = self.steps >= 3
        return RuntimeStep(
            observation={"step": self.steps},
            reward=0.0,
            diagnostics=TaskDiagnostics(
                success=False,
                terminal=terminal,
                failure_code=(
                    "sink_reentered_after_departure" if terminal else None
                ),
            ),
        )


class SemanticTerminalAdapter(FakeAdapter):
    def create_episode(self, *, seed, policy_contract, suite):
        del policy_contract, suite
        shared = SemanticTerminalRuntime()
        return EpisodeRuntime(
            episode_id=f"wx01-{seed}",
            seed=seed,
            env=FakeEnv(),
            observation={"step": 0},
            prompt="return cup",
            metadata={"shared_runtime": shared},
        )

    def check_success(self, runtime, info):
        del runtime, info
        return False


def test_semantic_terminal_stops_even_when_env_done_is_ignored(tmp_path: Path):
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(
            max_steps=10,
            replan_steps=2,
            ignore_done=True,
        ),
        artifacts=ArtifactConfig(record_video=False),
    )
    runner = RolloutRunner(client=FakeClient(), suite=suite)

    result = runner.run_episode(SemanticTerminalAdapter(), 1000)

    assert result.status == "failure"
    assert result.failure_reason == "sink_reentered_after_departure"
    assert result.env_steps == 3


class WR07TerminalRuntime(FakeEnv):
    def __init__(self, *, matched: bool, target_slot: str) -> None:
        from bench.tasks.wr07.evaluation import WR07EvaluationSnapshot, WR07Evaluator

        super().__init__()
        self.matched = matched
        self.target_slot = target_slot
        self.evaluator = WR07Evaluator()
        self.evaluator.reset(WR07EvaluationSnapshot(step_id=0, item_placed=True))

    def step(self, action) -> RuntimeStep:
        from bench.tasks.wr07.evaluation import WR07EvaluationSnapshot

        del action
        assert not self.evaluator.result().terminal, "action executed after WR-07 termination"
        self.steps += 1
        result = self.evaluator.observe(WR07EvaluationSnapshot(
            step_id=self.steps,
            item_picked=self.steps == 1,
            item_placed=self.steps >= 2,
            target_slot_attempted=self.target_slot if self.steps >= 2 else None,
            target_color_matches=self.matched and self.steps >= 2,
        ))
        return RuntimeStep(
            observation={"step": self.steps},
            reward=float(result.success),
            info={"success": result.success},
            diagnostics=TaskDiagnostics(
                success=result.success,
                terminal=result.terminal,
                failure_code=result.failure_code,
                details=result.to_diagnostics(),
                stop_execution=result.stop_execution,
            ),
        )


class WR07TerminalAdapter(FakeAdapter):
    report_stage_profile_id = "WR-07/report-v4"

    def __init__(self, *, matched: bool, target_slot: str) -> None:
        super().__init__(task_id="WR-07")
        self.matched = matched
        self.target_slot = target_slot

    def create_episode(self, *, seed, policy_contract, suite):
        del policy_contract, suite
        shared = WR07TerminalRuntime(matched=self.matched, target_slot=self.target_slot)
        return EpisodeRuntime(
            episode_id=f"wr07-{seed}",
            seed=seed,
            env=shared,
            observation={"step": 0},
            prompt="move the carton to the matching plate",
            metadata={"shared_runtime": shared},
        )

    def diagnostics(self, runtime):
        return runtime.env.evaluator.result().to_diagnostics()

    def report_stages(self, runtime):
        return runtime.env.evaluator.result().to_report_stages()


@pytest.mark.parametrize("matched", (False, True), ids=("wrong_plate", "success"))
@pytest.mark.parametrize("action_repeat", (1, 4))
@pytest.mark.parametrize("target_slot", ("target_left", "target_center", "target_right"))
@pytest.mark.parametrize("stop_on_success", (False, True))
@pytest.mark.parametrize("ignore_done", (False, True))
def test_wr07_terminal_stops_without_executing_remaining_actions(
    matched, action_repeat, target_slot, stop_on_success, ignore_done,
):
    canonical = load_eval_suite(
        Path(__file__).resolve().parents[2] / "configs/eval/suites/wr07_v1.yaml"
    )
    assert canonical.rollout.stop_on_success
    assert canonical.rollout.ignore_done
    suite = replace(
        canonical,
        seeds=(1050,),
        rollout=replace(
            canonical.rollout, max_steps=10, replan_steps=2, action_repeat=action_repeat,
            stop_on_success=stop_on_success, ignore_done=ignore_done,
        ),
        artifacts=ArtifactConfig(record_video=False),
    )
    result = RolloutRunner(client=FakeClient(), suite=suite).run_episode(
        WR07TerminalAdapter(matched=matched, target_slot=target_slot), 1050,
    )

    assert result.status == ("success" if matched else "failure"), result.error
    assert result.success is matched
    assert result.env_steps == 3
    assert result.success_step == (3 if matched else None)
    assert result.failure_reason == (None if matched else "wrong_target_placement")
    assert result.policy_calls == (2 if action_repeat == 1 else 1)
    assert result.policy_actions_consumed == (3 if action_repeat == 1 else 1)
    assert result.diagnostics["terminal"]
    assert result.diagnostics["stop_execution"]
    assert result.diagnostics["evaluation"]["terminal_step"] == 3
    assert result.report_stages["stages"]["item_placed"]["passed"]
    assert result.report_stages["stages"]["success"]["passed"] is matched


class WR08TerminalRuntime(FakeEnv):
    def __init__(self, *, wrong_item_index: int | None) -> None:
        from bench.tasks.wr08.domain.contracts import ITEM_IDS
        from bench.tasks.wr08.domain.episode import resolve_episode_plan
        from bench.tasks.wr08.evaluation import WR08Evaluator

        super().__init__()
        self.items = ITEM_IDS
        self.plan = resolve_episode_plan(seed=0)
        self.targets = {
            item: self.plan.target_slot_by_color[self.plan.source_color_by_item[item]]
            for item in self.items
        }
        self.wrong_item = None if wrong_item_index is None else self.items[wrong_item_index]
        self.evaluator = WR08Evaluator()
        self.evaluator.reset(self.snapshot())

    def snapshot(self):
        from bench.tasks.wr08.domain.contracts import TARGET_SLOTS
        from bench.tasks.wr08.evaluation import WR08EvaluationSnapshot

        first, second = self.items
        placed = {first: self.steps >= 2, second: self.steps >= 5}
        attempts = {
            item: next(slot for slot in TARGET_SLOTS if slot != self.targets[item])
            if item == self.wrong_item else self.targets[item]
            for item in self.items if placed[item]
        }
        return WR08EvaluationSnapshot(
            step_id=self.steps,
            source_slot_by_item=self.plan.source_slot_by_item,
            source_color_by_item=self.plan.source_color_by_item,
            target_slot_by_item=self.targets,
            item_picked_by_item={first: self.steps == 1, second: self.steps == 4},
            item_placed_by_item=placed,
            item_matched_by_item={item: attempts.get(item) == self.targets[item] for item in self.items},
            released_by_item={item: True for item in self.items},
            target_slot_attempted_by_item=attempts,
        )

    def step(self, action) -> RuntimeStep:
        del action
        assert not self.evaluator.result().stop_execution, "action executed after WR-08 termination"
        self.steps += 1
        result = self.evaluator.observe(self.snapshot())
        return RuntimeStep(
            observation={"step": self.steps}, info={"success": result.success},
            diagnostics=TaskDiagnostics(
                success=result.success, terminal=result.terminal,
                failure_code=result.failure_code, details=result.to_diagnostics(),
                stop_execution=result.stop_execution,
            ),
        )


class WR08TerminalAdapter(FakeAdapter):
    report_stage_profile_id = "WR-08/report-v3"

    def __init__(self, *, wrong_item_index: int | None) -> None:
        super().__init__(task_id="WR-08")
        self.wrong_item_index = wrong_item_index

    def create_episode(self, *, seed, policy_contract, suite):
        shared = WR08TerminalRuntime(wrong_item_index=self.wrong_item_index)
        return EpisodeRuntime(
            episode_id=f"wr08-{seed}", seed=seed, env=shared, observation={"step": 0},
            prompt="move both cartons to matching plates", metadata={"shared_runtime": shared},
        )

    def diagnostics(self, runtime):
        return runtime.env.evaluator.result().to_diagnostics()

    def report_stages(self, runtime):
        return runtime.env.evaluator.result().to_report_stages()


@pytest.mark.parametrize("wrong_item_index", (None, 0, 1), ids=("success", "first_wrong", "second_wrong"))
@pytest.mark.parametrize("action_repeat", (1, 4))
@pytest.mark.parametrize("stop_on_success", (False, True))
@pytest.mark.parametrize("ignore_done", (False, True))
def test_wr08_stops_at_the_final_decision_even_inside_an_action_chunk(
    wrong_item_index, action_repeat, stop_on_success, ignore_done,
):
    suite = load_eval_suite(
        Path(__file__).resolve().parents[2] / "configs/eval/suites/wr08_v1.yaml"
    )
    suite = replace(
        suite, seeds=(1050,), artifacts=ArtifactConfig(record_video=False),
        rollout=replace(
            suite.rollout, max_steps=10, replan_steps=2, action_repeat=action_repeat,
            stop_on_success=stop_on_success, ignore_done=ignore_done,
        ),
    )
    result = RolloutRunner(client=FakeClient(), suite=suite).run_episode(
        WR08TerminalAdapter(wrong_item_index=wrong_item_index), 1050,
    )
    success = wrong_item_index is None
    steps = 3 if wrong_item_index == 0 else 6
    assert result.status == ("success" if success else "failure"), result.error
    assert result.env_steps == steps
    assert result.success is success
    assert result.failure_reason == (None if success else "wrong_target_placement")
    assert result.policy_actions_consumed == (steps + action_repeat - 1) // action_repeat
    assert result.diagnostics["stop_execution"]
    assert result.diagnostics["evaluation"]["terminal_step"] == steps
    assert result.report_stages["stages"]["success"]["passed"] is success


class SuccessThenViolationRuntime:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, action) -> RuntimeStep:
        del action
        self.steps += 1
        success = self.steps in (2, 3)
        violation = self.steps >= 4
        return RuntimeStep(
            observation={"step": self.steps},
            info={"success": success},
            diagnostics=TaskDiagnostics(
                success=success,
                terminal=success or violation,
                failure_code="item_removed_after_success" if violation else None,
            ),
        )


class SuccessThenViolationAdapter(FakeAdapter):
    def create_episode(self, *, seed, policy_contract, suite):
        del policy_contract, suite
        shared = SuccessThenViolationRuntime()
        return EpisodeRuntime(
            episode_id=f"wr04-{seed}",
            seed=seed,
            env=FakeEnv(),
            observation={"step": 0},
            prompt="return sponge",
            metadata={"shared_runtime": shared},
        )


def test_full_horizon_success_is_revoked_by_later_violation() -> None:
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WR-04",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(
            max_steps=10,
            replan_steps=2,
            stop_on_success=False,
            ignore_done=True,
        ),
        artifacts=ArtifactConfig(record_video=False),
    )
    runner = RolloutRunner(client=FakeClient(), suite=suite)

    result = runner.run_episode(
        SuccessThenViolationAdapter(task_id="WR-04"),
        1000,
    )

    assert result.status == "failure"
    assert result.success is False
    assert result.success_step == 2
    assert result.failure_reason == "item_removed_after_success"
    assert result.env_steps == 4


class TransientSuccessEnv(FakeEnv):
    def step(self, action):
        del action
        self.steps += 1
        return {"step": self.steps}, 0.0, False, {"success": self.steps == 2}


class TransientSuccessAdapter(FakeAdapter):
    def create_episode(self, *, seed, policy_contract, suite):
        del policy_contract, suite
        return EpisodeRuntime(
            episode_id=f"wx01-{seed}",
            seed=seed,
            env=TransientSuccessEnv(),
            observation={"step": 0},
            prompt="return cup",
        )


def test_full_horizon_reports_success_that_is_not_preserved() -> None:
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(
            max_steps=4,
            replan_steps=2,
            stop_on_success=False,
            ignore_done=True,
        ),
        artifacts=ArtifactConfig(record_video=False),
    )

    result = RolloutRunner(client=FakeClient(), suite=suite).run_episode(
        TransientSuccessAdapter(),
        1000,
    )

    assert result.status == "failure"
    assert result.success_step == 2
    assert result.failure_reason == "terminal_state_not_preserved"


class CloseFailAdapter(FakeAdapter):
    def close(self, runtime):
        runtime.env.close()
        raise RuntimeError("close failed")


def test_close_failure_is_committed_as_infrastructure_error(tmp_path: Path):
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(max_steps=10, replan_steps=2),
        artifacts=ArtifactConfig(record_video=False),
    )
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=suite.seeds)
    runner = RolloutRunner(client=FakeClient(), suite=suite, artifacts=writer)

    result = runner.run_episode(CloseFailAdapter(), 1000)

    assert result.status == "error"
    assert "adapter.close" in str(result.error)
    payload = json.loads(
        (writer.episodes_dir / "seed_001000" / "result.json").read_text()
    )
    assert payload["status"] == "error"
    assert writer.write_summary()["publishable"] is False


class InterruptEnv(FakeEnv):
    def step(self, action):
        del action
        raise KeyboardInterrupt


class InterruptAdapter(FakeAdapter):
    def create_episode(self, *, seed, policy_contract, suite):
        del policy_contract, suite
        return EpisodeRuntime(
            episode_id=f"wx01-{seed}",
            seed=seed,
            env=InterruptEnv(),
            observation={"step": 0},
            prompt="return cup",
        )


def test_keyboard_interrupt_commits_cancelled_episode_then_reraises(tmp_path: Path):
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000,),
        rollout=RolloutConfig(max_steps=10, replan_steps=2),
        artifacts=ArtifactConfig(record_video=False),
    )
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=suite.seeds)
    runner = RolloutRunner(client=FakeClient(), suite=suite, artifacts=writer)

    with pytest.raises(KeyboardInterrupt):
        runner.run_episode(InterruptAdapter(), 1000)

    payload = json.loads(
        (writer.episodes_dir / "seed_001000" / "result.json").read_text()
    )
    assert payload["status"] == "cancelled"
    assert not list(writer.episodes_dir.glob(".seed_001000.*"))
