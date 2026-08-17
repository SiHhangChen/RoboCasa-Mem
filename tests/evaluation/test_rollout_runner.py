from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.evaluation.adapter import EpisodeRuntime
from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.rollout import RolloutRunner
from bench.evaluation.suite import ArtifactConfig, EvalSuiteConfig, RolloutConfig
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
