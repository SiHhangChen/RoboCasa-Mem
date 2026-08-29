from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from bench.evaluation.adapter import EpisodeRuntime
from bench.evaluation.canonical_stage import (
    CanonicalStageTask,
    RecordedExpertController,
    RemotePolicyModelAdapter,
)
from bench.evaluation.stage_rollout import StageSpec
from bench.evaluation.suite import EvalSuiteConfig, RolloutConfig


@dataclass
class _Contract:
    pass


class _SharedRuntime:
    def __init__(self) -> None:
        self.value = 0

    def step(self, action: np.ndarray) -> SimpleNamespace:
        self.value += 1
        return SimpleNamespace(
            observation={"value": self.value},
            terminal=False,
            terminated=False,
            truncated=False,
        )


class _Adapter:
    task_id = "TEST"

    def __init__(self) -> None:
        self.shared = _SharedRuntime()
        self.closed = 0

    def create_episode(self, *, seed, policy_contract, suite):
        del policy_contract, suite
        return EpisodeRuntime(
            episode_id=f"test-{seed}",
            seed=seed,
            env=object(),
            observation={"value": 0},
            prompt="do the task",
            metadata={"shared_runtime": self.shared},
        )

    def build_observation(self, runtime):
        return {"canonical_value": runtime.observation["value"]}

    def close(self, runtime):
        del runtime
        self.closed += 1


class _Client:
    def __init__(self) -> None:
        self.reset_ids = []
        self.infer_steps = []

    def reset(self, episode_id):
        self.reset_ids.append(episode_id)

    def infer(self, observation, *, episode_id, step):
        del observation, episode_id
        self.infer_steps.append(step)
        return SimpleNamespace(
            request_id="request",
            actions=np.zeros((2, 13), dtype=np.float32),
            timing={},
        )


def _task() -> CanonicalStageTask:
    adapter = _Adapter()
    suite = EvalSuiteConfig(
        suite_version="test",
        task_id="TEST",
        env_split="pretrain",
        seeds=(7,),
        rollout=RolloutConfig(max_steps=10, replan_steps=2),
    )
    return CanonicalStageTask(
        task_id="TEST",
        stage_specs=(StageSpec("A", lambda state: state["value"] > 0),),
        adapter=adapter,
        suite=suite,
        policy_contract=_Contract(),
        privileged_state_builder=lambda runtime: dict(runtime.observation),
    )


def test_canonical_stage_task_uses_shared_runtime_and_prompt() -> None:
    task = _task()
    observation = task.reset(7)
    assert observation == {"canonical_value": 0}
    assert task.instruction == "do the task"
    transition = task.step(np.zeros(13, dtype=np.float32))
    assert transition.observation == {"canonical_value": 1}
    assert transition.raw_transition.observation == {"value": 1}
    assert task.get_privileged_state() == {"value": 1}
    task.close()


def test_remote_policy_adapter_resets_once_and_uses_environment_step() -> None:
    task = _task()
    task.reset(7)
    client = _Client()
    model = RemotePolicyModelAdapter(client=client, task=task)
    model.reset_episode(task.instruction)
    task.step(np.zeros(13, dtype=np.float32))
    actions = model.predict(task.observe())
    assert actions.shape == (2, 13)
    assert client.reset_ids == ["test-7"]
    assert client.infer_steps == [1]


def test_recorded_expert_controller_replays_actions_in_order() -> None:
    actions = np.stack(
        [np.zeros(13, dtype=np.float32), np.ones(13, dtype=np.float32)]
    )
    expert = RecordedExpertController(actions)
    np.testing.assert_array_equal(expert.predict({}), actions[0])
    np.testing.assert_array_equal(expert.predict({}), actions[1])
    assert expert.remaining_steps == 0
