"""Task-owned execution stops cannot be bypassed at the simulator boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bench.runtime.contracts import InitializedTaskRuntime, RuntimeStep, TaskDiagnostics
from bench.runtime.robocasa import RoboCasaEpisodeRuntime


class CountingEnv:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self):
        self.steps = 0
        return {"step": self.steps}

    def step(self, action):
        del action
        self.steps += 1
        return {"step": self.steps}, 0.0, False, {}


def _runtime(*, hard_stop: bool, success: bool) -> RoboCasaEpisodeRuntime:
    env = CountingEnv()

    def diagnostics(initialized):
        terminal = initialized.env.steps >= 2
        return {
            "success": success and terminal,
            "terminal": terminal,
            "stop_execution": hard_stop and terminal,
            "failure_code": "wrong_target_placement" if terminal and not success else None,
        }

    def initialize(descriptor, env, observation, metadata):
        return InitializedTaskRuntime(
            descriptor=descriptor, env=env, observation=observation, metadata=dict(metadata),
        )

    episode = SimpleNamespace(task_id="WR-07")
    initialized = initialize(SimpleNamespace(episode=episode), env, env.reset(), {})
    api = SimpleNamespace(
        initialize_runtime=initialize,
        diagnostics=diagnostics,
        check_success=lambda initialized, info: diagnostics(initialized)["success"],
        build_expert_runtime=None,
    )
    return RoboCasaEpisodeRuntime(episode, initialized, api)


@pytest.mark.parametrize("success", (False, True), ids=("failure", "success"))
def test_runtime_rejects_actions_after_task_stop_until_reset(monkeypatch, success):
    runtime = _runtime(hard_stop=True, success=success)
    action = np.zeros(13, dtype=np.float32)
    assert not runtime.step(action).diagnostics.stop_execution
    final = runtime.step(action)
    assert final.terminal
    assert final.diagnostics.stop_execution
    assert final.diagnostics.success is success
    assert runtime.env.steps == 2

    for _ in range(2):
        with pytest.raises(RuntimeError, match="reset before stepping"):
            runtime.step(action)
        assert runtime.env.steps == 2
        assert runtime.observation == final.observation
        assert runtime.diagnostics() == final.diagnostics

    monkeypatch.setattr(
        "bench.env.robocasa_runtime.reset_env_with_initial_arm_qpos", lambda env: env.reset(),
    )
    assert runtime.reset() == {"step": 0}
    assert not runtime.diagnostics().stop_execution
    assert runtime.step(action).observation == {"step": 1}


def test_diagnostic_stop_also_blocks_the_next_action():
    runtime = _runtime(hard_stop=True, success=True)
    runtime.env.steps = 2
    assert runtime.diagnostics().stop_execution
    with pytest.raises(RuntimeError, match="reset before stepping"):
        runtime.step(np.zeros(13, dtype=np.float32))
    assert runtime.env.steps == 2


def test_candidate_success_without_hard_stop_allows_full_horizon_checks():
    runtime = _runtime(hard_stop=False, success=True)
    action = np.zeros(13, dtype=np.float32)
    runtime.step(action)
    candidate = runtime.step(action)
    assert candidate.terminal and candidate.diagnostics.success
    assert not candidate.diagnostics.stop_execution
    assert runtime.step(action).observation == {"step": 3}


def test_execution_stop_implies_a_terminal_transition():
    transition = RuntimeStep(
        observation={}, diagnostics=TaskDiagnostics(success=False, stop_execution=True),
    )
    assert transition.terminal
