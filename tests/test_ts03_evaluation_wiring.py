"""Wiring tests for the TS03 task evaluator.

These cover the two integration seams introduced when the TS03 runtime started
evaluating through ``TS03Evaluator``: the environment's confirmed-sugar
debounce feeding ``evaluation_snapshot()``, and the runtime's ``_initialize``
mounting a fresh evaluator on each episode.
"""

from __future__ import annotations

import bench.tasks.ts03.environment as bench_env
import bench.tasks.ts03.runtime as ts03_runtime
from bench.tasks.ts03.domain.semantics import PhysicalState


def _bare_env(sugar_names: tuple[str, ...], dwell: int):
    """A TS03 environment instance with only the debounce fields installed."""

    env = object.__new__(bench_env.TS03HierarchicalAddKSugarToHotChocolate)
    env.sugar_names = tuple(sugar_names)
    env.confirmed_sugar_dwell_steps = int(dwell)
    env._sugar_confirmed = {name: False for name in env.sugar_names}
    env._sugar_dwell = {name: 0 for name in env.sugar_names}
    env._last_confirmed_update_step = None
    env.confirmed_sugar_count = 0
    env.target_sugar_count = 1
    env.objects_released = True
    env.minimum_press_displacement_m = 0.004
    env._completion_button_press_latched = False
    env._completion_button_peak_displacement_m = 0.0
    env._completion_button_peak_step = None
    env._completion_button_threshold_active = False
    env._sugar_goal_complete_observed = False
    env.timestep = 0
    return env


def test_environment_kwargs_include_button_threshold() -> None:
    class _Config:
        operations = {"completion_button": {"minimum_press_displacement_m": 0.007}}

    class _Scene:
        sugar_plate_name = "plate"
        receptacle_name = "pan"

    class _Plan:
        target_sugar_count = 1
        total_sugar_count = 2
        sugar_names = ("sugar0", "sugar1")
        target_sugar_names = ("sugar0",)

    values = ts03_runtime._environment(
        _Config(),
        _Scene(),
        _Plan(),
        {"completion_button_position": "button.json"},
    )
    assert values["minimum_press_displacement_m"] == 0.007


class _GripperFarStub:
    def __init__(self, far: bool) -> None:
        self._far = bool(far)

    def gripper_obj_far(self, env: object, name: str, th: float) -> bool:
        return self._far


def _physical(sugars: dict[str, bool]) -> PhysicalState:
    return PhysicalState(sugars_in_saucepan=dict(sugars))


def test_confirmed_sugar_count_debounces(monkeypatch) -> None:
    env = _bare_env(("sugar0", "sugar1"), dwell=3)
    monkeypatch.setattr(bench_env, "OU", _GripperFarStub(far=True))
    for _ in range(2):
        env._update_confirmed_sugars(
            _physical({"sugar0": True, "sugar1": False})
        )
    assert env.confirmed_sugar_count == 0
    env._update_confirmed_sugars(_physical({"sugar0": True, "sugar1": False}))
    assert env.confirmed_sugar_count == 1
    assert env._sugar_confirmed == {"sugar0": True, "sugar1": False}


def test_confirmed_sugar_is_unconfirmed_when_cube_leaves(monkeypatch) -> None:
    env = _bare_env(("sugar0",), dwell=2)
    monkeypatch.setattr(bench_env, "OU", _GripperFarStub(far=True))
    env._update_confirmed_sugars(_physical({"sugar0": True}))
    env._update_confirmed_sugars(_physical({"sugar0": True}))
    assert env.confirmed_sugar_count == 1
    env._update_confirmed_sugars(_physical({"sugar0": False}))
    assert env.confirmed_sugar_count == 0
    assert env._sugar_dwell == {"sugar0": 0}


def test_confirmed_dwell_is_updated_once_per_simulator_step(monkeypatch) -> None:
    env = _bare_env(("sugar0",), dwell=5)
    physical = _physical({"sugar0": True})
    calls = {"count": 0}

    def _counted_update(value: PhysicalState) -> None:
        calls["count"] += 1

    monkeypatch.setattr(env, "_update_confirmed_sugars", _counted_update)
    env._update_task_state(physical)
    env._update_task_state(physical)

    assert calls["count"] == 1


def test_held_cube_is_not_confirmed(monkeypatch) -> None:
    env = _bare_env(("sugar0",), dwell=1)
    monkeypatch.setattr(bench_env, "OU", _GripperFarStub(far=False))
    env._update_confirmed_sugars(_physical({"sugar0": True}))
    assert env.confirmed_sugar_count == 0


def test_free_policy_button_press_latches_after_confirmed_sugar() -> None:
    env = _bare_env(("sugar0",), dwell=1)
    displacement = {"value": 0.0}
    env.completion_button_displacement = lambda: displacement["value"]

    env.confirmed_sugar_count = 1
    env.timestep = 10
    env._update_completion_button_state()
    assert env._sugar_goal_complete_observed

    displacement["value"] = 0.006
    env.timestep = 11
    env._update_completion_button_state()
    assert env._completion_button_press_latched
    assert env._completion_button_peak_displacement_m == 0.006
    assert env._completion_button_peak_step == 11


def test_button_pressed_before_sugar_requires_release_and_repress() -> None:
    env = _bare_env(("sugar0",), dwell=1)
    displacement = {"value": 0.006}
    env.completion_button_displacement = lambda: displacement["value"]

    env.timestep = 1
    env._update_completion_button_state()
    assert not env._completion_button_press_latched

    env.confirmed_sugar_count = 1
    env.timestep = 2
    env._update_completion_button_state()
    assert env._sugar_goal_complete_observed
    assert not env._completion_button_press_latched

    displacement["value"] = 0.0
    env.timestep = 3
    env._update_completion_button_state()
    displacement["value"] = 0.005
    env.timestep = 4
    env._update_completion_button_state()
    assert env._completion_button_press_latched


def test_runtime_initialize_mounts_fresh_evaluator(monkeypatch) -> None:
    calls = {"reset": 0, "snapshot": None}

    class _FakeEvaluator:
        def __init__(self, **kwargs: object) -> None:
            calls["kwargs"] = dict(kwargs)

        def reset(self, snapshot: object) -> None:
            calls["reset"] += 1
            calls["snapshot"] = snapshot

    monkeypatch.setattr(ts03_runtime, "TS03Evaluator", _FakeEvaluator)
    monkeypatch.setattr(
        ts03_runtime,
        "_initialize_sugar",
        lambda descriptor, env, observation: ("updated", "initialization", {}, "targets"),
    )

    class _FakeEnv:
        profile_id = "P2"
        target_sugar_count = 2
        total_sugar_count = 2

        def __init__(self) -> None:
            self._ts03_evaluator = None

        def evaluation_snapshot(self) -> str:
            return "SNAPSHOT"

    env = _FakeEnv()
    updated, initialization, metadata, targets = ts03_runtime._initialize(
        object(), env, None
    )
    assert updated == "updated"
    assert initialization == "initialization"
    assert targets == "targets"
    assert metadata["task_evaluator"] is env._ts03_evaluator
    assert calls["reset"] == 1
    assert calls["snapshot"] == "SNAPSHOT"
    assert calls["kwargs"] == {
        "profile_id": "P2",
        "target_sugar_count": 2,
        "total_sugar_count": 2,
    }


def test_runtime_registers_public_report_stage_projector() -> None:
    assert ts03_runtime.TASK_RUNTIME_API.report_stage_profile_id == "TS03/report-v2"
    assert callable(ts03_runtime.TASK_RUNTIME_API.report_stages)
