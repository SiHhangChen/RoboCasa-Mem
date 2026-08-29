from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from bench.tasks.wa02.domain.contracts import EpisodePlan
from bench.tasks.wa02.domain.semantics import SemanticState
from bench.tasks.wa02.environment import WA02ServeHotColdWaterCups


def _bare_environment() -> WA02ServeHotColdWaterCups:
    env = object.__new__(WA02ServeHotColdWaterCups)
    env._episode_plan = EpisodePlan(
        seed=0,
        hot_cup_id="cup_A",
        cold_cup_id="cup_B",
        requested_temperature="hot",
        requested_cup_id="cup_A",
        fill_temperature_order=("hot", "cold"),
        service_slot_by_cup={"cup_A": "slot_1", "cup_B": "slot_2"},
        max_horizon=100,
    )
    env._semantic_state = SemanticState(
        hot_cup_id="cup_A",
        cold_cup_id="cup_B",
        requested_temperature="hot",
        requested_cup_id="cup_A",
        service_slot_by_cup={"cup_A": "slot_1", "cup_B": "slot_2"},
    )
    env._pickup_min_grasp_steps = 3
    env._pickup_min_lift = 0.04
    env._selection_min_grasp_steps = 2
    env._reset_physical_event_tracking()
    env.timestep = 0
    env._sensor_cup_by_temperature = {"hot": None, "cold": None}
    env.cup_fill_progress = {"cup_A": 0.0, "cup_B": 0.0}
    env.cup_temperatures = {"cup_A": None, "cup_B": None}
    env.cup_filled = {"cup_A": False, "cup_B": False}
    env.service_slot_poses = {"slot_1": [0.0, 0.0, 0.0], "slot_2": [1.0, 0.0, 0.0]}
    env.cup_is_at_service_slot = lambda cup, slot: False
    env._cup_physically_staged = lambda cup, slot: False
    env.cup_is_under_sensor_faucet = lambda cup, temperature: False
    return env


def test_wa02_pickup_requires_stable_grasp_and_lift() -> None:
    env = _bare_environment()
    grasped = {"cup_A": True, "cup_B": False}
    positions = {
        "cup_A": np.asarray([0.0, 0.0, 1.0]),
        "cup_B": np.asarray([0.2, 0.0, 1.0]),
    }
    env._cup_physically_grasped = lambda cup: grasped[cup]
    env.object_position = lambda cup: positions[cup]

    for step, height in enumerate((1.0, 1.02), start=1):
        env.timestep = step
        positions["cup_A"] = np.asarray([0.0, 0.0, height])
        env._update_task_events_from_physics()
    assert "cup_A" not in env._physical_event_diagnostics["pickup_detected_by_cup"]

    env.timestep = 3
    positions["cup_A"] = np.asarray([0.0, 0.0, 1.05])
    env._update_task_events_from_physics()

    pickup = env._physical_event_diagnostics["pickup_detected_by_cup"]["cup_A"]
    assert pickup["step"] == 3
    assert pickup["lift_m"] == pytest.approx(0.05)


def test_wa02_staging_requires_release() -> None:
    env = _bare_environment()
    positions = {
        "cup_A": np.asarray([0.0, 0.0, 1.0]),
        "cup_B": np.asarray([0.2, 0.0, 1.0]),
    }
    held = {"cup_A": True, "cup_B": False}
    env.object_position = lambda cup: positions[cup]
    env._cup_physically_grasped = lambda cup: held[cup]
    env.detected_service_slot = lambda cup: "slot_2" if cup == "cup_A" else None

    env._update_task_events_from_physics()
    assert "cup_A" not in env._semantic_state.staged_slot_by_cup

    held["cup_A"] = False
    env.timestep += 1
    env._update_task_events_from_physics()
    assert env._semantic_state.staged_slot_by_cup == {"cup_A": "slot_2"}


def test_wa02_physical_staging_accepts_swapped_slot_assignment() -> None:
    env = _bare_environment()
    actual = {"cup_A": "slot_2", "cup_B": "slot_1"}
    env.cup_is_at_service_slot = lambda cup, slot: actual[cup] == slot

    assert env.are_cups_physically_staged()


def test_wa02_selection_requires_release_then_sustained_regrasp() -> None:
    env = _bare_environment()
    env._semantic_state.staged_slot_by_cup = {
        "cup_A": "slot_1",
        "cup_B": "slot_2",
    }
    env.are_cups_physically_staged = lambda: True
    held = {"cup_A": False, "cup_B": False}
    env._cup_physically_grasped = lambda cup: held[cup]
    env.object_position = lambda cup: np.asarray([0.0, 0.0, 1.0])

    env._update_task_events_from_physics()
    assert all(env._selection_armed_by_cup.values())

    held["cup_A"] = True
    env.timestep += 1
    env._update_task_events_from_physics()
    assert env._semantic_state.final_taken_cup_id is None

    env.timestep += 1
    env._update_task_events_from_physics()
    assert env._semantic_state.final_taken_cup_id == "cup_A"
    assert env._physical_event_diagnostics["selection_detected"] == "cup_A"


def test_wa02_physical_event_updates_are_idempotent_within_one_step() -> None:
    env = _bare_environment()
    env._pickup_min_grasp_steps = 2
    env._pickup_min_lift = 0.0
    env._cup_physically_grasped = lambda cup: cup == "cup_A"
    env.object_position = lambda cup: np.asarray([0.0, 0.0, 1.0])

    env.timestep = 1
    env._update_task_events_from_physics()
    env._update_task_events_from_physics()
    assert env._grasp_steps_by_cup["cup_A"] == 1
    assert "cup_A" not in env._physical_event_diagnostics["pickup_detected_by_cup"]

    env.timestep = 2
    env._update_task_events_from_physics()
    assert env._physical_event_diagnostics["pickup_detected_by_cup"]["cup_A"][
        "step"
    ] == 2
