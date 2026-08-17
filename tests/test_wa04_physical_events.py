"""Physical-event coverage for WA-04 replay and learned-policy evaluation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from bench.tasks.wa04.domain.contracts import EpisodePlan
from bench.tasks.wa04.domain.semantics import SemanticState, evaluate_success
from bench.tasks.wa04.environment import WA04ServeUniqueTemperatureCupAmongThree


def _bare_environment() -> WA04ServeUniqueTemperatureCupAmongThree:
    env = object.__new__(WA04ServeUniqueTemperatureCupAmongThree)
    temperatures = {"cup_A": "hot", "cup_B": "cold", "cup_C": "cold"}
    slots = {"cup_A": "slot_1", "cup_B": "slot_2", "cup_C": "slot_3"}
    env._episode_plan = EpisodePlan(
        seed=0,
        temperature_by_cup=temperatures,
        requested_temperature="hot",
        requested_cup_id="cup_A",
        fill_cup_order=("cup_A", "cup_B", "cup_C"),
        service_slot_by_cup=slots,
        max_horizon=8500,
    )
    env._semantic_state = SemanticState(
        temperature_by_cup=temperatures,
        requested_temperature="hot",
        requested_cup_id="cup_A",
        service_slot_by_cup=slots,
    )
    env.cup_temperatures = {cup: None for cup in temperatures}
    env.cup_filled = {cup: False for cup in temperatures}
    env.cup_fill_progress = {cup: 0.975 for cup in temperatures}
    env._sensor_cup_by_temperature = {"hot": None, "cold": None}
    env._selection_min_grasp_steps = 2
    env._reset_physical_event_tracking()
    env.set_cup_fill_progress = lambda cup, value: env.cup_fill_progress.__setitem__(
        cup, float(value)
    )
    env.cup_is_under_sensor_faucet = lambda cup, temperature: (
        env._sensor_cup_by_temperature.get(temperature) == cup
    )
    staged: set[str] = set()
    env._cup_physically_staged = lambda cup, slot: cup in staged
    env.are_cups_physically_staged = lambda: len(staged) == 3
    grasped: set[str] = set()
    env._cup_moved_from_staging = lambda cup: cup in grasped
    env._cup_physically_grasped = lambda cup: (
        cup in grasped and env._cup_moved_from_staging(cup)
    )
    env.object_position = lambda cup: np.asarray(
        [float(ord(cup[-1]) - ord("A")), 0.0, 1.0], dtype=float
    )
    env._test_staged = staged
    env._test_grasped = grasped
    return env


def test_wa04_physical_events_cover_all_three_cups_and_final_selection() -> None:
    env = _bare_environment()
    for cup, temperature in env._episode_plan.temperature_by_cup.items():
        env._sensor_cup_by_temperature = {"hot": None, "cold": None}
        env._sensor_cup_by_temperature[temperature] = cup
        env._update_task_events_from_physics()

    assert env._semantic_state.filled_temperature_by_cup == dict(
        env._episode_plan.temperature_by_cup
    )

    env._sensor_cup_by_temperature = {"hot": None, "cold": None}
    env._test_staged.update(("cup_A", "cup_B", "cup_C"))
    env._update_task_events_from_physics()
    assert env._semantic_state.staged_slot_by_cup == dict(
        env._episode_plan.service_slot_by_cup
    )
    assert env._semantic_state.cups_staged_before_final_pick

    env._test_grasped.add("cup_A")
    env._update_task_events_from_physics()
    env._update_task_events_from_physics()

    assert env._semantic_state.final_taken_cup_id == "cup_A"
    assert env._physical_event_diagnostics["selection_detected"] == "cup_A"
    assert evaluate_success(env._semantic_state).success


def test_wa04_physical_fill_rejects_wrong_temperature() -> None:
    env = _bare_environment()
    env._sensor_cup_by_temperature["cold"] = "cup_A"
    env._update_task_events_from_physics()

    assert "cup_A" not in env._semantic_state.filled_temperature_by_cup
    assert not env.cup_filled["cup_A"]


def test_wa04_final_selection_requires_pickup_motion() -> None:
    env = _bare_environment()
    for cup, temperature in env._episode_plan.temperature_by_cup.items():
        env._sensor_cup_by_temperature = {"hot": None, "cold": None}
        env._sensor_cup_by_temperature[temperature] = cup
        env._update_task_events_from_physics()
    env._sensor_cup_by_temperature = {"hot": None, "cold": None}
    env._test_staged.update(("cup_A", "cup_B", "cup_C"))
    env._update_task_events_from_physics()

    # Contact while the cup remains in its service slot is not a final pick.
    env._test_grasped.add("cup_A")
    env._cup_moved_from_staging = lambda cup: False
    env._update_task_events_from_physics()
    env._update_task_events_from_physics()
    assert env._semantic_state.final_taken_cup_id is None

    env._cup_moved_from_staging = lambda cup: cup == "cup_A"
    env._update_task_events_from_physics()
    env._update_task_events_from_physics()
    assert env._semantic_state.final_taken_cup_id == "cup_A"


def test_wa04_verified_event_wrappers_are_idempotent_but_not_permissive() -> None:
    env = _bare_environment()
    env.set_cup_fill_progress = lambda cup, value: env.cup_fill_progress.__setitem__(
        cup, float(value)
    )

    assert env.record_verified_fill("cup_A", "hot")
    assert env.record_verified_fill("cup_A", "hot")
    assert not env.record_verified_fill("cup_A", "cold")

    assert env.record_verified_staging("cup_A", "slot_1")
    assert env.record_verified_staging("cup_A", "slot_1")
    assert not env.record_verified_staging("cup_A", "slot_2")
