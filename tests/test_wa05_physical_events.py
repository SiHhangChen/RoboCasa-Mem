"""Physical-event coverage for WA-05 evaluation."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from bench.tasks.wa05.domain.contracts import EpisodePlan
from bench.tasks.wa05.domain.semantics import SemanticState
from bench.tasks.wa05.environment import (
    WA05FillOneCupAndMatchTemperatureButton,
)


def _bare_environment() -> WA05FillOneCupAndMatchTemperatureButton:
    env = object.__new__(WA05FillOneCupAndMatchTemperatureButton)
    env._episode_plan = EpisodePlan(
        seed=0,
        cup_id="cup_A",
        expert_fill_temperature="hot",
        service_slot="slot_1",
        max_horizon=100,
    )
    env._semantic_state = SemanticState()
    env.cup_fill_progress = {"cup_A": 0.0}
    env.cup_temperatures = {"cup_A": None}
    env.cup_filled = {"cup_A": False}
    env.fill_under_water_steps_required = 40
    env._sensor_cup_by_temperature = {"hot": "cup_A", "cold": None}
    env._sensor_faucet_affinity_by_cup = {}
    env._sensor_zone_entry_steps = {"hot": 2, "cold": 0}
    env._active_sensor_faucets = {"hot"}
    env._reset_physical_event_tracking()
    env.set_cup_fill_progress = (
        lambda cup, value: env.cup_fill_progress.__setitem__(cup, float(value))
    )
    env.cup_is_under_sensor_faucet = (
        lambda cup, temperature: (
            env._sensor_cup_by_temperature.get(temperature) == cup
        )
    )
    env._cup_physically_staged = lambda cup, slot: False
    return env


def test_first_verified_water_contact_qualifies_fill() -> None:
    env = _bare_environment()

    env._update_task_events_from_physics()

    assert env._semantic_state.filled_temperature == "hot"
    assert env.cup_filled["cup_A"]
    assert env.cup_temperatures["cup_A"] == "hot"
    assert env.cup_fill_progress["cup_A"] == pytest.approx(0.025)
    assert env._physical_event_diagnostics["fill_detected_by_cup"] == {
        "cup_A": "hot"
    }


def test_fill_requires_verified_active_water_contact() -> None:
    env = _bare_environment()
    env._sensor_cup_by_temperature = {"hot": None, "cold": None}

    env._update_task_events_from_physics()

    assert env._semantic_state.filled_temperature is None
    assert not env.cup_filled["cup_A"]
    assert env.cup_fill_progress["cup_A"] == 0.0


def test_liquid_animation_continues_after_semantic_fill() -> None:
    env = _bare_environment()

    env._update_task_events_from_physics()
    env._update_task_events_from_physics()

    assert env._semantic_state.filled_temperature == "hot"
    assert env.cup_fill_progress["cup_A"] == pytest.approx(0.05)


def test_later_other_temperature_remains_absorbing_failure() -> None:
    env = _bare_environment()
    env._update_task_events_from_physics()
    env._sensor_cup_by_temperature = {"hot": None, "cold": "cup_A"}

    env._update_task_events_from_physics()

    assert env._semantic_state.absorbing_failure
    assert env._semantic_state.failure_code == "mixed_temperature_fill"
    assert env._physical_event_diagnostics["mixed_fill_detected"] == {
        "cup_id": "cup_A",
        "first_temperature": "hot",
        "second_temperature": "cold",
    }
