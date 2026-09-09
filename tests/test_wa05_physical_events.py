"""Physical-event coverage for WA-05 evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from bench.tasks.wa05.domain.contracts import EpisodePlan
from bench.tasks.wa05.domain.semantics import SemanticState
from bench.tasks.wa05.environment import (
    WA05FillOneCupAndMatchTemperatureButton,
)
from bench.tasks.wa05.execution import handlers as wa05_handlers


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


def _placement_environment(
    monkeypatch,
    *,
    cup_xy: tuple[float, float],
    counter_contact: bool,
    tilt_deg: float,
    gripper_distance: float,
) -> WA05FillOneCupAndMatchTemperatureButton:
    from robocasa.utils import object_utils as OU

    env = _bare_environment()
    env._update_task_events_from_physics()
    env._sensor_cup_by_temperature = {"hot": None, "cold": None}
    del env._cup_physically_staged
    position = np.array([*cup_xy, 1.0])
    angle = np.deg2rad(tilt_deg)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            body_xpos=np.array([position]),
            body_xmat=np.array([rotation.reshape(-1)]),
            site_xpos=np.array([position + [0.0, 0.0, gripper_distance]]),
        ),
    )
    env.obj_body_id = {"cup_A": 0}
    env.robots = [SimpleNamespace(
        eef_site_id={"right": 0},
        gripper={"right": SimpleNamespace(joints=[], current_action=np.array([-1.0]))},
    )]
    env.counter = object()
    env.cup_upright_threshold_deg = 15.0
    env.service_threshold = 0.12
    env.service_slot_poses = {"slot_1": [0.0, 0.0, 0.95]}
    env.service_slot_regions = {
        "slot_1": {"x_bounds": [-0.06, 0.06], "y_bounds": [-0.06, 0.06]},
    }
    monkeypatch.setattr(
        OU, "check_obj_fixture_contact",
        lambda actual_env, cup, counter: counter_contact,
    )
    monkeypatch.setattr(OU, "get_fixture_to_point_rel_offset", lambda counter, point: point)
    return env


@pytest.mark.parametrize("verification_path", ("policy_evaluation", "expert_event", "expert_handler"))
@pytest.mark.parametrize(
    "cup_xy,counter_contact,tilt_deg,gripper_distance,expected_evaluation,expected_expert",
    [
        ((0.02205, -0.073619), True, 0.089, 0.5526, True, False),
        ((0.17466, 0.02798), True, 0.09, 0.6629, True, False),
        ((0.0, 0.0), True, 86.69, 0.55, False, False),
        ((0.0, 0.0), False, 0.09, 1.04, False, False),
        ((0.0, 0.0), True, 0.09, 0.02, False, False),
        ((0.0, 0.0), True, 15.1, 0.55, False, False),
        ((0.0, 0.0), True, 14.9, 0.121, True, True),
        ((0.0, 0.0), True, 14.9, 0.119, False, False),
    ],
    ids=[
        "seed1076-outside-rectangle", "beyond-service-distance", "tipped-cup",
        "sink-or-airborne", "still-held", "tilt-over-limit",
        "upright-and-released", "gripper-not-retreated",
    ],
)
def test_staging_requires_upright_counter_placement_and_release(
    monkeypatch, verification_path, cup_xy, counter_contact, tilt_deg,
    gripper_distance, expected_evaluation, expected_expert,
) -> None:
    env = _placement_environment(
        monkeypatch,
        cup_xy=cup_xy,
        counter_contact=counter_contact,
        tilt_deg=tilt_deg,
        gripper_distance=gripper_distance,
    )
    env.require_service_region_for_staging = verification_path != "policy_evaluation"
    expected = expected_expert if env.require_service_region_for_staging else expected_evaluation
    if verification_path != "expert_handler":
        env._update_task_events_from_physics()
    else:
        monkeypatch.setattr(wa05_handlers, "_refresh", lambda controller: env)
        monkeypatch.setattr(wa05_handlers, "get_object_hold_status", lambda *args, **kwargs: {})
        controller = SimpleNamespace(env=env, runtime_config=SimpleNamespace(execute=True))
        handler = wa05_handlers.build_stage_handlers(controller)["record_verified_staging"]
        result = handler("cup_A", service_slot="slot_1")
        assert result["pass"] is expected

    assert env.are_cups_physically_staged() is expected
    assert env._semantic_state.staged_before_answer is expected
