"""Regression coverage for WA-02's held-cup sink-exit controller."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.wa02.execution.handlers import (
    _sink_exit_translation_profile,
    lift_held_cup_after_sink_exit,
)


def test_sink_exit_profile_removes_the_legacy_low_speed_tail() -> None:
    env = SimpleNamespace(control_timestep=0.05)
    profile = _sink_exit_translation_profile(
        env,
        enabled=True,
        min_action=0.025,
        max_action=0.35,
        acceleration=1.6,
        deceleration=1.2,
    )
    assert profile is not None

    remaining_x = 0.421
    target_overshoot = 0.005
    route_local_y_per_x = 0.02583025830258303
    command_norms: list[float] = []
    for _ in range(200):
        if remaining_x <= 0.0:
            break
        control_remaining_x = remaining_x + target_overshoot
        controller_error = np.asarray(
            [
                -control_remaining_x,
                -control_remaining_x * route_local_y_per_x,
            ],
            dtype=float,
        )
        command = profile.step_action(controller_error, tolerance=0.0)
        command_norms.append(float(np.linalg.norm(command)))
        remaining_x -= abs(float(command[0])) * env.control_timestep

    assert remaining_x <= 0.0
    assert -remaining_x < 0.020
    assert len(command_norms) < 50
    assert max(command_norms) <= 0.35 + 1.0e-9
    assert command_norms[0] == pytest.approx(1.6 * env.control_timestep)


def test_sink_exit_profile_can_fall_back_to_legacy_control() -> None:
    env = SimpleNamespace(control_timestep=0.05)

    assert (
        _sink_exit_translation_profile(
            env,
            enabled=False,
            min_action=0.025,
            max_action=0.35,
            acceleration=1.6,
            deceleration=1.2,
        )
        is None
    )


def test_sink_exit_defaults_to_profile_without_changing_task_semantics() -> None:
    controller = SimpleNamespace(runtime_config=SimpleNamespace(execute=False))

    result = lift_held_cup_after_sink_exit(
        controller,
        cup_id="cup_A",
        sink_exterior_local_x_min=-0.4724099655124329,
    )

    assert result["pass"] is True
    assert result["boundary_control_mode"] == "velocity_profile_xy"
    assert result["strategy"] == "cup_crosses_fixed_sink_local_x_then_vertical_lift"
    assert result["physical_validation"] == {"deferred_to_execute": True}


@pytest.mark.parametrize(
    "override",
    [
        {"boundary_profile_acceleration": 0.0},
        {"boundary_profile_deceleration": 0.0},
        {"boundary_profile_target_overshoot": 0.0},
        {"boundary_profile_target_overshoot": 0.020},
    ],
)
def test_sink_exit_rejects_invalid_profile_rates(override: dict[str, float]) -> None:
    controller = SimpleNamespace(runtime_config=SimpleNamespace(execute=False))

    with pytest.raises(ValueError, match="velocity-profile"):
        lift_held_cup_after_sink_exit(
            controller,
            cup_id="cup_A",
            sink_exterior_local_x_min=-0.4724099655124329,
            **override,
        )
