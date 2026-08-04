"""Explicit Cartesian arm-height transition between WR-01 cabinet levels."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.tasks.wr03.execution.calibration import target_in_current_base, without_obs


def lower_arm_z(
    controller: Any,
    *,
    delta_m: float,
    position_tolerance_m: float,
    gripper_action: float,
) -> dict[str, Any]:
    """Plan and execute a world-z descent while preserving EEF orientation."""

    requested_delta = float(delta_m)
    tolerance = float(position_tolerance_m)
    if requested_delta <= 0.0 or not np.isfinite(requested_delta):
        raise ValueError("arm z lowering delta must be positive and finite")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("arm z position tolerance must be non-negative and finite")

    controller.refresh_obs()
    initial_world = np.asarray(
        controller.stage_executor.current_eef_site_world(),
        dtype=float,
    ).reshape(4, 4)
    target_world = initial_world.copy()
    target_world[2, 3] -= requested_delta
    target_base = target_in_current_base(controller, target_world)
    stage = controller.run_planned_stage(
        target_base,
        stage="lower_arm_z_20cm_for_lower_cabinet",
        gripper_action=float(gripper_action),
        execute=bool(controller.runtime_config.execute),
    )
    controller.refresh_obs()
    final_world = np.asarray(
        controller.stage_executor.current_eef_site_world(),
        dtype=float,
    ).reshape(4, 4)
    actual_delta = float(initial_world[2, 3] - final_world[2, 3])
    checks = {
        "planned_and_executed": bool(stage.get("pass", False)),
        "requested_z_delta_reached": abs(actual_delta - requested_delta) <= tolerance,
    }
    return {
        "primitive": "lower_arm_z",
        "coordinate_frame": "world",
        "initial_eef_z_m": float(initial_world[2, 3]),
        "requested_delta_m": requested_delta,
        "target_eef_z_m": float(target_world[2, 3]),
        "final_eef_z_m": float(final_world[2, 3]),
        "actual_delta_m": actual_delta,
        "position_tolerance_m": tolerance,
        "gripper_action": float(gripper_action),
        "stage": without_obs(stage),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": None if all(checks.values()) else "arm_z_lowering_failed",
    }


__all__ = ["lower_arm_z"]
