"""Physical TS01 completion-button press primitive."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.execution.planning import matrix_from_base_obs, pose_from_matrix


PRESS_COMPLETION_BUTTON_OPERATION = "press_completion_button"


def press_completion_button(
    controller: Any,
    *,
    prepress_z_offset: float,
    press_z_offset: float,
    hold_steps: int,
    gripper_action: float,
    near_miss_position_tolerance: float,
    minimum_press_displacement_m: float,
    **_unused: Any,
) -> dict[str, Any]:
    """Press the live plunger and latch completion from physical evidence.

    The spring-loaded MuJoCo joint returns after retreat, so the environment
    stores a one-way event only after measured joint travel passes the configured
    threshold.  The sugar-count goal is checked before any completion event can
    be committed.
    """

    if int(hold_steps) < 1:
        return _failure("button_hold_steps_must_be_positive")
    if float(minimum_press_displacement_m) <= 0.0:
        return _failure("minimum_press_displacement_m_must_be_positive")
    if float(near_miss_position_tolerance) <= 0.0:
        return _failure("near_miss_position_tolerance_must_be_positive")
    if float(prepress_z_offset) <= float(press_z_offset):
        return _failure("prepress_target_must_be_above_press_target")
    if not bool(controller.runtime_config.execute):
        return {
            "primitive": PRESS_COMPLETION_BUTTON_OPERATION,
            "dry_run": True,
            "semantic_event_deferred": True,
            "checks": {"deferred_to_execute": True},
            "pass": True,
        }

    env = controller.env
    try:
        controller.refresh_obs()
        env.update_state()
        sugar_goal_ready = bool(env.sugar_goal_complete())
        button = np.asarray(env.completion_button_position(), dtype=float).reshape(3)
        displacement_before = float(env.completion_button_displacement())
        prepress_pose = _button_press_pose_world(
            controller.obs,
            button_position_world=button,
            z_offset=float(prepress_z_offset),
        )
        press_pose = _button_press_pose_world(
            controller.obs,
            button_position_world=button,
            z_offset=float(press_z_offset),
        )
    except Exception as exc:
        return _failure(
            "completion_button_target_unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )
    if not sugar_goal_ready:
        return {
            "primitive": PRESS_COMPLETION_BUTTON_OPERATION,
            "button_position_world": button.astype(float).tolist(),
            "checks": {"sugar_goal_complete_before_press": False},
            "pass": False,
            "reason": "sugar_goal_incomplete_before_completion_press",
        }

    prepress = _run_world_stage(
        controller,
        prepress_pose,
        stage="completion_button_prepress",
        gripper_action=float(gripper_action),
    )
    press = (
        _run_world_stage(
            controller,
            press_pose,
            stage="completion_button_press",
            gripper_action=float(gripper_action),
            record_trajectory=True,
        )
        if bool(prepress.get("pass", False))
        else _skipped("completion_button_press", "prepress_failed")
    )
    displacement_after_press = float(env.completion_button_displacement())
    press_eef = np.asarray(
        controller.stage_executor.current_eef_site_world(), dtype=float
    ).reshape(4, 4)
    endpoint_error = float(np.linalg.norm(press_eef[:3, 3] - press_pose[:3, 3]))
    press_endpoint = endpoint_error <= float(near_miss_position_tolerance)
    hold = (
        controller.run_gripper_command(float(gripper_action), int(hold_steps))
        if bool(press.get("pass", False))
        else _skipped("completion_button_hold", "press_failed")
    )
    displacement_after_hold = float(env.completion_button_displacement())
    peak_displacement = max(
        displacement_before,
        displacement_after_press,
        displacement_after_hold,
    )
    press_travel = max(0.0, peak_displacement - displacement_before)
    physical_motion = bool(press_travel >= float(minimum_press_displacement_m))
    hold_ok = _gripper_command_pass(hold, required_steps=int(hold_steps))
    motion_verified = bool(
        prepress.get("pass", False)
        and press.get("pass", False)
        and press_endpoint
        and hold_ok
        and physical_motion
    )
    completion_latched = bool(
        env.record_verified_completion_button_press(
            physical_motion_verified=motion_verified,
        )
    )
    retreat = (
        _run_world_stage(
            controller,
            prepress_pose,
            stage="completion_button_retreat",
            gripper_action=float(gripper_action),
        )
        if bool(press.get("pass", False)) and hold_ok
        else _skipped("completion_button_retreat", "press_or_hold_failed")
    )
    checks = {
        "sugar_goal_complete_before_press": sugar_goal_ready,
        "prepress": bool(prepress.get("pass", False)),
        "press": bool(press.get("pass", False)),
        "press_endpoint": bool(press_endpoint),
        "button_hold": bool(hold_ok),
        "physical_button_motion": physical_motion,
        "completion_event_committed": completion_latched,
        "retreat": bool(retreat.get("pass", False)),
    }
    passed = bool(all(checks.values()))
    return {
        "primitive": PRESS_COMPLETION_BUTTON_OPERATION,
        "button_position_world": button.astype(float).tolist(),
        "prepress_target_world": prepress_pose.astype(float).tolist(),
        "press_target_world": press_pose.astype(float).tolist(),
        "press_endpoint_world": press_eef[:3, 3].astype(float).tolist(),
        "press_endpoint_error_m": endpoint_error,
        "near_miss_position_tolerance": float(near_miss_position_tolerance),
        "button_displacement_m": {
            "before": displacement_before,
            "after_press": displacement_after_press,
            "after_hold": displacement_after_hold,
            "peak": peak_displacement,
            "press_travel_from_baseline": press_travel,
            "minimum_required": float(minimum_press_displacement_m),
        },
        "stages": {
            "prepress": _without_obs(prepress),
            "press": _without_obs(press),
            "hold": _without_obs(hold),
            "retreat": _without_obs(retreat),
        },
        "checks": checks,
        "pass": passed,
        "reason": None if passed else "completion_button_operation_failed",
    }


def _button_press_pose_world(
    obs: Mapping[str, Any],
    *,
    button_position_world: Any,
    z_offset: float,
) -> np.ndarray:
    position = np.asarray(button_position_world, dtype=float).reshape(3)
    if not np.all(np.isfinite(position)):
        raise ValueError("completion button position must be finite")
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = matrix_from_base_obs(dict(obs))[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = position + np.asarray([0.0, 0.0, float(z_offset)], dtype=float)
    return pose


def _run_world_stage(
    controller: Any,
    pose_world: np.ndarray,
    *,
    stage: str,
    gripper_action: float,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    base_world = matrix_from_base_obs(dict(controller.obs))
    target_base = pose_from_matrix(
        np.linalg.inv(base_world) @ np.asarray(pose_world, dtype=float)
    )
    return controller.run_planned_stage(
        target_base,
        stage=stage,
        gripper_action=float(gripper_action),
        record_trajectory=bool(record_trajectory),
    )


def _gripper_command_pass(result: dict[str, Any], *, required_steps: int) -> bool:
    if bool(result.get("dry_run", False)):
        return True
    return int(result.get("steps", 0)) >= int(required_steps)


def _skipped(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "skipped": True, "pass": False, "reason": reason}


def _without_obs(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "obs"}


def _failure(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "primitive": PRESS_COMPLETION_BUTTON_OPERATION,
        "pass": False,
        "reason": reason,
        **details,
    }


__all__ = ["PRESS_COMPLETION_BUTTON_OPERATION", "press_completion_button"]
