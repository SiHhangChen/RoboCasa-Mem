"""Generic object hold, release, and transport helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def get_object_world_position(env: Any, object_name: str) -> np.ndarray:
    """Return a task object's current world-frame position."""
    return np.array(
        env.sim.data.body_xpos[env.obj_body_id[object_name]],
        dtype=float,
    )


def get_object_hold_status(
    env: Any,
    object_name: str,
    *,
    max_end_effector_distance: float = 0.18,
    contact_checker: Callable[..., bool] | None = None,
) -> dict[str, Any]:
    """Inspect a hold, optionally reusing contact evidence for this sim state."""
    from robocasa.utils import object_utils as OU

    grasp_status: dict[str, Any] = {}
    contact_error = None
    joint_positions = None
    try:
        grasp_status["contact"] = bool(
            (contact_checker or env.check_contact)(
                env.robots[0].gripper["right"],
                env.objects[object_name],
            )
        )
    except Exception as exc:
        contact_error = exc
        grasp_status["contact_error"] = f"{type(exc).__name__}: {exc}"

    for threshold in (0.035, 0.6):
        key = f"grasped_threshold_{threshold:g}"
        try:
            if contact_checker is None:
                grasp_status[key] = bool(
                    OU.check_obj_grasped(env, object_name, threshold=threshold)
                )
            else:
                # Preserve RoboCasa's signed joint threshold, not WR08's abs check.
                if joint_positions is None:
                    joint_positions = [
                        env.sim.data.qpos[env.sim.model.get_joint_qpos_addr(joint)]
                        for joint in ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")
                    ]
                if contact_error is not None:
                    raise contact_error
                grasp_status[key] = bool(
                    grasp_status["contact"] and all(pos < threshold for pos in joint_positions)
                )
        except Exception as exc:
            grasp_status[f"{key}_error"] = f"{type(exc).__name__}: {exc}"

    object_position = get_object_world_position(env, object_name)
    end_effector_position = np.array(
        env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]],
        dtype=float,
    )
    end_effector_distance = float(
        np.linalg.norm(object_position - end_effector_position)
    )
    attached = bool(
        grasp_status.get("contact", False)
        or grasp_status.get("grasped_threshold_0.6", False)
        or grasp_status.get("grasped_threshold_0.035", False)
    )
    checks = {
        "finite_pose": bool(
            np.all(np.isfinite(object_position))
            and np.all(np.isfinite(end_effector_position))
        ),
        "grasp_or_contact": attached,
        "near_end_effector": (
            end_effector_distance <= float(max_end_effector_distance)
        ),
    }
    return {
        "object_name": object_name,
        "contact_and_grasp": grasp_status,
        "object_position": object_position.astype(float).tolist(),
        "end_effector_position": end_effector_position.astype(float).tolist(),
        "end_effector_distance": end_effector_distance,
        "max_end_effector_distance": float(max_end_effector_distance),
        "checks": checks,
        "held": bool(all(checks.values())),
    }


def get_object_transport_snapshot(
    env: Any,
    object_name: str,
) -> dict[str, Any]:
    """Capture the object-to-EEF translation used to verify carried motion."""

    object_position = get_object_world_position(env, object_name)
    end_effector_position = np.array(
        env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]],
        dtype=float,
    )
    offset = object_position - end_effector_position
    finite = bool(
        np.all(np.isfinite(object_position))
        and np.all(np.isfinite(end_effector_position))
        and np.all(np.isfinite(offset))
    )
    return {
        "object_name": str(object_name),
        "object_position": object_position.astype(float).tolist(),
        "end_effector_position": end_effector_position.astype(float).tolist(),
        "object_from_end_effector": offset.astype(float).tolist(),
        "finite": finite,
    }


def evaluate_object_transport(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    minimum_end_effector_motion: float = 0.0,
    minimum_motion_ratio: float = 0.7,
    max_relative_offset_drift: float = 0.02,
) -> dict[str, Any]:
    """Verify that an object followed an EEF translation instead of staying behind."""

    if minimum_end_effector_motion < 0.0:
        raise ValueError("minimum_end_effector_motion must be non-negative")
    if not 0.0 < minimum_motion_ratio <= 1.0:
        raise ValueError("minimum_motion_ratio must be in (0, 1]")
    if max_relative_offset_drift <= 0.0:
        raise ValueError("max_relative_offset_drift must be positive")

    try:
        object_before = _finite_vec3(before["object_position"])
        object_after = _finite_vec3(after["object_position"])
        eef_before = _finite_vec3(before["end_effector_position"])
        eef_after = _finite_vec3(after["end_effector_position"])
        offset_before = _finite_vec3(before["object_from_end_effector"])
        offset_after = _finite_vec3(after["object_from_end_effector"])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "pass": False,
            "reason": f"invalid_transport_snapshot:{type(exc).__name__}:{exc}",
            "before": dict(before),
            "after": dict(after),
            "checks": {"finite_snapshots": False},
        }

    object_delta = object_after - object_before
    eef_delta = eef_after - eef_before
    object_motion = float(np.linalg.norm(object_delta))
    end_effector_motion = float(np.linalg.norm(eef_delta))
    relative_offset_drift = float(np.linalg.norm(offset_after - offset_before))
    motion_ratio = (
        1.0
        if end_effector_motion <= 1e-9 and object_motion <= max_relative_offset_drift
        else object_motion / max(end_effector_motion, 1e-9)
    )
    checks = {
        "finite_snapshots": True,
        "end_effector_moved": (
            end_effector_motion + 1e-9 >= float(minimum_end_effector_motion)
        ),
        "object_followed": motion_ratio + 1e-9 >= float(minimum_motion_ratio),
        "relative_offset_stable": (
            relative_offset_drift <= float(max_relative_offset_drift)
        ),
    }
    return {
        "object_name": str(before.get("object_name", after.get("object_name", ""))),
        "before": dict(before),
        "after": dict(after),
        "object_delta": object_delta.astype(float).tolist(),
        "end_effector_delta": eef_delta.astype(float).tolist(),
        "object_motion": object_motion,
        "end_effector_motion": end_effector_motion,
        "motion_ratio": motion_ratio,
        "minimum_end_effector_motion": float(minimum_end_effector_motion),
        "minimum_motion_ratio": float(minimum_motion_ratio),
        "relative_offset_drift": relative_offset_drift,
        "max_relative_offset_drift": float(max_relative_offset_drift),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def is_object_released(
    env: Any,
    object_name: str,
    *,
    distance: float = 0.12,
) -> bool:
    """Return whether the right gripper is sufficiently far from an object."""
    from robocasa.utils import object_utils as OU

    return bool(OU.gripper_obj_far(env, object_name, th=float(distance)))


def validate_gripper_force_limit(
    value: float,
    name: str = "gripper force limit",
) -> None:
    """Reject values that cannot be written to the MuJoCo actuator force range."""
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


def configure_right_gripper_force_limit(
    env: Any,
    force_limit: float,
) -> dict[str, Any]:
    """Limit the right-gripper per-finger actuator force without editing shared assets.

    Writes the per-episode MuJoCo model in place and returns the previous force
    ranges so callers can restore them after the fragile object is released.
    """
    validate_gripper_force_limit(force_limit)
    model = env.sim.model
    gripper = env.robots[0].gripper["right"]
    actuator_names = list(gripper.actuators)
    before: dict[str, list[float]] = {}
    for actuator_name in actuator_names:
        actuator_id = int(model.actuator_name2id(actuator_name))
        before[str(actuator_name)] = [
            float(value) for value in model.actuator_forcerange[actuator_id]
        ]
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = [
            -float(force_limit),
            float(force_limit),
        ]
    return {
        "actuator_names": [str(name) for name in actuator_names],
        "before": before,
        "force_limit": float(force_limit),
    }


def lift_held_object_for_transport(
    controller: Any,
    object_name: str,
    *,
    stage: str = "lift_held_object_for_transport",
    lift_height: float = 0.18,
    settle_steps: int = 80,
) -> dict[str, Any]:
    """Lift a held object vertically without changing its orientation."""
    from bench.execution.planning import (
        matrix_from_base_obs,
        matrix_from_site_pose,
        pose_from_matrix,
    )

    env = controller.env
    obs = controller.refresh_obs()
    end_effector_world = matrix_from_site_pose(env, env.robots[0])
    target_world = np.array(end_effector_world, dtype=float, copy=True)
    target_world[2, 3] += float(lift_height)
    target_base = np.linalg.inv(matrix_from_base_obs(obs)) @ target_world
    result = controller.run_planned_stage(
        pose_from_matrix(target_base),
        stage=stage,
        gripper_action=1.0,
        holding_object=object_name,
        settle_steps=int(settle_steps),
    )
    result.update(
        {
            "primitive": "lift_held_object_for_transport",
            "object_name": object_name,
            "lift_height": float(lift_height),
            "settle_steps": int(settle_steps),
            "target_end_effector_world": pose_from_matrix(target_world),
        }
    )
    return result


def _finite_vec3(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError("expected three finite values")
    return array


__all__ = [
    "configure_right_gripper_force_limit",
    "evaluate_object_transport",
    "get_object_hold_status",
    "get_object_transport_snapshot",
    "get_object_world_position",
    "is_object_released",
    "lift_held_object_for_transport",
    "validate_gripper_force_limit",
]
