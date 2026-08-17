"""WR-06 stage handlers with bottle-safe supported-release gating.

Both the plate staging place and the shelf return place run through the
guarded supported-release path: the descent trajectory can early-stop once the
bottle bottom enters its support guard, and opening the gripper is gated on a
physical pre-release check of the live bottle bottom, XY, and upright pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import qpos_for_controller
from bench.execution.planning import (
    normalize_placement_reference,
    object_reference_pos,
)
from bench.execution.planning.pick_place_targets import upright_object_rotation
from bench.tasks.wr06.execution.geometry import accessible_object_pos_world


def build_operation_handlers(
    controller: Any,
    *,
    dry_run_only: bool = True,
) -> dict[str, Any]:
    """Return WR-06 handlers for explicit pick-and-place targets."""

    def place_at_accessible_handler(target: str, **kwargs: Any) -> dict[str, Any]:
        return place_at_accessible(
            controller,
            object_name=str(target),
            dry_run_only=bool(kwargs.pop("dry_run_only", dry_run_only)),
            **kwargs,
        )

    def set_torso_height_handler(target: str, **kwargs: Any) -> dict[str, Any]:
        return set_torso_height(
            controller,
            holding_object=str(target),
            **kwargs,
        )

    def latch_memory_handler(target: str, **kwargs: Any) -> dict[str, Any]:
        del target, kwargs
        env = controller.env
        passed = bool(env.latch_original_level_hidden())
        return {
            "primitive": "latch_original_level_hidden",
            "pass": passed,
            "checks": {"memory_barrier_latched": passed},
            "reason": None if passed else "memory_barrier_not_ready",
        }

    def return_item_handler(target: str, **kwargs: Any) -> dict[str, Any]:
        object_name = str(target)
        env = controller.env
        level = str(kwargs.pop("return_level", getattr(env, "demo_original_level", "")))
        target_pos = kwargs.pop("target_pos_world", None)
        if target_pos is None:
            target_pos = env.return_region_pose(level)
        result = place_at_accessible(
            controller,
            object_name=object_name,
            target_pos_world=target_pos,
            target_name=str(kwargs.pop("target_name", f"{object_name}_return_shelf_region")),
            **kwargs,
        )
        if bool(result.get("pass", False)):
            result["return_verification_mode"] = "physical_supported_release"
            result["return_completion_pending"] = True
        return result

    return {
        "place_at_accessible": place_at_accessible_handler,
        "set_torso_height": set_torso_height_handler,
        "latch_original_level_hidden": latch_memory_handler,
        "return_item_to_level": return_item_handler,
    }


def place_at_accessible(
    controller: Any,
    *,
    object_name: str,
    target_pos: Any | None = None,
    target_pos_world: Any | None = None,
    target_name: str | None = None,
    target_max_base_x: float = 0.56,
    dry_run_only: bool = True,
    require_pre_release_support: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Place an object at a world pose, with a dry-run reachable proxy if needed."""
    # Kept as a config compatibility key, but intentionally ignored.  A
    # successful operation must leave the bottle wherever MuJoCo contact
    # dynamics placed it; no post-release pose correction is permitted.
    kwargs.pop("settle_condiment_upright", None)
    target_resolver = str(kwargs.pop("target_resolver", ""))
    if target_pos_world is None and target_pos is None:
        env = getattr(controller, "env", None)
        if target_resolver == "plate_stage_pose" and env is not None:
            target_pos_world = env.plate_stage_pose()
        elif target_resolver == "home_return_pose" and env is not None:
            target_pos_world = env.return_region_pose(
                kwargs.get("return_level", getattr(env, "demo_original_level", ""))
            )
    target_value = target_pos_world if target_pos_world is not None else target_pos
    if target_value is None:
        return {
            "primitive": "place_at_accessible",
            "object_name": object_name,
            "pass": False,
            "reason": "target_pos_required",
        }
    target_arr = np.asarray(target_value, dtype=float).reshape(-1)
    if target_arr.size < 3 or not np.all(np.isfinite(target_arr[:3])):
        return {
            "primitive": "place_at_accessible",
            "object_name": object_name,
            "pass": False,
            "reason": "invalid_target_pos",
        }

    execute = bool(controller.runtime_config.execute)
    adjusted = target_arr[:3].copy()
    adjustment = {
        "applied": False,
        "reason": "execute_true" if execute and dry_run_only else "within_limit",
    }
    if (not execute) or not bool(dry_run_only):
        adjusted, adjustment = accessible_object_pos_world(
            controller.obs,
            target_arr[:3],
            max_base_x=float(target_max_base_x),
        )

    if bool(require_pre_release_support):
        result = _place_bottle_at_supported_pose(
            controller,
            object_name=object_name,
            target_pos_world=adjusted,
            target_name=target_name or object_name,
            **kwargs,
        )
    else:
        result = controller.place_at_world_pose(
            object_name,
            target_pos_world=adjusted,
            target_name=target_name or object_name,
            **kwargs,
        )
    result["primitive"] = "place_at_accessible"
    result["intended_target_pos_world"] = target_arr[:3].astype(float).tolist()
    result["adjusted_target_pos_world"] = np.asarray(adjusted, dtype=float).tolist()
    result["target_adjustment"] = adjustment
    result["semantic_proxy_applied"] = bool((not execute) and adjustment.get("applied", False))
    return result


def _place_bottle_at_supported_pose(
    controller: Any,
    *,
    object_name: str,
    target_pos_world: Any,
    target_name: str | None = None,
    placement_reference: str = "bbox_bottom_center",
    placement_object_upright: bool = False,
    upright_quat_xyzw: Any | None = None,
    pre_release_bottom_min_offset: float = -0.04,
    pre_release_bottom_max_offset: float = 0.01,
    pre_release_xy_tolerance: float = 0.12,
    post_release_xy_tolerance: float | None = None,
    pre_release_upright_tolerance_deg: float = 15.0,
    place_descent_stop_bottom_offset: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Release only after the upright bottle bottom is at its support height."""

    # Historical WR06 configs may still carry these names.  They are consumed
    # as inert compatibility options; no release pinning or pose correction is
    # implemented on the physical path.
    kwargs.pop("release_separation_assist_distance", None)
    kwargs.pop("release_anchor_align_support", None)

    target_arr = np.asarray(target_pos_world, dtype=float).reshape(-1)
    if target_arr.size < 3 or not np.all(np.isfinite(target_arr[:3])):
        return {
            "primitive": "place_at_accessible",
            "object_name": str(object_name),
            "pass": False,
            "reason": "invalid_target_pos",
        }

    reference = normalize_placement_reference(placement_reference)
    if reference == "bbox_bottom_center":
        support_target = target_arr[:3].copy()
    elif reference == "object_origin":
        env = controller.env
        body_id = env.obj_body_id[str(object_name)]
        target_object_pose = np.eye(4, dtype=float)
        current_rotation = np.asarray(
            env.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        target_object_pose[:3, :3] = (
            upright_object_rotation(current_rotation)
            if bool(placement_object_upright)
            else current_rotation
        )
        target_object_pose[:3, 3] = target_arr[:3]
        support_target = object_reference_pos(
            env,
            str(object_name),
            placement_reference="bbox_bottom_center",
            object_pose_world=target_object_pose,
        )
    else:
        return {
            "primitive": "place_at_accessible",
            "object_name": str(object_name),
            "pass": False,
            "reason": f"unsupported_support_reference:{reference}",
        }

    min_bottom = float(pre_release_bottom_min_offset)
    max_bottom = float(pre_release_bottom_max_offset)
    xy_tolerance = float(pre_release_xy_tolerance)
    post_xy_tolerance = (
        xy_tolerance
        if post_release_xy_tolerance is None
        else float(post_release_xy_tolerance)
    )
    upright_threshold = float(
        np.cos(np.deg2rad(float(pre_release_upright_tolerance_deg)))
    )

    def physical_pose_sample() -> dict[str, Any]:
        env = controller.env
        bottom = object_reference_pos(
            env,
            str(object_name),
            placement_reference="bbox_bottom_center",
        )
        rotation = np.asarray(
            env.sim.data.body_xmat[env.obj_body_id[str(object_name)]],
            dtype=float,
        ).reshape(3, 3)
        return {
            "bottle_bottom_world": bottom.astype(float).tolist(),
            "bottom_height_offset_m": float(bottom[2] - support_target[2]),
            "xy_error_m": float(
                np.linalg.norm(bottom[:2] - support_target[:2])
            ),
            "upright_z_axis": float(rotation[2, 2]),
        }

    descent_stop_offset = (
        None
        if place_descent_stop_bottom_offset is None
        else float(place_descent_stop_bottom_offset)
    )

    def place_trajectory_stop_check() -> dict[str, Any]:
        sample = physical_pose_sample()
        checks = {
            "bottle_bottom_reached_support_guard": bool(
                sample["bottom_height_offset_m"] <= descent_stop_offset
            ),
            "bottle_xy_at_support_target": bool(
                sample["xy_error_m"] <= xy_tolerance
            ),
            "bottle_upright": bool(
                sample["upright_z_axis"] >= upright_threshold
            ),
        }
        return {
            "stop": bool(all(checks.values())),
            "reason": (
                "bottle_reached_plate_support_guard"
                if all(checks.values())
                else None
            ),
            "checks": checks,
            "stop_bottom_offset_m": descent_stop_offset,
            **sample,
        }

    def pre_release_check() -> dict[str, Any]:
        controller.refresh_obs()
        sample = physical_pose_sample()
        bottom_offset = float(sample["bottom_height_offset_m"])
        xy_error = float(sample["xy_error_m"])
        upright_z_axis = float(sample["upright_z_axis"])
        checks = {
            "bottle_bottom_at_support": bool(
                min_bottom <= bottom_offset <= max_bottom
            ),
            "bottle_xy_at_support_target": bool(xy_error <= xy_tolerance),
            "bottle_upright": bool(upright_z_axis >= upright_threshold),
        }
        return {
            "pass": bool(all(checks.values())),
            "reason": (
                None
                if all(checks.values())
                else "physical_bottle_support_not_verified"
            ),
            "checks": checks,
            "bottle_bottom_world": sample["bottle_bottom_world"],
            "support_target_world": support_target.astype(float).tolist(),
            "bottom_height_offset_m": bottom_offset,
            "bottom_height_bounds_m": [min_bottom, max_bottom],
            "xy_error_m": xy_error,
            "xy_tolerance_m": xy_tolerance,
            "upright_z_axis": upright_z_axis,
            "upright_threshold": upright_threshold,
        }

    result = controller.place_at_world_pose(
        str(object_name),
        target_pos_world=target_arr[:3],
        target_name=target_name or str(object_name),
        placement_reference=reference,
        placement_object_upright=bool(placement_object_upright),
        upright_quat_xyzw=upright_quat_xyzw,
        pre_release_check=pre_release_check,
        place_trajectory_stop_check=(
            place_trajectory_stop_check
            if descent_stop_offset is not None
            else None
        ),
        **kwargs,
    )
    if bool(controller.runtime_config.execute) and bool(result.get("pass", False)):
        controller.refresh_obs()
        post_release = physical_pose_sample()
        post_bottom_offset = float(post_release["bottom_height_offset_m"])
        post_xy_error = float(post_release["xy_error_m"])
        post_upright = float(post_release["upright_z_axis"])
        post_checks = {
            "bottle_bottom_at_support": bool(
                min_bottom <= post_bottom_offset <= max_bottom
            ),
            "bottle_xy_at_support_target": bool(
                post_xy_error <= post_xy_tolerance
            ),
            "bottle_upright": bool(post_upright >= upright_threshold),
        }
        stability = {
            "enabled": True,
            "pass": bool(all(post_checks.values())),
            "reason": (
                None
                if all(post_checks.values())
                else "released_bottle_not_stable_at_support"
            ),
            "checks": post_checks,
            "xy_tolerance_m": post_xy_tolerance,
            **post_release,
        }
        result["post_release_stability_check"] = stability
        result.setdefault("checks", {})["post_release_stability"] = bool(
            stability["pass"]
        )
        result["pass"] = bool(all(result["checks"].values()))
    else:
        result["post_release_stability_check"] = {
            "enabled": bool(controller.runtime_config.execute),
            "pass": not bool(controller.runtime_config.execute),
            "skipped": True,
            "reason": (
                "dry_run"
                if not bool(controller.runtime_config.execute)
                else "place_not_completed"
            ),
        }
    return result


def set_torso_height(
    controller: Any,
    *,
    holding_object: str,
    target_height: float,
    tolerance: float = 0.002,
    max_steps: int = 240,
    gripper_action: float = 1.0,
    holding_follow_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Move only the PandaOmron torso while preserving a verified held object."""
    target = float(target_height)
    height_tolerance = float(tolerance)
    step_limit = int(max_steps)
    follow_tolerance = float(holding_follow_tolerance)
    if not np.isfinite(target):
        raise ValueError("set_torso_height target_height must be finite")
    if not np.isfinite(height_tolerance) or height_tolerance <= 0.0:
        raise ValueError("set_torso_height tolerance must be positive and finite")
    if step_limit <= 0:
        raise ValueError("set_torso_height max_steps must be positive")
    if not np.isfinite(follow_tolerance) or follow_tolerance <= 0.0:
        raise ValueError(
            "set_torso_height holding_follow_tolerance must be positive and finite"
        )

    env = controller.env
    robot = env.robots[0]
    composite = robot.composite_controller
    torso = composite.part_controllers.get("torso")
    right = composite.part_controllers.get("right")
    if torso is None or right is None:
        return {
            "primitive": "set_torso_height",
            "holding_object": str(holding_object),
            "requested_height": target,
            "pass": False,
            "reason": "required_robot_controller_missing",
            "checks": {
                "torso_controller": torso is not None,
                "right_arm_controller": right is not None,
            },
        }

    held_before = getattr(controller, "_virtual_held_object", None)
    holding_adopted = held_before == str(holding_object)
    initial_height = float(qpos_for_controller(env, torso).reshape(-1)[0])
    if not holding_adopted:
        return {
            "primitive": "set_torso_height",
            "holding_object": str(holding_object),
            "requested_height": target,
            "initial_height": initial_height,
            "actual_height": initial_height,
            "pass": False,
            "reason": "holding_object_not_adopted",
            "checks": {
                "holding_object_adopted": False,
                "target_height_reached": False,
            },
        }

    joint_indexes = np.asarray(torso.joint_index, dtype=int).reshape(-1)
    if joint_indexes.size < 1:
        raise RuntimeError("torso controller does not expose a joint index")
    joint_range = np.asarray(
        env.sim.model.jnt_range[int(joint_indexes[0])],
        dtype=float,
    ).reshape(-1)
    if (
        joint_range.size < 2
        or not np.all(np.isfinite(joint_range[:2]))
        or target < float(joint_range[0]) - 1e-9
        or target > float(joint_range[1]) + 1e-9
    ):
        return {
            "primitive": "set_torso_height",
            "holding_object": str(holding_object),
            "requested_height": target,
            "initial_height": initial_height,
            "actual_height": initial_height,
            "joint_range": joint_range[:2].astype(float).tolist(),
            "pass": False,
            "reason": "target_height_outside_joint_range",
            "checks": {
                "holding_object_adopted": True,
                "target_height_valid": False,
                "target_height_reached": False,
            },
        }

    body_id = env.obj_body_id[str(holding_object)]
    initial_object_pos = np.asarray(
        env.sim.data.body_xpos[body_id],
        dtype=float,
    ).copy()
    execute = bool(controller.runtime_config.execute)
    if not execute:
        semantic_recorded = True
        if hasattr(env, "record_transport_posture_normalized"):
            semantic_recorded = bool(
                env.record_transport_posture_normalized(physically_normalized=True)
            )
        return {
            "primitive": "set_torso_height",
            "holding_object": str(holding_object),
            "requested_height": target,
            "initial_height": initial_height,
            "actual_height": initial_height,
            "joint_range": joint_range[:2].astype(float).tolist(),
            "dry_run": True,
            "steps": 0,
            "checks": {
                "holding_object_adopted": True,
                "target_height_valid": True,
                "transport_posture_normalized": semantic_recorded,
            },
            "pass": semantic_recorded,
        }

    from bench.execution.navigation.base_motion import step_base_and_hold_arm

    base_start, base_end = composite._action_split_indexes["base"]
    base_action = np.zeros(int(base_end) - int(base_start), dtype=float)
    arm_qpos = qpos_for_controller(env, right)
    actual_height = initial_height
    executed_steps = 0
    obs = controller.obs
    for _ in range(step_limit):
        error = target - actual_height
        if abs(error) <= height_tolerance:
            break
        torso_action = float(np.clip(error / 0.05, -1.0, 1.0))
        obs = step_base_and_hold_arm(
            env,
            arm_qpos=arm_qpos,
            base_action=base_action,
            base_mode=1.0,
            gripper_action=float(gripper_action),
            torso_action=torso_action,
        )
        executed_steps += 1
        actual_height = float(qpos_for_controller(env, torso).reshape(-1)[0])
    controller._sync_obs(obs)

    final_object_pos = np.asarray(
        env.sim.data.body_xpos[body_id],
        dtype=float,
    ).copy()
    expected_object_delta = np.array(
        [0.0, 0.0, actual_height - initial_height],
        dtype=float,
    )
    actual_object_delta = final_object_pos - initial_object_pos
    follow_error = float(np.linalg.norm(actual_object_delta - expected_object_delta))
    height_reached = abs(actual_height - target) <= height_tolerance
    holding_preserved = (
        getattr(controller, "_virtual_held_object", None) == str(holding_object)
    )
    object_followed = follow_error <= follow_tolerance
    checks = {
        "target_height_reached": bool(height_reached),
        "holding_object_preserved": bool(holding_preserved),
        "holding_object_followed_torso": bool(object_followed),
    }
    passed = bool(all(checks.values()))
    if passed and hasattr(env, "record_transport_posture_normalized"):
        checks["transport_posture_normalized"] = bool(
            env.record_transport_posture_normalized(physically_normalized=True)
        )
        passed = bool(all(checks.values()))
    return {
        "primitive": "set_torso_height",
        "holding_object": str(holding_object),
        "requested_height": target,
        "initial_height": initial_height,
        "actual_height": actual_height,
        "height_error": actual_height - target,
        "tolerance": height_tolerance,
        "joint_range": joint_range[:2].astype(float).tolist(),
        "gripper_action": float(gripper_action),
        "steps": int(executed_steps),
        "initial_object_pos": initial_object_pos.astype(float).tolist(),
        "final_object_pos": final_object_pos.astype(float).tolist(),
        "actual_object_delta": actual_object_delta.astype(float).tolist(),
        "expected_object_delta": expected_object_delta.astype(float).tolist(),
        "holding_follow_error": follow_error,
        "holding_follow_tolerance": follow_tolerance,
        "checks": checks,
        "pass": passed,
        "reason": None if passed else "torso_motion_validation_failed",
    }


__all__ = [
    "build_operation_handlers",
    "place_at_accessible",
    "set_torso_height",
]
