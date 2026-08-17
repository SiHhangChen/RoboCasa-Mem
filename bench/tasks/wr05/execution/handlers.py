"""WR-05 operation handlers with physically guarded release gating.

Plate staging and shelf return both run through the guarded supported-release
place: the descent can early-stop once the bottle bottom enters its support
guard, and opening the gripper is gated on a physical pre-release check.
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
from bench.tasks.wr05.execution.geometry import accessible_object_pos_world


def build_operation_handlers(
    controller: Any,
    *,
    dry_run_only: bool = True,
    targets: Any | None = None,
) -> dict[str, Any]:
    """Return WR-05 handlers for explicit pick-and-place targets."""

    def place_at_accessible_handler(target: str, **kwargs: Any) -> dict[str, Any]:
        target_spec = kwargs.pop("target_pos", None)
        if target_spec is not None:
            kwargs["target_pos_world"] = _resolve_symbolic_target(
                object_name=str(target),
                target_spec=target_spec,
                targets=targets,
            )
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

    return {
        "place_at_accessible": place_at_accessible_handler,
        "set_torso_height": set_torso_height_handler,
    }


def _resolve_symbolic_target(
    *,
    object_name: str,
    target_spec: Any,
    targets: Any | None,
) -> Any:
    """Resolve WR-05's plan-only target token to a world XYZ pose.

    The canonical expert plan is built before a simulator exists, so it uses
    ``object:region`` tokens.  The initialized runtime owns the physical
    ``EpisodeTargets`` and resolves those tokens immediately before placing.
    """

    if not isinstance(target_spec, str) or ":" not in target_spec:
        return target_spec
    target_object, region = (part.strip() for part in target_spec.split(":", 1))
    if target_object != str(object_name):
        raise ValueError(
            f"WR-05 target object {target_object!r} does not match "
            f"operation object {object_name!r}"
        )
    if targets is None:
        raise ValueError("WR-05 symbolic place target requires initialized episode targets")

    if region in {"plate_left", "plate_right"}:
        try:
            return tuple(float(value) for value in targets.use_pose_by_condiment[target_object])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"WR-05 has no plate target pose for {target_object!r}"
            ) from exc

    if region in {"shelf_lower", "shelf_upper"}:
        try:
            sampling = targets.return_place_sampling[target_object]
            sampled_xy = sampling["sampled_target_xy"]
            original_home = targets.original_home_pose_by_condiment[target_object]
            return (
                float(sampled_xy[0]),
                float(sampled_xy[1]),
                float(original_home[2]),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"WR-05 has no sampled home target pose for {target_object!r}"
            ) from exc

    raise ValueError(f"unknown WR-05 place target region: {region!r}")


def place_at_accessible(
    controller: Any,
    *,
    object_name: str,
    target_pos: Any | None = None,
    target_pos_world: Any | None = None,
    target_name: str | None = None,
    target_max_base_x: float = 0.56,
    dry_run_only: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Place an object at a world pose, with a dry-run reachable proxy if needed."""
    # Kept as a config compatibility key, but intentionally ignored.  A
    # successful operation must leave the bottle wherever MuJoCo contact
    # dynamics placed it; no post-release pose correction is permitted.
    kwargs.pop("settle_condiment_upright", None)
    position_retry_max_error = kwargs.get("position_retry_max_error")
    if position_retry_max_error is not None:
        # Source WR-05 inherited one Cartesian retry within this error bound.
        # Canonical execution exposes the same safe behavior as an entry-only
        # near-miss gate; the final place and physical release checks remain
        # strict.
        kwargs.setdefault("allow_place_entry_near_miss", True)
        kwargs.setdefault(
            "place_entry_near_miss_max_position_error",
            float(position_retry_max_error),
        )
    # These settings are part of source WR-05's effective inherited contract,
    # but the canonical shared controller has no matching fallback paths. Keep
    # them in task configuration for provenance and consume them here so they
    # cannot accidentally become shared-controller arguments.
    for legacy_key in (
        "position_retry_count",
        "position_retry_max_error",
        "position_retry_settle_steps",
        "allow_shelf_region_release",
        "semantic_place_if_unreachable",
        "object_relative_place_grasp_pose_file",
        "release_separation_assist_distance",
        "release_anchor_align_support",
    ):
        kwargs.pop(legacy_key, None)
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

    if not bool(kwargs.pop("require_pre_release_support", False)):
        result = controller.place_at_world_pose(
            object_name,
            target_pos_world=adjusted,
            target_name=target_name or object_name,
            **kwargs,
        )
    else:
        result = _place_bottle_at_supported_pose(
            controller,
            object_name=object_name,
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
    pre_release_upright_tolerance_deg: float = 15.0,
    place_descent_stop_bottom_offset: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Release only after the upright bottle bottom is at its support height."""

    # Historical WR05 configs may still carry these names.  They are consumed
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

    # Plate targets directly name the bottle-bottom reference, whereas shelf
    # return targets retain the bottle's remembered object-origin position.
    # Convert either contract to the expected support point before checking
    # the live bottle bottom.  This keeps the shelf check independent of the
    # condiment asset's origin-to-bottom offset.
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
    return controller.place_at_world_pose(
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
            },
            "pass": True,
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
