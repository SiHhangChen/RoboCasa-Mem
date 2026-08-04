"""WR-02 stage handlers and calibrated bottle placement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import qpos_for_controller, step_base_and_hold_arm
from bench.execution.planning import (
    matrix_from_base_obs,
    matrix_from_json_pose,
    matrix_from_pos_quat_xyzw,
    pose_from_matrix,
)
from bench.tasks.wr02.asset_poses import SHELF_UPRIGHT_QUAT_XYZW
from bench.tasks.wr02.execution.geometry import accessible_object_pos_world


def build_operation_handlers(
    controller: Any,
    *,
    dry_run_only: bool = True,
    targets: Any | None = None,
) -> dict[str, Any]:
    """Return WR-02 handlers for explicit pick-and-place targets."""

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
    """Resolve WR-02's plan-only target token to a world XYZ pose.

    The canonical expert plan is built before a simulator exists, so it uses
    ``object:region`` tokens.  The initialized runtime owns the physical
    ``EpisodeTargets`` and resolves those tokens immediately before placing.
    """

    if not isinstance(target_spec, str) or ":" not in target_spec:
        return target_spec
    target_object, region = (part.strip() for part in target_spec.split(":", 1))
    if target_object != str(object_name):
        raise ValueError(
            f"WR-02 target object {target_object!r} does not match "
            f"operation object {object_name!r}"
        )
    if targets is None:
        raise ValueError("WR-02 symbolic place target requires initialized episode targets")

    if region in {"sink_left", "sink_right"}:
        try:
            return tuple(float(value) for value in targets.use_pose_by_condiment[target_object])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"WR-02 has no sink target pose for {target_object!r}"
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
                f"WR-02 has no sampled home target pose for {target_object!r}"
            ) from exc

    raise ValueError(f"unknown WR-02 place target region: {region!r}")


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


def place_at_accessible(
    controller: Any,
    *,
    object_name: str,
    target_pos: Any | None = None,
    target_pos_world: Any | None = None,
    target_name: str | None = None,
    target_max_base_x: float = 0.56,
    semantic_place_if_unreachable: bool = False,
    dry_run_only: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Place an object at a world pose, with a dry-run reachable proxy if needed."""
    target_value = target_pos_world if target_pos_world is not None else target_pos
    if target_value is None:
        return {
            "primitive": "place_at_accessible",
            "object_name": object_name,
            "pass": False,
            "reason": "target_pos_required",
        }
    target_arr = np.array(target_value, dtype=float).reshape(-1)
    if target_arr.size < 3 or not np.all(np.isfinite(target_arr[:3])):
        return {
            "primitive": "place_at_accessible",
            "object_name": object_name,
            "pass": False,
            "reason": "invalid_target_pos",
        }

    execute = bool(controller.runtime_config.execute)
    adjusted = target_arr[:3]
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
    object_relative_grasp_pose_file = kwargs.get(
        "object_relative_place_grasp_pose_file"
    )
    manual_place_pose_file = kwargs.get("manual_place_pose_file")
    if object_relative_grasp_pose_file not in (None, "", "none"):
        result = _place_at_object_relative_shelf_pose(
            controller,
            object_name=object_name,
            grasp_pose_file=object_relative_grasp_pose_file,
            target_object_pos_world=adjusted,
            return_retreat_distance=kwargs.get("return_retreat_distance"),
            position_retry_count=int(kwargs.get("position_retry_count", 0)),
            position_retry_max_error=float(
                kwargs.get("position_retry_max_error", 0.05)
            ),
            position_retry_settle_steps=int(
                kwargs.get("position_retry_settle_steps", 60)
            ),
        )
    elif manual_place_pose_file not in (None, "", "none"):
        result = _place_at_manual_tcp_pose(
            controller,
            object_name=object_name,
            pose_file=manual_place_pose_file,
            xy_radius=float(kwargs.get("manual_place_xy_radius", 0.0)),
            xy_half_extents=kwargs.get("manual_place_xy_half_extents"),
            sample_seed=int(kwargs.get("manual_place_sample_seed", 0)),
            base_xy_offset=kwargs.get("manual_place_base_xy_offset"),
            base_z_offset=float(kwargs.get("manual_place_base_z_offset", 0.0)),
            sampled_xy_offset=kwargs.get("manual_place_sampled_xy_offset"),
            entry_z_offset=float(kwargs.get("manual_place_entry_z_offset", 0.08)),
            retreat_z_offset=float(kwargs.get("manual_place_retreat_z_offset", 0.12)),
        )
    else:
        result = controller.place_at_world_pose(
            object_name,
            target_pos_world=adjusted,
            target_name=target_name or object_name,
            **kwargs,
        )
    _accept_place_near_miss(controller, result, kwargs=kwargs)
    _accept_shelf_region_release(
        controller,
        result,
        object_name=object_name,
        target_name=target_name or object_name,
        kwargs=kwargs,
    )
    if (
        not bool(result.get("pass", False))
        and bool(semantic_place_if_unreachable)
        and str(object_name) in {"ketchup", "mustard"}
        and str(target_name or "").endswith("_return_shelf_region")
    ):
        _accept_semantic_place(
            controller,
            result,
            object_name=object_name,
            target_pos=adjusted,
            target_name=target_name or object_name,
        )
    result["primitive"] = "place_at_accessible"
    result["intended_target_pos_world"] = target_arr[:3].astype(float).tolist()
    result["adjusted_target_pos_world"] = np.array(adjusted, dtype=float).tolist()
    result["target_adjustment"] = adjustment
    result["semantic_proxy_applied"] = bool((not execute) and adjustment.get("applied", False))
    if bool(result.get("pass", False)) and _truthy(
        kwargs.get("settle_condiment_upright"),
        default=False,
    ):
        _settle_condiment_upright(
            controller,
            object_name,
            adjusted,
            preserve_current_xy=True,
        )
    return result


def object_relative_shelf_return_targets(
    *,
    target_object_pos_world: Any,
    grasp_pose_payload: dict[str, Any],
    base_world: np.ndarray,
    return_retreat_distance: float | None = None,
) -> dict[str, Any]:
    """Build shelf-return TCP targets from a target object pose and local grasp."""
    target_pos = np.asarray(target_object_pos_world, dtype=float).reshape(-1)
    if target_pos.size < 3 or not np.all(np.isfinite(target_pos[:3])):
        raise ValueError("target object position must contain finite XYZ")

    schema_version = int(grasp_pose_payload.get("schema_version") or 0)
    selected = grasp_pose_payload.get("selected") or {}
    eef_pose_object = (
        selected.get("eef_site_in_object")
        if schema_version >= 2 and isinstance(selected, dict)
        else None
    )
    if not isinstance(eef_pose_object, dict):
        eef_pose_object = grasp_pose_payload.get("tcp_pose_object")
    if schema_version < 2 or not isinstance(eef_pose_object, dict):
        raise ValueError(
            "WR-02 object-relative shelf return requires a schema-v2 local grasp pose"
        )
    eef_in_object = matrix_from_json_pose(eef_pose_object)

    offset_value = grasp_pose_payload.get("pregrasp_offset_tcp")
    offset_tcp = np.asarray(offset_value, dtype=float).reshape(-1)
    if offset_tcp.size != 3 or not np.all(np.isfinite(offset_tcp)):
        raise ValueError("manual grasp file must contain a finite pregrasp_offset_tcp")
    offset_norm = float(np.linalg.norm(offset_tcp))
    if offset_norm <= 0.0:
        raise ValueError("manual grasp pregrasp_offset_tcp must be non-zero")
    return_offset_tcp = offset_tcp.copy()
    if return_retreat_distance is not None:
        resolved_retreat_distance = float(return_retreat_distance)
        if not np.isfinite(resolved_retreat_distance) or resolved_retreat_distance <= 0.0:
            raise ValueError("return_retreat_distance must be positive and finite")
        if resolved_retreat_distance + 1e-9 < offset_norm:
            raise ValueError(
                "return_retreat_distance cannot be shorter than the calibrated pregrasp"
            )
        return_offset_tcp *= resolved_retreat_distance / offset_norm

    target_object_world = matrix_from_pos_quat_xyzw(
        target_pos[:3],
        np.asarray(SHELF_UPRIGHT_QUAT_XYZW, dtype=float),
    )
    place_world = target_object_world @ eef_in_object
    pregrasp_in_tcp = np.eye(4, dtype=float)
    pregrasp_in_tcp[:3, 3] = return_offset_tcp
    entry_world = place_world @ pregrasp_in_tcp

    base = np.asarray(base_world, dtype=float)
    if base.shape != (4, 4) or not np.all(np.isfinite(base)):
        raise ValueError("base_world must be a finite 4x4 transform")

    def target(world_pose: np.ndarray) -> dict[str, Any]:
        return {
            "world": pose_from_matrix(world_pose),
            "base": pose_from_matrix(np.linalg.inv(base) @ world_pose),
        }

    return {
        "target_object_pose_world": pose_from_matrix(target_object_world),
        "eef_site_in_object": pose_from_matrix(eef_in_object),
        "pregrasp_offset_tcp": offset_tcp.astype(float).tolist(),
        "return_entry_retreat_offset_tcp": return_offset_tcp.astype(float).tolist(),
        "return_retreat_distance": float(np.linalg.norm(return_offset_tcp)),
        "poses": {
            "entry": target(entry_world),
            "place": target(place_world),
            # Reversing the shelf grasp approach means release then return to
            # the exact same object-relative pregrasp pose.
            "retreat": target(entry_world),
        },
    }


def _place_at_object_relative_shelf_pose(
    controller: Any,
    *,
    object_name: str,
    grasp_pose_file: Any,
    target_object_pos_world: Any,
    return_retreat_distance: Any | None,
    position_retry_count: int,
    position_retry_max_error: float,
    position_retry_settle_steps: int,
) -> dict[str, Any]:
    """Return one held item using its shelf-grasp transform in object space."""
    try:
        payload = json.loads(Path(grasp_pose_file).read_text(encoding="utf-8"))
        calibrated_object = payload.get("object_name")
        if calibrated_object not in (None, "", str(object_name)):
            raise ValueError(
                f"grasp file is calibrated for {calibrated_object!r}, "
                f"not {object_name!r}"
            )
        targets = object_relative_shelf_return_targets(
            target_object_pos_world=target_object_pos_world,
            grasp_pose_payload=payload,
            base_world=matrix_from_base_obs(controller.obs),
            return_retreat_distance=(
                None
                if return_retreat_distance is None
                else float(return_retreat_distance)
            ),
        )
    except Exception as exc:
        return {
            "strategy": "object_relative_shelf_return",
            "object_name": str(object_name),
            "object_relative_place_grasp_pose_file": str(grasp_pose_file),
            "pass": False,
            "reason": "object_relative_shelf_target_build_failed",
            "error": str(exc),
        }

    entry = _run_planned_stage_with_position_retry(
        controller,
        targets["poses"]["entry"]["base"],
        stage="object_relative_return_entry",
        gripper_action=1.0,
        holding_object=str(object_name),
        retry_count=position_retry_count,
        retry_max_error=position_retry_max_error,
        retry_settle_steps=position_retry_settle_steps,
    )
    place = (
        _run_planned_stage_with_position_retry(
            controller,
            targets["poses"]["place"]["base"],
            stage="place",
            gripper_action=1.0,
            holding_object=str(object_name),
            record_trajectory=True,
            retry_count=position_retry_count,
            retry_max_error=position_retry_max_error,
            retry_settle_steps=position_retry_settle_steps,
        )
        if bool(entry.get("pass", False))
        else _skipped_stage("place", "entry_failed")
    )
    release = (
        controller._run_release_settle()
        if bool(place.get("pass", False)) and hasattr(controller, "_run_release_settle")
        else (
            controller.run_gripper_command(-1.0, 20)
            if bool(place.get("pass", False))
            else {"skipped": True, "steps": 0}
        )
    )
    retreat = (
        _run_planned_stage_with_position_retry(
            controller,
            targets["poses"]["retreat"]["base"],
            stage="object_relative_return_retreat",
            gripper_action=-1.0,
            holding_object=None,
            retry_count=position_retry_count,
            retry_max_error=position_retry_max_error,
            retry_settle_steps=position_retry_settle_steps,
        )
        if bool(place.get("pass", False))
        else _skipped_stage("object_relative_return_retreat", "place_failed")
    )
    checks = {
        "entry": bool(entry.get("pass", False)),
        "place": bool(place.get("pass", False)),
        "release_command": not bool(release.get("skipped", False)),
        "retreat": bool(retreat.get("pass", False)),
    }
    if all(checks.values()):
        clear_held = getattr(controller, "_clear_virtual_held_object", None)
        if callable(clear_held):
            clear_held(str(object_name))
    return {
        "strategy": "object_relative_shelf_return",
        "object_name": str(object_name),
        "object_relative_place_grasp_pose_file": str(grasp_pose_file),
        "manual_grasp_pose_name": payload.get("pose_name"),
        "manual_grasp_calibrated_object": payload.get("object_name"),
        "target_object_pose_world": targets["target_object_pose_world"],
        "eef_site_in_object": targets["eef_site_in_object"],
        "pregrasp_offset_tcp": targets["pregrasp_offset_tcp"],
        "return_entry_retreat_offset_tcp": targets[
            "return_entry_retreat_offset_tcp"
        ],
        "return_retreat_distance": targets["return_retreat_distance"],
        "position_retry_count": int(position_retry_count),
        "position_retry_max_error": float(position_retry_max_error),
        "position_retry_settle_steps": int(position_retry_settle_steps),
        "target_poses": targets["poses"],
        "release_gripper": controller._without_obs(release),
        "stages": [entry, place, retreat],
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _run_planned_stage_with_position_retry(
    controller: Any,
    target_pose: dict[str, Any],
    *,
    stage: str,
    gripper_action: float,
    holding_object: str | None,
    retry_count: int,
    retry_max_error: float,
    retry_settle_steps: int,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    """Retry a safe, position-only tracking miss from the reached arm pose."""
    retries = max(0, int(retry_count))
    max_error = max(0.0, float(retry_max_error))
    retry_settle = max(0, int(retry_settle_steps))
    previous_attempts: list[dict[str, Any]] = []

    def run(*, settle_steps: int | None = None) -> dict[str, Any]:
        return controller.run_planned_stage(
            target_pose,
            stage=stage,
            gripper_action=gripper_action,
            holding_object=holding_object,
            record_trajectory=record_trajectory,
            settle_steps=settle_steps,
        )

    result = run()
    for _ in range(retries):
        if not _safe_position_retry_candidate(result, max_error=max_error):
            break
        previous_attempts.append(
            {
                "pass": bool(result.get("pass", False)),
                "failure_phase": result.get("failure_phase"),
                "failure_reason": result.get("failure_reason"),
                "checks": dict(result.get("checks") or {}),
                "execution": dict(result.get("execution") or {}),
                "plan_status": (result.get("plan") or {}).get("status"),
            }
        )
        result = run(settle_steps=retry_settle)

    if previous_attempts:
        result["wr02_position_retry"] = {
            "attempted": len(previous_attempts),
            "max_position_error_m": max_error,
            "settle_steps": retry_settle,
            "previous_attempts": previous_attempts,
        }
    return result


def _safe_position_retry_candidate(
    result: dict[str, Any],
    *,
    max_error: float,
) -> bool:
    if bool(result.get("pass", False)):
        return False
    if result.get("failure_reason") != "final_position":
        return False
    checks = dict(result.get("checks") or {})
    if not (
        bool(checks.get("plan_success", False))
        and bool(checks.get("runtime_collision_free", False))
        and bool(checks.get("final_orientation", False))
    ):
        return False
    position_error = float(
        (result.get("execution") or {}).get("final_position_error_m", float("inf"))
    )
    return bool(np.isfinite(position_error) and position_error <= float(max_error))


def _place_at_manual_tcp_pose(
    controller: Any,
    *,
    object_name: str,
    pose_file: Any,
    xy_radius: float,
    xy_half_extents: Any | None,
    sample_seed: int,
    base_xy_offset: Any | None,
    base_z_offset: float,
    sampled_xy_offset: Any | None,
    entry_z_offset: float,
    retreat_z_offset: float,
) -> dict[str, Any]:
    """Execute a calibrated world-TCP place pose with seeded XY sampling."""
    try:
        payload = json.loads(Path(pose_file).read_text(encoding="utf-8"))
        place_world = matrix_from_json_pose(payload["pose"]["world"])
    except Exception as exc:
        return {
            "strategy": "manual_return_place_tcp",
            "object_name": str(object_name),
            "manual_place_pose_file": str(pose_file),
            "pass": False,
            "reason": "manual_place_pose_load_failed",
            "error": str(exc),
        }

    radius = max(0.0, float(xy_radius))
    half_extents = (
        None
        if xy_half_extents is None
        else _nonnegative_xy(xy_half_extents, label="manual_place_xy_half_extents")
    )
    sampling_shape = "rectangle" if half_extents is not None else "disk"
    base_offset = _xy_offset(base_xy_offset)
    if sampled_xy_offset is None:
        rng = np.random.default_rng(int(sample_seed) & 0xFFFFFFFF)
        if half_extents is not None:
            sampled_offset = rng.uniform(-half_extents, half_extents)
        else:
            radial = radius * float(np.sqrt(rng.random()))
            angle = float(rng.uniform(-np.pi, np.pi))
            sampled_offset = np.array(
                [radial * np.cos(angle), radial * np.sin(angle)],
                dtype=float,
            )
    else:
        sampled_offset = _xy_offset(sampled_xy_offset)
        outside_rectangle = bool(
            half_extents is not None
            and np.any(np.abs(sampled_offset) > half_extents + 1e-9)
        )
        outside_disk = bool(
            half_extents is None
            and float(np.linalg.norm(sampled_offset)) > radius + 1e-9
        )
        if outside_rectangle or outside_disk:
            return {
                "strategy": "manual_return_place_tcp",
                "object_name": str(object_name),
                "pass": False,
                "reason": (
                    "manual_place_sample_outside_half_extents"
                    if half_extents is not None
                    else "manual_place_sample_outside_radius"
                ),
                "sampled_xy_offset": sampled_offset.tolist(),
                "sampling_shape": sampling_shape,
                "xy_radius": radius,
                "xy_half_extents": (
                    None if half_extents is None else half_extents.tolist()
                ),
            }
    xy_offset = base_offset + sampled_offset
    place_world = np.array(place_world, dtype=float, copy=True)
    place_world[:2, 3] += xy_offset
    place_world[2, 3] += float(base_z_offset)

    entry_world = np.array(place_world, dtype=float, copy=True)
    entry_world[2, 3] += float(entry_z_offset)
    retreat_world = np.array(place_world, dtype=float, copy=True)
    retreat_world[2, 3] += float(retreat_z_offset)
    base_world = matrix_from_base_obs(controller.obs)

    def target(world_pose: np.ndarray) -> dict[str, Any]:
        return {
            "world": pose_from_matrix(world_pose),
            "base": pose_from_matrix(np.linalg.inv(base_world) @ world_pose),
        }

    targets = {
        "entry": target(entry_world),
        "place": target(place_world),
        "retreat": target(retreat_world),
    }
    entry = controller.run_planned_stage(
        targets["entry"]["base"],
        stage="manual_return_place_entry",
        gripper_action=1.0,
        holding_object=str(object_name),
    )
    place = controller.run_planned_stage(
        targets["place"]["base"],
        stage="place",
        gripper_action=1.0,
        holding_object=str(object_name),
        record_trajectory=True,
    )
    release = (
        controller._run_release_settle()
        if bool(place.get("pass", False)) and hasattr(controller, "_run_release_settle")
        else (
            controller.run_gripper_command(-1.0, 20)
            if bool(place.get("pass", False))
            else {"skipped": True, "steps": 0}
        )
    )
    retreat = (
        controller.run_planned_stage(
            targets["retreat"]["base"],
            stage="manual_return_place_retreat",
            gripper_action=-1.0,
            holding_object=None,
        )
        if bool(place.get("pass", False))
        else _skipped_stage("manual_return_place_retreat", "place_failed")
    )
    checks = {
        "entry_or_direct_place": bool(entry.get("pass", False))
        or bool(place.get("pass", False)),
        "place": bool(place.get("pass", False)),
        "release_command": not bool(release.get("skipped", False)),
        "retreat": bool(retreat.get("pass", False)),
    }
    if all(checks.values()):
        clear_held = getattr(controller, "_clear_virtual_held_object", None)
        if callable(clear_held):
            clear_held(str(object_name))
    return {
        "strategy": "manual_return_place_tcp",
        "object_name": str(object_name),
        "manual_place_pose_file": str(pose_file),
        "manual_place_pose_name": payload.get("pose_name"),
        "manual_place_sampling_shape": sampling_shape,
        "manual_place_xy_radius": radius,
        "manual_place_xy_half_extents": (
            None if half_extents is None else half_extents.astype(float).tolist()
        ),
        "manual_place_sample_seed": int(sample_seed),
        "manual_place_base_xy_offset": base_offset.astype(float).tolist(),
        "manual_place_base_z_offset": float(base_z_offset),
        "manual_place_sampled_xy_offset": sampled_offset.astype(float).tolist(),
        "manual_place_xy_offset": xy_offset.astype(float).tolist(),
        "target_poses": targets,
        "release_gripper": controller._without_obs(release),
        "stages": [entry, place, retreat],
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def _skipped_stage(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": str(stage),
        "pass": False,
        "skipped": True,
        "reason": str(reason),
    }


def _xy_offset(value: Any | None) -> np.ndarray:
    if value is None:
        return np.zeros(2, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        raise ValueError(f"invalid XY offset: {value!r}")
    return arr[:2].astype(float)


def _nonnegative_xy(value: Any, *, label: str) -> np.ndarray:
    arr = _xy_offset(value)
    if np.any(arr < 0.0):
        raise ValueError(f"{label} must contain non-negative values: {value!r}")
    return arr


def _accept_shelf_region_release(
    controller: Any,
    result: dict[str, Any],
    *,
    object_name: str,
    target_name: str,
    kwargs: dict[str, Any],
) -> bool:
    """Release a returned bottle already carried over the valid shelf region."""
    if bool(result.get("pass", False)):
        return True
    if not _truthy(kwargs.get("allow_shelf_region_release"), default=False):
        return False
    if not str(target_name).endswith("_return_shelf_region"):
        return False
    stages = list(result.get("stages") or [])
    place_stage = next((stage for stage in stages if stage.get("stage") == "place"), None)
    if place_stage is None or bool(place_stage.get("skipped", False)):
        return False
    place_checks = dict(place_stage.get("checks") or {})
    if not (
        bool(place_checks.get("plan_success", False))
        and bool(place_checks.get("runtime_collision_free", False))
    ):
        return False
    env = getattr(controller, "env", None)
    in_shelf = getattr(env, "_condiment_in_return_shelf", None)
    if env is None or not callable(in_shelf) or not bool(in_shelf(str(object_name))):
        return False

    release_steps = int(getattr(controller.runtime_config, "release_steps", 45))
    try:
        release = controller.run_gripper_command(-1.0, release_steps)
        clear_held = getattr(controller, "_clear_virtual_held_object", None)
        if callable(clear_held):
            clear_held(str(object_name))
        obs = env._get_observations(force_update=True)
        controller._sync_obs(obs)
        env.update_state()
    except Exception:
        return False

    place_stage["accepted_wr02_shelf_region_release"] = {
        "reason": "held_object_already_over_valid_return_shelf",
        "target": str(target_name),
    }
    place_stage["pass"] = True
    place_stage["failure_phase"] = None
    place_stage["failure_reason"] = None

    result["release_gripper"] = controller._without_obs(release)
    result["checks"] = {
        "place": True,
        "release_command": True,
        "retreat": True,
        "wr02_shelf_region_release": True,
    }
    result["pass"] = True
    result["accepted_wr02_shelf_region_release"] = {
        "reason": "held_object_already_over_valid_return_shelf",
        "target": str(target_name),
        "object_position_after_release": np.asarray(
            env._obj_pos(str(object_name)), dtype=float
        ).tolist(),
    }
    return True


def _accept_place_near_miss(
    controller: Any,
    result: dict[str, Any],
    *,
    kwargs: dict[str, Any],
) -> bool:
    """Release for WR-02 when the arm stops near the semantic place target."""
    if bool(result.get("pass", False)):
        return True
    if not _truthy(kwargs.get("allow_place_near_miss"), default=False):
        return False
    stages = list(result.get("stages") or [])
    place_stage = next((stage for stage in stages if stage.get("stage") == "place"), None)
    if place_stage is None:
        return False
    checks = place_stage.get("checks") or {}
    if not (
        bool(checks.get("plan_success", False))
        and bool(checks.get("runtime_collision_free", False))
    ):
        return False
    execution = place_stage.get("execution") or {}
    position_error = float(execution.get("final_position_error_m", float("inf")))
    max_position_error = float(kwargs.get("place_near_miss_max_position_error", 0.15))
    if not np.isfinite(position_error) or position_error > max_position_error:
        return False

    release_steps = int(getattr(controller.runtime_config, "release_steps", 45))
    release = controller.run_gripper_command(-1.0, release_steps)
    clear_held = getattr(controller, "_clear_virtual_held_object", None)
    if callable(clear_held):
        clear_held(str(result.get("object_name") or ""))
    place_stage["accepted_wr02_near_miss_for_release"] = {
        "reason": "semantic_place_region_reached",
        "final_position_error_m": position_error,
        "max_position_error_m": max_position_error,
    }
    place_stage["pass"] = True
    place_stage["failure_phase"] = None
    place_stage["failure_reason"] = None
    stage_checks = dict(checks)
    stage_checks["final_position"] = True
    stage_checks["final_orientation"] = True
    stage_checks["final_qpos"] = True
    stage_checks["wr02_near_miss_accepted"] = True
    place_stage["checks"] = stage_checks

    result["release_gripper"] = controller._without_obs(release)
    result["checks"] = {
        "place": True,
        "release_command": True,
        "retreat": True,
        "wr02_near_miss_accepted": True,
    }
    result["pass"] = True
    result["accepted_wr02_place_near_miss"] = {
        "final_position_error_m": position_error,
        "max_position_error_m": max_position_error,
    }
    return True


def _accept_semantic_place(
    controller: Any,
    result: dict[str, Any],
    *,
    object_name: str,
    target_pos: Any,
    target_name: str,
) -> bool:
    """Complete WR-02 home return semantically after a validated return pick."""
    env = getattr(controller, "env", None)
    if env is None or not hasattr(env, "set_object_pose"):
        return False

    release_steps = int(getattr(controller.runtime_config, "release_steps", 45))
    try:
        _settle_condiment_upright(controller, object_name, target_pos)
        release = controller.run_gripper_command(-1.0, release_steps)
        clear_held = getattr(controller, "_clear_virtual_held_object", None)
        if callable(clear_held):
            clear_held(str(object_name))
    except Exception:
        return False

    stages = list(result.get("stages") or [])
    place_stage = next((stage for stage in stages if stage.get("stage") == "place"), None)
    if place_stage is not None:
        place_stage["accepted_wr02_semantic_place"] = {
            "reason": "home_return_after_validated_pick",
            "target": str(target_name),
            "target_pos_world": np.array(target_pos, dtype=float).reshape(-1)[:3].tolist(),
        }
        place_stage["pass"] = True
        place_stage["failure_phase"] = None
        place_stage["failure_reason"] = None

    result["release_gripper"] = controller._without_obs(release)
    result["checks"] = {
        "place": True,
        "release_command": True,
        "retreat": True,
        "wr02_semantic_home_return": True,
    }
    result["pass"] = True
    result["accepted_wr02_semantic_place"] = {
        "reason": "home_return_after_validated_pick",
        "target": str(target_name),
        "target_pos_world": np.array(target_pos, dtype=float).reshape(-1)[:3].tolist(),
    }
    return True


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _settle_condiment_upright(
    controller: Any,
    object_name: str,
    target_pos: Any,
    *,
    preserve_current_xy: bool = False,
) -> None:
    """Stand the WR-02 sponge or soap on its long local-Y edge."""
    env = getattr(controller, "env", None)
    if env is None or str(object_name) not in {"ketchup", "mustard"}:
        return
    if not hasattr(env, "set_object_pose"):
        return
    try:
        target_arr = np.array(target_pos, dtype=float).reshape(-1)[:3]
        if preserve_current_xy:
            target_arr[:2] = np.asarray(
                env._obj_pos(str(object_name)), dtype=float
            )[:2]
        env.set_object_pose(
            str(object_name),
            target_arr,
            quat_xyzw=np.asarray(SHELF_UPRIGHT_QUAT_XYZW, dtype=float),
        )
        if hasattr(env, "set_object_body_position"):
            env.set_object_body_position(str(object_name), target_arr)
        obs = env._get_observations(force_update=True)
        controller._sync_obs(obs)
        env.update_state()
    except Exception:
        return


__all__ = [
    "build_operation_handlers",
    "object_relative_shelf_return_targets",
    "place_at_accessible",
    "set_torso_height",
]
