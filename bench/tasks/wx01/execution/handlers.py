"""Thin adapters connecting WX-01 stages, controllers and semantic events."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from bench.execution.planning import matrix_from_base_obs, matrix_with_pose, pose_from_matrix
from bench.tasks.wx01.execution.geometry import (
    accessible_object_pos_world,
    env_pose,
)


def build_stage_handlers(
    controller: Any,
    *,
    settings: Mapping[str, Any],
    targets: Any,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Create only the two custom operations used by the six-stage workflow."""

    dry_wipe_proxy = bool(settings.get("dry_run_wipe_semantic_proxy", True))
    accessible_place_dry_run_only = bool(
        settings.get("accessible_place_dry_run_only", True)
    )

    def wipe(target: str, **kwargs: Any) -> dict[str, Any]:
        kwargs.pop("wipe_sampling", None)
        return wipe_remembered_area(
            controller,
            tool_name=str(target),
            target_pose_attr=str(
                kwargs.pop("target_pose_attr", "original_blocking_object_pose")
            ),
            offsets_xy=kwargs.pop("wipe_offsets_xy"),
            plan_execute_wipe_sweep=bool(kwargs.pop("plan_execute_wipe_sweep", True)),
            contact_z_offset=float(kwargs.pop("contact_z_offset", 0.005)),
            settle_steps=int(kwargs.pop("settle_steps", 15)),
            allow_final_qpos_near_miss=bool(
                kwargs.pop("allow_final_qpos_near_miss", True)
            ),
            maximum_final_qpos_error=float(kwargs.pop("max_final_qpos_error", 0.30)),
            dry_run_semantic_proxy=dry_wipe_proxy,
        )

    def place_at_accessible(target: str, **kwargs: Any) -> dict[str, Any]:
        resolver = kwargs.pop("target_resolver", None)
        if kwargs.get("target_pos") is None and kwargs.get("target_pos_world") is None:
            if resolver == "post_wipe_tool_pose":
                kwargs["target_pos_world"] = targets.post_wipe_tool_pose
            elif resolver == "original_blocking_pose":
                kwargs["target_pos_world"] = targets.original_blocking_pose
        return place_at_accessible_target(
            controller,
            object_name=str(target),
            dry_run_only=accessible_place_dry_run_only,
            **kwargs,
        )

    return {"wipe": wipe, "place_at_accessible": place_at_accessible}


def place_at_accessible_target(
    controller: Any,
    *,
    object_name: str,
    target_pos: Any | None = None,
    target_pos_world: Any | None = None,
    target_name: str | None = None,
    target_max_base_x: float = 0.52,
    dry_run_only: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Place at a world target, using a reach proxy only where configured."""

    raw_target = target_pos_world if target_pos_world is not None else target_pos
    if raw_target is None:
        return {
            "primitive": "wx01_place_at_accessible_target",
            "object_name": object_name,
            "pass": False,
            "reason": "target_pos_required",
        }
    intended = np.asarray(raw_target, dtype=float).reshape(-1)
    if intended.size < 3 or not np.all(np.isfinite(intended[:3])):
        return {
            "primitive": "wx01_place_at_accessible_target",
            "object_name": object_name,
            "pass": False,
            "reason": "invalid_target_pos",
        }
    execute = bool(controller.runtime_config.execute)
    adjusted = intended[:3]
    adjustment = {
        "applied": False,
        "reason": "execute_true" if execute and dry_run_only else "within_limit",
    }
    if not execute or not dry_run_only:
        adjusted, adjustment = accessible_object_pos_world(
            controller.obs,
            intended[:3],
            max_base_x=float(target_max_base_x),
        )
    result = controller.place_at_world_pose(
        object_name,
        target_pos_world=adjusted,
        target_name=target_name or object_name,
        pre_release_check=_make_blocking_origin_pre_release_check(
            controller,
            object_name=object_name,
            intended_target=intended[:3],
            enabled=bool(kwargs.pop("require_pre_release_check", False)),
            xy_tolerance=float(kwargs.pop("pre_release_xy_tolerance", 0.015)),
            z_tolerance=float(kwargs.pop("pre_release_z_tolerance", 0.03)),
            upright_threshold_deg=float(
                kwargs.pop("pre_release_upright_threshold_deg", 15.0)
            ),
        ),
        **kwargs,
    )
    result.update(
        {
            "primitive": "wx01_place_at_accessible_target",
            "intended_target_pos_world": intended[:3].tolist(),
            "adjusted_target_pos_world": np.asarray(adjusted, dtype=float).tolist(),
            "target_adjustment": adjustment,
            "semantic_proxy_applied": bool(
                not execute and adjustment.get("applied", False)
            ),
        }
    )
    return result


def _make_blocking_origin_pre_release_check(
    controller: Any,
    *,
    object_name: str,
    intended_target: np.ndarray,
    enabled: bool,
    xy_tolerance: float,
    z_tolerance: float,
    upright_threshold_deg: float,
) -> Callable[[], dict[str, Any]] | None:
    """Prevent release until the single-wall-held cup is aligned and upright."""

    if not enabled or object_name != "blocking_object":
        return None
    if xy_tolerance <= 0.0 or z_tolerance <= 0.0:
        raise ValueError("WX-01 pre-release tolerances must be positive")
    if not 0.0 < upright_threshold_deg < 90.0:
        raise ValueError("WX-01 upright threshold must be in (0, 90) degrees")

    def check() -> dict[str, Any]:
        if not bool(controller.runtime_config.execute):
            return {
                "pass": True,
                "deferred_to_execute": True,
                "reason": "dry_run_physical_check_deferred",
            }
        env = controller.env
        controller.refresh_obs()
        position = np.asarray(
            env.sim.data.body_xpos[env.obj_body_id[object_name]], dtype=float
        ).reshape(3)
        rotation = np.asarray(
            env.sim.data.body_xmat[env.obj_body_id[object_name]], dtype=float
        ).reshape(3, 3)
        xy_error = float(np.linalg.norm(position[:2] - intended_target[:2]))
        z_error = float(abs(position[2] - intended_target[2]))
        upright_axis = float(rotation[2, 2])
        upright_minimum = float(np.cos(np.deg2rad(upright_threshold_deg)))
        checks = {
            "blocking_cup_xy_at_origin": bool(xy_error <= xy_tolerance),
            "blocking_cup_near_origin_height": bool(z_error <= z_tolerance),
            "blocking_cup_upright": bool(upright_axis >= upright_minimum),
        }
        return {
            "pass": bool(all(checks.values())),
            "reason": (
                None
                if all(checks.values())
                else "blocking_cup_origin_pre_release_check_failed"
            ),
            "checks": checks,
            "object_position_world": position.tolist(),
            "intended_target_world": intended_target.tolist(),
            "xy_error_m": xy_error,
            "xy_tolerance_m": xy_tolerance,
            "z_error_m": z_error,
            "z_tolerance_m": z_tolerance,
            "upright_axis": upright_axis,
            "upright_minimum": upright_minimum,
            "upright_threshold_deg": upright_threshold_deg,
        }

    return check


def wipe_remembered_area(
    controller: Any,
    *,
    tool_name: str,
    target_pose_attr: str,
    offsets_xy: Any,
    plan_execute_wipe_sweep: bool,
    contact_z_offset: float,
    settle_steps: int,
    allow_final_qpos_near_miss: bool,
    maximum_final_qpos_error: float,
    dry_run_semantic_proxy: bool,
) -> dict[str, Any]:
    """Plan the wipe sweep and latch its historical semantic event."""

    env = controller.env
    target = env_pose(env, target_pose_attr, fallback_object="blocking_object")
    if target is None:
        return {
            "primitive": "wx01_wipe_remembered_area",
            "tool_name": tool_name,
            "pass": False,
            "reason": "target_pose_unavailable",
        }
    offsets = _offset_array(offsets_xy)
    execute = bool(controller.runtime_config.execute)
    surface_z, surface_metadata = dining_table_surface_z(
        env,
        fallback=float(target[2]),
    )
    planned_stages: list[dict[str, Any]] = []
    if execute or plan_execute_wipe_sweep:
        planned_stages = _plan_wipe_sweep(
            controller,
            target,
            offsets,
            tool_name=tool_name,
            surface_z=surface_z,
            contact_z_offset=float(contact_z_offset),
            settle_steps=int(settle_steps),
            allow_final_qpos_near_miss=bool(allow_final_qpos_near_miss),
            maximum_final_qpos_error=float(maximum_final_qpos_error),
            execute=execute,
        )
    planning_passed = bool(planned_stages) and all(
        bool(stage.get("wipe_stage_accepted", stage.get("pass", False)))
        for stage in planned_stages
    )
    semantic: dict[str, Any] = {}
    if execute:
        contacts = [
            np.asarray(contact, dtype=float).reshape(-1)[:2]
            for contact in getattr(env, "table_contact_positions", [])
        ]
        physical_accepted = bool(getattr(env, "original_area_wiped", False))
        semantic = {
            "table_contact_timer": int(getattr(env, "table_contact_timer", 0)),
            "wipe_sweep_range": float(getattr(env, "wipe_sweep_range", 0.0)),
            "remembered_area_wiped": physical_accepted,
            "contact_positions": [contact.tolist() for contact in contacts],
            "accepted": physical_accepted,
        }
        semantic_accepted = bool(planning_passed and physical_accepted)
    elif dry_run_semantic_proxy and planning_passed:
        semantic = commit_wipe_event(env, target, offsets)
        semantic_accepted = bool(semantic.get("accepted", False))
    else:
        semantic_accepted = False
    checks = {
        "target_pose_available": True,
        "planned_sweep": (not planned_stages) or planning_passed,
        "wipe_event_recorded": semantic_accepted,
    }
    return {
        "primitive": "wx01_wipe_remembered_area",
        "tool_name": tool_name,
        "target_pose_attr": target_pose_attr,
        "target_pose_world": target.tolist(),
        "surface_z": surface_z,
        "surface_metadata": surface_metadata,
        "contact_z_offset": float(contact_z_offset),
        "settle_steps": int(settle_steps),
        "allow_final_qpos_near_miss": bool(allow_final_qpos_near_miss),
        "maximum_final_qpos_error": float(maximum_final_qpos_error),
        "stages": planned_stages,
        "checks": checks,
        "dry_run": not execute,
        "semantic_proxy_applied": bool(not execute and semantic_accepted),
        "semantic": semantic,
        "pass": all(checks.values()),
    }


def commit_wipe_event(
    env: Any,
    target: np.ndarray,
    offsets_xy: np.ndarray,
) -> dict[str, Any]:
    """Install accepted contact evidence and trigger the domain semantic event."""

    center = np.asarray(target, dtype=float).reshape(-1)[:2]
    contacts = [center + np.asarray(offset, dtype=float)[:2] for offset in offsets_xy]
    env.table_contact_positions = [
        np.asarray(contact, dtype=float) for contact in contacts
    ]
    env.table_contact_timer = len(contacts)
    if contacts:
        positions = np.asarray(contacts, dtype=float)
        env.wipe_sweep_range = float(
            np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
        )
    else:
        env.wipe_sweep_range = 0.0
    covered = any(
        np.linalg.norm(contact - center) <= float(env.wipe_radius)
        for contact in contacts
    )
    accepted = bool(
        covered
        and env.table_contact_timer >= int(env.minimum_contact_count)
        and env.wipe_sweep_range >= float(env.minimum_wipe_sweep)
    )
    if accepted:
        env.record_wipe_completed()
    return {
        "table_contact_timer": int(env.table_contact_timer),
        "wipe_sweep_range": float(env.wipe_sweep_range),
        "remembered_area_wiped": bool(env.original_area_wiped),
        "contact_positions": [contact.tolist() for contact in contacts],
        "accepted": accepted,
    }


def dining_table_surface_z(
    env: Any, *, fallback: float
) -> tuple[float, dict[str, Any]]:
    """Read the dining-table top height without importing RoboCasa in domain code."""

    fixture = getattr(env, "dining_table", None)
    if fixture is not None:
        try:
            points = np.asarray(fixture.get_bbox_points(), dtype=float)
            if (
                points.ndim == 2
                and points.shape[1] >= 3
                and np.all(np.isfinite(points[:, :3]))
            ):
                return float(np.max(points[:, 2])), {
                    "source": "dining_table.get_bbox_points.max_z",
                    "fixture": str(getattr(fixture, "name", "dining_table")),
                    "bbox_min": np.min(points[:, :3], axis=0).tolist(),
                    "bbox_max": np.max(points[:, :3], axis=0).tolist(),
                }
        except Exception as exc:
            bbox_error = f"{type(exc).__name__}: {exc}"
        else:
            bbox_error = "invalid_bbox_points"
        position = _optional_pose3(getattr(fixture, "pos", None))
        size = _optional_pose3(getattr(fixture, "size", None))
        if position is not None and size is not None:
            return float(position[2] + 0.5 * size[2]), {
                "source": "dining_table.pos_size.max_z",
                "fixture": str(getattr(fixture, "name", "dining_table")),
                "fixture_pos": position.tolist(),
                "fixture_size": size.tolist(),
                "bbox_error": bbox_error,
            }
    return float(fallback), {"source": "fallback_target_z", "fallback": float(fallback)}


def _plan_wipe_sweep(
    controller: Any,
    target: np.ndarray,
    offsets_xy: np.ndarray,
    *,
    tool_name: str,
    surface_z: float,
    contact_z_offset: float,
    settle_steps: int,
    allow_final_qpos_near_miss: bool,
    maximum_final_qpos_error: float,
    execute: bool,
) -> list[dict[str, Any]]:
    current_eef = controller.stage_executor.current_eef_site_world()
    rotation = current_eef[:3, :3]
    base_world = matrix_from_base_obs(controller.obs)
    target_poses = [
        pose_from_matrix(
            np.linalg.inv(base_world)
            @ matrix_with_pose(
                np.asarray(
                    [
                        target[0] + offset[0],
                        target[1] + offset[1],
                        surface_z + contact_z_offset,
                    ],
                    dtype=float,
                ),
                rotation,
            )
        )
        for offset in offsets_xy
    ]
    final_offset = offsets_xy[-1]
    lift_pose = pose_from_matrix(
        np.linalg.inv(base_world)
        @ matrix_with_pose(
            np.asarray(
                [
                    target[0] + final_offset[0],
                    target[1] + final_offset[1],
                    surface_z + contact_z_offset + 0.12,
                ],
                dtype=float,
            ),
            rotation,
        )
    )
    stages: list[dict[str, Any]] = []
    waypoints = [
        (f"wipe_{index:02d}", target_pose)
        for index, target_pose in enumerate(target_poses)
    ]
    waypoints.append(("wipe_lift", lift_pose))
    for stage_name, target_pose in waypoints:
        stage = controller.run_planned_stage(
            target_pose,
            stage=stage_name,
            gripper_action=1.0,
            holding_object=tool_name,
            execute=execute,
            settle_steps=max(0, int(settle_steps)),
        )
        accepted = bool(stage.get("pass", False))
        if not accepted and _accept_final_qpos_near_miss(
            stage,
            enabled=allow_final_qpos_near_miss,
            maximum_error=maximum_final_qpos_error,
        ):
            accepted = True
            stage["wipe_final_qpos_near_miss"] = {
                "accepted": True,
                "maximum_error": float(maximum_final_qpos_error),
                "actual_error": float(stage["execution"]["final_qpos_error"]),
            }
        stage["wipe_stage_accepted"] = accepted
        stages.append(stage)
        if not accepted:
            break
    controller.stage_executor.clear_virtual_state()
    return stages


def _accept_final_qpos_near_miss(
    stage: Mapping[str, Any],
    *,
    enabled: bool,
    maximum_error: float,
) -> bool:
    checks = stage.get("checks") or {}
    failed_checks = {name for name, passed in checks.items() if not bool(passed)}
    error = float((stage.get("execution") or {}).get("final_qpos_error", np.inf))
    return bool(
        enabled
        and failed_checks == {"final_qpos"}
        and np.isfinite(error)
        and error <= float(maximum_error)
    )


def _offset_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError("WX-01 wipe offsets must be a finite sequence of XY pairs")
    return array


def _optional_pose3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3]


__all__ = [
    "build_stage_handlers",
    "commit_wipe_event",
    "dining_table_surface_z",
    "place_at_accessible_target",
    "wipe_remembered_area",
]
