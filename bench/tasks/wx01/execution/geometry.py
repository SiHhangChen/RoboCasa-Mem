"""Resolve WX-01 domain targets into controller-facing world geometry."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.planning import matrix_from_base_obs, make_object_pick_target_poses
from bench.tasks.wx01.domain.contracts import EpisodePlan, EpisodeTargets


def resolve_episode_targets(env: Any, episode: EpisodePlan) -> EpisodeTargets:
    """Bind pure relative episode offsets to reset-time original world poses."""

    original_tool = env_pose(
        env,
        "original_tool_pose",
        fallback_object="cleaning_tool",
    )
    original_blocking = env_pose(
        env,
        "original_blocking_object_pose",
        fallback_object="blocking_object",
    )
    if original_tool is None or original_blocking is None:
        raise ValueError("WX-01 original poses are unavailable after initialization")
    temporary_blocking = np.asarray(original_blocking, dtype=float).copy()
    temporary_blocking[:2] += np.asarray(
        episode.temporary_blocking_offset_xy,
        dtype=float,
    )
    post_wipe_tool = np.asarray(
        [original_blocking[0], original_blocking[1], original_tool[2]],
        dtype=float,
    )
    post_wipe_tool[:2] += np.asarray(episode.temporary_tool_offset_xy, dtype=float)
    return EpisodeTargets(
        original_tool_pose=_tuple3(original_tool),
        original_blocking_pose=_tuple3(original_blocking),
        temporary_blocking_pose=_tuple3(temporary_blocking),
        post_wipe_tool_pose=_tuple3(post_wipe_tool),
    )


def make_blocking_pick_payload_selector(
    controller: Any,
    *,
    maximum_base_x: float,
    dry_run_only: bool,
) -> Any:
    """Keep the displaced cup planning proxy reachable in dry-run checks."""

    def selector(request: dict[str, Any]) -> dict[str, Any] | None:
        object_name = str(request.get("object_name", ""))
        if object_name != "blocking_object":
            return None
        if dry_run_only and bool(controller.runtime_config.execute):
            return None
        kwargs = dict(request.get("kwargs") or {})
        object_position = _object_position(
            request["env"],
            object_name,
            override=kwargs.get("object_pos_world"),
        )
        adjusted, adjustment = accessible_object_pos_world(
            request["obs"],
            object_position,
            max_base_x=float(kwargs.get("blocking_object_max_base_x", maximum_base_x)),
        )
        config = request["pick_place_config"]
        payload = make_object_pick_target_poses(
            request["env"],
            request["obs"],
            object_name=object_name,
            orientation_mode=str(kwargs.get("orientation_mode", "object-y-topdown")),
            pregrasp_height=float(
                kwargs.get("pregrasp_height", config.pregrasp_height)
            ),
            grasp_clearance=float(
                kwargs.get("grasp_clearance", config.grasp_clearance)
            ),
            lift_height=float(request.get("lift_height", config.lift_height)),
            grasp_yaw_offset_deg=float(
                kwargs.get("grasp_yaw_offset_deg", config.grasp_yaw_offset_deg)
            ),
            object_pos_world=adjusted,
        )
        return {
            "payload": payload,
            "selection": {
                "enabled": True,
                "mode": "wx01_accessible_blocking_object",
                "dry_run_only": bool(dry_run_only),
                "object_pos_world": object_position.tolist(),
                "adjusted_object_pos_world": adjusted.tolist(),
                "adjustment": adjustment,
                "pass": True,
            },
        }

    return selector


def accessible_object_pos_world(
    obs: dict[str, Any],
    object_pos_world: Any,
    *,
    max_base_x: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp a planning-only target to a maximum forward Base reach."""

    base_world = matrix_from_base_obs(obs)
    world_base = np.linalg.inv(base_world)
    object_h = np.ones(4, dtype=float)
    object_h[:3] = np.asarray(object_pos_world, dtype=float)[:3]
    object_base = world_base @ object_h
    adjusted_base = object_base.copy()
    applied = bool(adjusted_base[0] > float(max_base_x))
    if applied:
        adjusted_base[0] = float(max_base_x)
    adjusted_world = base_world @ adjusted_base
    return adjusted_world[:3], {
        "applied": applied,
        "max_base_x": float(max_base_x),
        "object_pos_base": object_base[:3].tolist(),
        "adjusted_object_pos_base": adjusted_base[:3].tolist(),
    }


def env_pose(
    env: Any, attribute: str, *, fallback_object: str | None = None
) -> np.ndarray | None:
    """Read a finite XYZ environment attribute with an object-body fallback."""

    value = getattr(env, attribute, None)
    pose = _optional_pose3(value)
    if pose is not None:
        return pose
    if fallback_object is None:
        return None
    try:
        return np.asarray(
            env.sim.data.body_xpos[env.obj_body_id[fallback_object]],
            dtype=float,
        )[:3]
    except Exception:
        return None


def _object_position(env: Any, object_name: str, *, override: Any = None) -> np.ndarray:
    if override is not None:
        pose = _optional_pose3(override)
        if pose is not None:
            return pose
    return np.asarray(
        env.sim.data.body_xpos[env.obj_body_id[object_name]], dtype=float
    )[:3]


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


def _tuple3(value: Any) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0]), float(array[1]), float(array[2])


__all__ = [
    "accessible_object_pos_world",
    "env_pose",
    "make_blocking_pick_payload_selector",
    "resolve_episode_targets",
]
