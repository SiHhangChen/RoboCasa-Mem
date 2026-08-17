"""Bind WR-06 episode facts to reset-time world geometry."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.planning import matrix_from_base_obs
from bench.tasks.wr06.domain.contracts import EpisodePlan, EpisodeTargets, ITEM_ID


def resolve_episode_targets(env: Any, episode: EpisodePlan) -> EpisodeTargets:
    level = str(episode.demo_original_level)
    plate_pose = _tuple3(env.plate_stage_pose())
    home_pose = _tuple3(env.return_region_pose(level))
    return EpisodeTargets(
        plate_stage_pose=plate_pose,
        home_return_pose=home_pose,
    )


def accessible_object_pos_world(
    obs: dict[str, Any], object_pos_world: Any, *, max_base_x: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp a planning target to a maximum forward Base reach."""

    base_world = matrix_from_base_obs(obs)
    object_h = np.ones(4, dtype=float)
    object_h[:3] = np.asarray(object_pos_world, dtype=float)[:3]
    object_base = np.linalg.inv(base_world) @ object_h
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


def _tuple3(value: Any) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        raise ValueError("WR-06 world pose must contain finite XYZ")
    return float(array[0]), float(array[1]), float(array[2])


__all__ = ["accessible_object_pos_world", "resolve_episode_targets"]
