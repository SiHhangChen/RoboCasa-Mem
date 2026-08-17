"""Bind pure WR-05 episode decisions to reset-time world geometry."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.planning import matrix_from_base_obs
from bench.tasks.wr05.domain.contracts import (
    CONDIMENT_IDS,
    EpisodePlan,
    EpisodeTargets,
)


def resolve_episode_targets(env: Any, episode: EpisodePlan) -> EpisodeTargets:
    use_poses = {
        name: _tuple3(env.use_station_pose(name)) for name in CONDIMENT_IDS
    }
    home_poses = {
        name: _tuple3(env.return_region_pose(episode.home_assignment[name]))
        for name in CONDIMENT_IDS
    }
    desired_centers = {
        name: np.asarray(home_poses[name], dtype=float)[:2] for name in CONDIMENT_IDS
    }
    result: dict[str, Any] = {}
    for name in CONDIMENT_IDS:
        sampled = np.asarray(episode.return_sampled_offsets_xy[name], dtype=float)
        result[name] = {
            "target_region": episode.home_assignment[name],
            "sampled_xy_offset": sampled.astype(float).tolist(),
            "sampled_target_xy": (
                desired_centers[name] + sampled
            ).astype(float).tolist(),
            "sample_radius": float(np.linalg.norm(sampled)),
        }
    result["diagnostics"] = dict(episode.return_sampling_metadata)
    return EpisodeTargets(
        use_pose_by_condiment=use_poses,
        home_pose_by_condiment=home_poses,
        return_place_sampling=result,
        original_home_pose_by_condiment={
            name: _tuple3(env.original_home_pose_by_condiment[name])
            for name in CONDIMENT_IDS
        },
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
        raise ValueError("WR-05 world pose must contain finite XYZ")
    return float(array[0]), float(array[1]), float(array[2])


__all__ = ["accessible_object_pos_world", "resolve_episode_targets"]
