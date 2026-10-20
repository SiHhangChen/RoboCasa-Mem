"""Resolve WX-01 domain targets into controller-facing world geometry."""
from __future__ import annotations
from typing import Any
import numpy as np
from bench.tasks.wx01.domain.contracts import EpisodePlan, EpisodeTargets

def resolve_episode_targets(env: Any, episode: EpisodePlan) -> EpisodeTargets:
    """Bind pure relative episode offsets to reset-time original world poses."""
    original_tool = env_pose(env, 'original_tool_pose', fallback_object='cleaning_tool')
    original_blocking = env_pose(env, 'original_blocking_object_pose', fallback_object='blocking_object')
    if original_tool is None or original_blocking is None:
        raise ValueError('WX-01 original poses are unavailable after initialization')
    temporary_blocking = np.asarray(original_blocking, dtype=float).copy()
    temporary_blocking[:2] += np.asarray(episode.temporary_blocking_offset_xy, dtype=float)
    post_wipe_tool = np.asarray([original_blocking[0], original_blocking[1], original_tool[2]], dtype=float)
    post_wipe_tool[:2] += np.asarray(episode.temporary_tool_offset_xy, dtype=float)
    return EpisodeTargets(original_tool_pose=_tuple3(original_tool), original_blocking_pose=_tuple3(original_blocking), temporary_blocking_pose=_tuple3(temporary_blocking), post_wipe_tool_pose=_tuple3(post_wipe_tool))

def env_pose(env: Any, attribute: str, *, fallback_object: str | None=None) -> np.ndarray | None:
    """Read a finite XYZ environment attribute with an object-body fallback."""
    value = getattr(env, attribute, None)
    pose = _optional_pose3(value)
    if pose is not None:
        return pose
    if fallback_object is None:
        return None
    try:
        return np.asarray(env.sim.data.body_xpos[env.obj_body_id[fallback_object]], dtype=float)[:3]
    except Exception:
        return None

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
    return (float(array[0]), float(array[1]), float(array[2]))
