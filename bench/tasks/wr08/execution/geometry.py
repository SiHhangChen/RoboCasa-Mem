"""Bind WR-08 episode facts to live world geometry."""
from __future__ import annotations
from typing import Any
import numpy as np
from bench.tasks.wr08.domain.contracts import EpisodePlan, EpisodeTargets, ITEM_IDS

def resolve_episode_targets(env: Any, episode: EpisodePlan) -> EpisodeTargets:
    return EpisodeTargets(target_plate_pose_by_item={item: _tuple3(env.target_plate_pose(episode.target_slot_by_color[episode.source_color_by_item[item]], item)) for item in ITEM_IDS})

def _tuple3(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        raise ValueError('WR-08 world pose must contain finite XYZ')
    return tuple((float(v) for v in arr[:3]))
