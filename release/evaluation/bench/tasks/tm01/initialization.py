"""TM01 reset-time mobile-base initialization."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from bench.execution.navigation.base_motion import base_yaw, set_base_pose_kinematic
from bench.execution.navigation.station_poses import quat_xyzw_to_yaw
from bench.tasks.tm01.execution.navigation import sample_episode_path_poses

def apply_manual_path_initial_base_pose(env: Any, obs: dict[str, Any], *, path_file: str | Path, sample_index: int, seed: int=0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Place the reset-time base at the episode-sampled path pose."""
    path = Path(path_file)
    payload = json.loads(path.read_text(encoding='utf-8'))
    raw_samples = {int(item.get('index', fallback)): dict(item) for fallback, item in enumerate(payload.get('samples', ()))}
    index = int(sample_index)
    if index not in raw_samples:
        raise KeyError(f'TM01 initial manual-path sample {index} is unavailable')
    poses, summaries = sample_episode_path_poses(payload, seed=int(seed))
    pose = poses[index]
    xy = np.asarray(pose.position[:2], dtype=float)
    yaw = quat_xyzw_to_yaw(pose.quaternion)
    sample = raw_samples[index]
    before_xy = np.asarray(obs['robot0_base_pos'], dtype=float).reshape(-1)[:2]
    before_yaw = float(base_yaw(obs))
    updated = set_base_pose_kinematic(env, target_xy=xy, target_yaw=yaw)
    after_xy = np.asarray(updated['robot0_base_pos'], dtype=float).reshape(-1)[:2]
    after_yaw = float(base_yaw(updated))
    return (updated, {'mode': 'manual_path_sample', 'recorded_in_hdf5_actions': False, 'manual_base_path_file': str(path), 'sample_index': index, 'sample_name': sample.get('name'), 'episode_seed': int(seed), 'sampling': dict(summaries[index]), 'before_xy': before_xy.astype(float).tolist(), 'before_yaw': before_yaw, 'target_xy': xy.astype(float).tolist(), 'target_yaw': yaw, 'after_xy': after_xy.astype(float).tolist(), 'after_yaw': after_yaw})
