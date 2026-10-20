"""TM01 station poses and manual-path mobile-base assembly."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import field, fields, replace
from pathlib import Path
from typing import Any
import numpy as np
from bench.execution.navigation.contracts import Pose
from bench.execution.navigation.station_poses import yaw_to_quat_xyzw
from bench.execution.planning import NavigationConfig
from bench.execution.planning.nav_algo import normalize_angle
from bench.execution.sampling.circle_sampling import sample_uniform_disk, sample_uniform_yaw
from bench.execution.sampling.rng import SeedStream
MAIN_LOOP_TURN_INDICES = (19, 4, 6, 7, 20, 14, 16, 18)

def build_navigation_config(settings: dict[str, Any], *, manual_base_path_file: Path) -> NavigationConfig:
    """Make the measured manual path authoritative for TM01 navigation."""
    values = dict(settings)
    allowed = {field.name for field in fields(NavigationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError('unknown TM01 navigation settings: ' + ', '.join(unknown))
    values.update({'manual_base_path_enabled': True, 'manual_base_path_file': Path(manual_base_path_file), 'station_base_pose_file': None, 'use_station_base_poses': False, 'use_station_astar_paths': False})
    return replace(NavigationConfig(), **values)

def sample_episode_path_poses(payload: Mapping[str, Any], *, seed: int) -> tuple[dict[int, Pose], dict[int, dict[str, Any]]]:
    """Resolve one reproducible, closed set of sampled TM01 path poses."""
    raw_samples = list(payload.get('samples', ()))
    samples: dict[int, dict[str, Any]] = {}
    for fallback, raw in enumerate(raw_samples):
        item = dict(raw)
        index = int(item.get('index', fallback))
        if index in samples:
            raise ValueError(f'duplicate TM01 sampled path index {index}')
        samples[index] = item
    if sorted(samples) != list(range(len(samples))):
        raise ValueError('TM01 sampled path indexes must be contiguous from zero')
    seeds = SeedStream(base_seed=int(seed), namespace='tm01_navigation')
    sampled_xy: dict[int, np.ndarray] = {}
    for index, item in samples.items():
        circle = item.get('circle')
        if not isinstance(circle, Mapping):
            raise ValueError(f'TM01 path sample {index} requires a circle')
        sampled_xy[index] = sample_uniform_disk(circle.get('center_xy', ()), float(circle.get('radius_m', -1.0)), seed=seeds.seed('waypoint', index, 'xy'), salt=1414344753)
    turn_offsets = _sample_closed_turn_offsets(samples, seed=seeds.seed('main_loop_turns'))
    resolved_yaw: dict[int, float] = {}
    resolving: set[int] = set()

    def yaw_for(index: int) -> float:
        if index in resolved_yaw:
            return resolved_yaw[index]
        if index in resolving:
            raise ValueError(f'TM01 path yaw dependency cycle at sample {index}')
        if index not in samples:
            raise ValueError(f'TM01 path yaw source {index} is unavailable')
        resolving.add(index)
        item = samples[index]
        turn = item.get('relative_turn')
        inherited_from = item.get('inherit_yaw_from')
        if isinstance(turn, Mapping):
            source_index = int(turn['source_index'])
            nominal = float(turn['nominal_angle_rad'])
            half_range = float(turn['angle_half_range_rad'])
            if index in turn_offsets:
                offset = turn_offsets[index]
            else:
                sampled = sample_uniform_yaw(nominal, half_range, seed=seeds.seed('turn', index), salt=1414877774)
                offset = normalize_angle(sampled - nominal)
            yaw = normalize_angle(yaw_for(source_index) + nominal + offset)
        elif inherited_from is not None:
            yaw = yaw_for(int(inherited_from))
        else:
            circle = item['circle']
            yaw = normalize_angle(float(circle.get('base_yaw', item.get('yaw'))))
        resolving.remove(index)
        resolved_yaw[index] = yaw
        return yaw
    poses: dict[int, Pose] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for index, item in samples.items():
        yaw = yaw_for(index)
        circle = dict(item['circle'])
        summary: dict[str, Any] = {'path_index': int(index), 'sample_name': item.get('name'), 'route_role': 'station_circle', 'target_xy': sampled_xy[index].astype(float).tolist(), 'target_yaw': float(yaw), 'circle': {'center_xy': [float(value) for value in circle['center_xy']], 'radius_m': float(circle['radius_m'])}}
        turn = item.get('relative_turn')
        if isinstance(turn, Mapping):
            nominal = float(turn['nominal_angle_rad'])
            offset = turn_offsets.get(index)
            if offset is None:
                offset = normalize_angle(yaw - yaw_for(int(turn['source_index'])) - nominal)
            summary.update({'route_role': 'relative_turn_circle', 'source_index': int(turn['source_index']), 'nominal_turn_angle': nominal, 'turn_angle_half_range': float(turn['angle_half_range_rad']), 'sampled_turn_offset': float(offset), 'sampled_turn_angle': nominal + float(offset), 'closed_loop_constraint': index in MAIN_LOOP_TURN_INDICES})
        elif item.get('inherit_yaw_from') is not None:
            summary.update({'route_role': 'inherited_yaw_circle', 'yaw_inherited_from_path_index': int(item['inherit_yaw_from'])})
        poses[index] = Pose(position=(float(sampled_xy[index][0]), float(sampled_xy[index][1]), 0.0), quaternion=yaw_to_quat_xyzw(yaw))
        summaries[index] = summary
    return (poses, summaries)

def _sample_closed_turn_offsets(samples: Mapping[int, Mapping[str, Any]], *, seed: int) -> dict[int, float]:
    """Sample bounded turn noise whose sum closes the clockwise main loop."""
    half_ranges = []
    for index in MAIN_LOOP_TURN_INDICES:
        turn = samples[index].get('relative_turn')
        if not isinstance(turn, Mapping):
            raise ValueError(f'TM01 main-loop turn {index} lacks relative_turn')
        half_ranges.append(float(turn['angle_half_range_rad']))
    if not half_ranges or any((value <= 0.0 for value in half_ranges)):
        raise ValueError('TM01 main-loop turn ranges must be positive')
    if not np.allclose(half_ranges, half_ranges[0]):
        raise ValueError('TM01 closed-loop turns must share one half-range')
    half_range = half_ranges[0]
    rng = np.random.default_rng(int(seed))
    for _ in range(10000):
        values = rng.uniform(-half_range, half_range, size=len(MAIN_LOOP_TURN_INDICES))
        values -= float(np.mean(values))
        if float(np.max(np.abs(values))) <= half_range:
            return dict(zip(MAIN_LOOP_TURN_INDICES, values.astype(float).tolist()))
    raise RuntimeError('TM01 could not sample bounded closed-loop turn offsets')
