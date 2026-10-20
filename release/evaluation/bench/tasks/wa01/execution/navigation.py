"""WA-01 特有的 station 映射和移动底盘装配。"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import numpy as np
from bench.execution.navigation import StationPoseSampler
from bench.execution.planning import NavigationConfig
from bench.execution.planning.nav_algo import normalize_angle
from bench.execution.sampling.circle_sampling import sample_uniform_yaw
STATION_ALIASES: dict[Any, str] = {'coffee_A': 'coffee_machine_station', 'coffee_B': 'coffee_machine_station', 'coffee_machine': 'coffee_machine_station', 'sugar_cube': 'coffee_machine_station', 'sugar_plate': 'coffee_machine_station', 'dining_counter': 'dining_table_station', 'dining_delivery': 'dining_table_station', 'dining_table': 'dining_table_station', 'table': 'dining_table_station'}
DEFAULT_INITIAL_YAW_HALF_RANGE_RAD = float(np.deg2rad(5.0))
DEFAULT_TURNAROUND_DELTA_RAD = float(np.pi)
DEFAULT_TURNAROUND_HALF_RANGE_RAD = float(np.deg2rad(5.0))
INITIAL_YAW_SALT = 1463890009
_NAV_RANDOMIZATION_KEYS = ('initial_yaw_half_range_rad', 'turnaround_delta_rad', 'turnaround_half_range_rad')

@dataclass(frozen=True)
class StationRandomization:
    """XY 圆之外的朝向随机：仅初始 yaw 与原地转向。"""
    initial_yaw_half_range_rad: float = DEFAULT_INITIAL_YAW_HALF_RANGE_RAD
    turnaround_delta_rad: float = DEFAULT_TURNAROUND_DELTA_RAD
    turnaround_half_range_rad: float = DEFAULT_TURNAROUND_HALF_RANGE_RAD

def resolve_station_randomization(settings: Mapping[str, Any] | None=None) -> StationRandomization:
    """从任务 YAML 读取朝向随机参数，缺省为 180° ± 5°。"""
    values = dict(settings or {})
    return StationRandomization(initial_yaw_half_range_rad=float(values.get('initial_yaw_half_range_rad', DEFAULT_INITIAL_YAW_HALF_RANGE_RAD)), turnaround_delta_rad=float(values.get('turnaround_delta_rad', DEFAULT_TURNAROUND_DELTA_RAD)), turnaround_half_range_rad=float(values.get('turnaround_half_range_rad', DEFAULT_TURNAROUND_HALF_RANGE_RAD)))

def sample_initial_yaw(center_yaw: float, *, seed: int, randomization: StationRandomization) -> float:
    """在标定朝向周围独立采样 reset yaw。"""
    return normalize_angle(sample_uniform_yaw(float(center_yaw), float(randomization.initial_yaw_half_range_rad), seed=int(seed), salt=INITIAL_YAW_SALT))

def build_station_sampler(config: NavigationConfig, *, seed: int, base_xy_provider: Any | None=None) -> StationPoseSampler:
    """按 episode seed 构建每次访问重抽 XY 的 station sampler。"""
    return StationPoseSampler.from_file(config.station_base_pose_file, config=config, seed=int(seed), base_xy_provider=base_xy_provider, station_aliases=dict(STATION_ALIASES))

def build_navigation_config(settings: dict[str, Any], *, manual_base_path_file: Path, station_base_pose_file: Path) -> NavigationConfig:
    """将任务导航 YAML 转换为通用 NavigationConfig 并拒绝未知字段。"""
    values = dict(settings)
    for key in _NAV_RANDOMIZATION_KEYS:
        values.pop(key, None)
    config = replace(NavigationConfig(), kinematic_base_navigation=bool(values.pop('kinematic_base_navigation', False)), base_action_frame=str(values.pop('base_action_frame', 'world')), base_translation_action_sign=float(values.pop('base_translation_action_sign', 1.0)), base_tolerance=float(values.pop('base_tolerance', 0.025)), base_yaw_tolerance=float(values.pop('base_yaw_tolerance', 0.06)), base_max_steps=int(values.pop('base_max_steps', 1500)), manual_base_path_enabled=bool(values.pop('manual_base_path_enabled', True)), manual_base_path_file=Path(manual_base_path_file), manual_base_path_start_index=int(values.pop('manual_base_path_start_index', 0)), manual_base_path_xy_tolerance=float(values.pop('manual_base_path_xy_tolerance', 0.03)), manual_base_path_yaw_tolerance=float(values.pop('manual_base_path_yaw_tolerance', 0.06)), base_single_axis_tracking=bool(values.pop('base_single_axis_tracking', False)), base_terminal_yaw_distance=float(values.pop('base_terminal_yaw_distance', 0.05)), base_snap_to_target_after_success=bool(values.pop('base_snap_to_target_after_success', False)), base_profile_v_max=float(values.pop('base_profile_v_max', 0.75)), base_profile_a_max=float(values.pop('base_profile_a_max', 1.3)), base_profile_a_dec=float(values.pop('base_profile_a_dec', 1.5)), base_profile_v_min=float(values.pop('base_profile_v_min', 0.15)), base_profile_w_max=float(values.pop('base_profile_w_max', 0.75)), base_profile_alpha_max=float(values.pop('base_profile_alpha_max', 1.5)), base_profile_alpha_dec=float(values.pop('base_profile_alpha_dec', 1.3)), base_profile_w_min=float(values.pop('base_profile_w_min', 0.08)), use_station_astar_paths=bool(values.pop('use_station_astar_paths', False)), station_nav_use_docking=bool(values.pop('station_nav_use_docking', False)), station_nav_dock_standoff=float(values.pop('station_nav_dock_standoff', 0.0)), station_nav_require_footprint=bool(values.pop('station_nav_require_footprint', False)), station_base_pose_file=Path(station_base_pose_file), use_station_base_poses=bool(values.pop('use_station_base_poses', True)))
    if values:
        unknown = ', '.join(sorted(values))
        raise ValueError(f'unknown WA-01 navigation settings: {unknown}')
    return config
