"""WR-01 station mapping and mobile-base assembly."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import Any
from bench.execution.planning import NavigationConfig

def build_navigation_config(settings: dict[str, Any], *, manual_base_path_file: Path) -> NavigationConfig:
    values = dict(settings)
    config = replace(NavigationConfig(), kinematic_base_navigation=bool(values.pop('kinematic_base_navigation', False)), base_action_frame=str(values.pop('base_action_frame', 'world')), base_translation_action_sign=float(values.pop('base_translation_action_sign', 1.0)), base_tolerance=float(values.pop('base_tolerance', 0.04)), base_yaw_tolerance=float(values.pop('base_yaw_tolerance', 0.08)), base_max_steps=int(values.pop('base_max_steps', 1500)), manual_base_path_enabled=bool(values.pop('manual_base_path_enabled', True)), manual_base_path_file=Path(manual_base_path_file), manual_base_path_start_index=int(values.pop('manual_base_path_start_index', 0)), manual_base_path_xy_tolerance=float(values.pop('manual_base_path_xy_tolerance', 0.04)), manual_base_path_yaw_tolerance=float(values.pop('manual_base_path_yaw_tolerance', 0.08)), base_profile_v_max=float(values.pop('base_profile_v_max', 0.3)), base_profile_a_max=float(values.pop('base_profile_a_max', 0.45)), base_profile_a_dec=float(values.pop('base_profile_a_dec', 0.45)), base_profile_v_min=float(values.pop('base_profile_v_min', 0.08)), base_profile_w_max=float(values.pop('base_profile_w_max', 0.5)), base_profile_alpha_max=float(values.pop('base_profile_alpha_max', 0.8)), base_profile_alpha_dec=float(values.pop('base_profile_alpha_dec', 0.8)), base_profile_w_min=float(values.pop('base_profile_w_min', 0.08)), use_station_astar_paths=bool(values.pop('use_station_astar_paths', False)), station_nav_use_docking=bool(values.pop('station_nav_use_docking', False)), station_nav_dock_standoff=float(values.pop('station_nav_dock_standoff', 0.0)), station_nav_require_footprint=bool(values.pop('station_nav_require_footprint', False)))
    if values:
        raise ValueError('unknown WR-01 navigation settings: ' + ', '.join(sorted(values)))
    return config
