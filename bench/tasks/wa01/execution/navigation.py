"""WA-01 特有的 station 映射和移动底盘装配。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
    RoboCasaNavigationAdapter,
    StationPathPlanner,
    StationPoseSampler,
    station_specs_from_candidates,
)
from bench.execution.navigation.env_adapter import callable_xy, fixture_xy, object_xy, safe_name
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler


STATION_ALIASES: dict[Any, str] = {
    "coffee_A": "coffee_machine_station",
    "coffee_B": "coffee_machine_station",
    "coffee_machine": "coffee_machine_station",
    "sugar_cube": "coffee_machine_station",
    "sugar_plate": "coffee_machine_station",
    "dining_counter": "dining_table_station",
    "dining_delivery": "dining_table_station",
    "dining_table": "dining_table_station",
    "table": "dining_table_station",
}


@dataclass(frozen=True)
class NavigationAssembly:
    """汇总 Base 导航控制器及其 sampler、driver 和 adapter。"""

    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


def build_navigation_config(
    settings: dict[str, Any],
    *,
    manual_base_path_file: Path,
    station_base_pose_file: Path,
) -> NavigationConfig:
    """将任务导航 YAML 转换为通用 NavigationConfig 并拒绝未知字段。"""

    values = dict(settings)
    config = replace(
        NavigationConfig(),
        kinematic_base_navigation=bool(values.pop("kinematic_base_navigation", False)),
        base_action_frame=str(values.pop("base_action_frame", "world")),
        base_translation_action_sign=float(values.pop("base_translation_action_sign", 1.0)),
        base_tolerance=float(values.pop("base_tolerance", 0.025)),
        base_yaw_tolerance=float(values.pop("base_yaw_tolerance", 0.06)),
        base_max_steps=int(values.pop("base_max_steps", 1500)),
        manual_base_path_enabled=bool(values.pop("manual_base_path_enabled", True)),
        manual_base_path_file=Path(manual_base_path_file),
        manual_base_path_start_index=int(values.pop("manual_base_path_start_index", 0)),
        manual_base_path_xy_tolerance=float(values.pop("manual_base_path_xy_tolerance", 0.030)),
        manual_base_path_yaw_tolerance=float(values.pop("manual_base_path_yaw_tolerance", 0.06)),
        base_single_axis_tracking=bool(values.pop("base_single_axis_tracking", False)),
        base_terminal_yaw_distance=float(values.pop("base_terminal_yaw_distance", 0.05)),
        base_snap_to_target_after_success=bool(values.pop("base_snap_to_target_after_success", False)),
        base_profile_v_max=float(values.pop("base_profile_v_max", 0.75)),
        base_profile_a_max=float(values.pop("base_profile_a_max", 1.30)),
        base_profile_a_dec=float(values.pop("base_profile_a_dec", 1.50)),
        base_profile_v_min=float(values.pop("base_profile_v_min", 0.15)),
        base_profile_w_max=float(values.pop("base_profile_w_max", 0.75)),
        base_profile_alpha_max=float(values.pop("base_profile_alpha_max", 1.50)),
        base_profile_alpha_dec=float(values.pop("base_profile_alpha_dec", 1.30)),
        base_profile_w_min=float(values.pop("base_profile_w_min", 0.08)),
        use_station_astar_paths=bool(values.pop("use_station_astar_paths", False)),
        station_nav_use_docking=bool(values.pop("station_nav_use_docking", False)),
        station_nav_dock_standoff=float(values.pop("station_nav_dock_standoff", 0.0)),
        station_nav_require_footprint=bool(values.pop("station_nav_require_footprint", False)),
        station_base_pose_file=Path(station_base_pose_file),
        use_station_base_poses=bool(values.pop("use_station_base_poses", True)),
    )
    if values:
        unknown = ", ".join(sorted(values))
        raise ValueError(f"unknown WA-01 navigation settings: {unknown}")
    return config


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """从当前环境物品和 fixture 计算咖啡机、餐桌 station 描述。"""

    coffee_xy = callable_xy(
        getattr(env, "get_coffee_machine_pour_site_world_position", None)
    )
    if coffee_xy is None:
        coffee_xy = fixture_xy(getattr(env, "coffee_machine", None))
    points = [
        point
        for point in (
            coffee_xy,
            object_xy(env, "coffee_A"),
            object_xy(env, "coffee_B"),
            object_xy(env, "sugar_plate"),
        )
        if point is not None
    ]
    coffee_station_xy = np.mean(np.vstack(points), axis=0) if points else coffee_xy
    dining_xy = fixture_xy(getattr(env, "dining_counter", None))
    return {
        "coffee_machine_station": {
            "station_id": "coffee_machine_station",
            "fixture": getattr(env, "counter", None),
            "anchor_xy": coffee_station_xy,
            "source_fixture": safe_name(getattr(env, "coffee_machine", None)),
            "source_object_or_phase": "coffee_machine,mugs,sugar_plate",
        },
        "dining_table_station": {
            "station_id": "dining_table_station",
            "fixture": getattr(env, "dining_counter", None),
            "anchor_xy": dining_xy,
            "source_fixture": safe_name(getattr(env, "dining_counter", None)),
            "source_object_or_phase": "dining_delivery",
        },
    }


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    config: NavigationConfig,
    seed: int,
    base_driver: Any | None = None,
) -> NavigationAssembly:
    """创建 WA-01 使用的 Base driver、station sampler 和导航控制器。"""

    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    driver = base_driver or BaseDriverAdapter(env, obs, nav_config=config)
    pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    if config.use_station_base_poses:
        sampler: Any = StationPoseSampler.from_file(
            config.station_base_pose_file,
            config=config,
            seed=int(seed),
            base_xy_provider=pose_provider,
            station_aliases=dict(STATION_ALIASES),
        )
    else:
        generated = station_specs_from_candidates(
            adapter.station_candidates(),
            aliases={key: value for key, value in STATION_ALIASES.items() if isinstance(key, str)},
        )
        sampler = SimpleStationSampler(
            stations=generated,
            seeds=SeedStream(base_seed=int(seed), namespace="wa_01_navigation"),
        )
    planner = None
    if not config.manual_base_path_enabled and config.use_station_astar_paths:
        planner = StationPathPlanner(obstacles=adapter.build_fixture_obstacles, config=config)
    controller = NavigationController(
        station_sampler=sampler,
        base_driver=driver,
        path_planner=planner,
        base_pose_provider=pose_provider,
    )
    return NavigationAssembly(
        controller=controller,
        env_adapter=adapter,
        station_sampler=sampler,
        path_planner=planner,
        base_driver=driver,
        config=config,
    )


__all__ = [
    "NavigationAssembly",
    "STATION_ALIASES",
    "build_navigation_assembly",
    "build_navigation_config",
    "station_specs",
]
