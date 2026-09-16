"""WA-01 特有的 station 映射和移动底盘装配。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation import (
    BaseDriverAdapter,
    ManualBasePathSample,
    NavigationController,
    RoboCasaNavigationAdapter,
    StationPathPlanner,
    StationPoseSampler,
    station_specs_from_candidates,
)
from bench.execution.navigation.env_adapter import callable_xy, fixture_xy, object_xy, safe_name
from bench.execution.planning import NavigationConfig
from bench.execution.planning.nav_algo import normalize_angle
from bench.execution.sampling.circle_sampling import sample_uniform_yaw
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

DEFAULT_INITIAL_YAW_HALF_RANGE_RAD = float(np.deg2rad(5.0))
DEFAULT_TURNAROUND_DELTA_RAD = float(np.pi)
DEFAULT_TURNAROUND_HALF_RANGE_RAD = float(np.deg2rad(5.0))
INITIAL_YAW_SALT = 0x57413059
TURNAROUND_SALT = 0x57413054
_NAV_RANDOMIZATION_KEYS = (
    "initial_yaw_half_range_rad",
    "turnaround_delta_rad",
    "turnaround_half_range_rad",
)


@dataclass(frozen=True)
class StationRandomization:
    """XY 圆之外的朝向随机：仅初始 yaw 与原地转向。"""

    initial_yaw_half_range_rad: float = DEFAULT_INITIAL_YAW_HALF_RANGE_RAD
    turnaround_delta_rad: float = DEFAULT_TURNAROUND_DELTA_RAD
    turnaround_half_range_rad: float = DEFAULT_TURNAROUND_HALF_RANGE_RAD


@dataclass(frozen=True)
class NavigationAssembly:
    """汇总 Base 导航控制器及其 sampler、driver 和 adapter。"""

    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


def resolve_station_randomization(
    settings: Mapping[str, Any] | None = None,
) -> StationRandomization:
    """从任务 YAML 读取朝向随机参数，缺省为 180° ± 5°。"""

    values = dict(settings or {})
    return StationRandomization(
        initial_yaw_half_range_rad=float(
            values.get("initial_yaw_half_range_rad", DEFAULT_INITIAL_YAW_HALF_RANGE_RAD)
        ),
        turnaround_delta_rad=float(
            values.get("turnaround_delta_rad", DEFAULT_TURNAROUND_DELTA_RAD)
        ),
        turnaround_half_range_rad=float(
            values.get("turnaround_half_range_rad", DEFAULT_TURNAROUND_HALF_RANGE_RAD)
        ),
    )


def sample_initial_yaw(center_yaw: float, *, seed: int, randomization: StationRandomization) -> float:
    """在标定朝向周围独立采样 reset yaw。"""

    return normalize_angle(
        sample_uniform_yaw(
            float(center_yaw),
            float(randomization.initial_yaw_half_range_rad),
            seed=int(seed),
            salt=INITIAL_YAW_SALT,
        )
    )


def sample_turnaround_yaw(
    current_yaw: float,
    *,
    seed: int,
    draw_index: int,
    randomization: StationRandomization,
) -> tuple[float, float]:
    """相对当前朝向采样一次原地转向，返回 (目标 yaw, 实际转角)。"""

    draw_seed = int(
        np.random.SeedSequence(
            [int(seed), TURNAROUND_SALT & 0xFFFFFFFF, int(draw_index)]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    delta = sample_uniform_yaw(
        float(randomization.turnaround_delta_rad),
        float(randomization.turnaround_half_range_rad),
        seed=int(draw_seed),
        salt=TURNAROUND_SALT,
    )
    return normalize_angle(float(current_yaw) + float(delta)), float(delta)


def build_station_sampler(
    config: NavigationConfig,
    *,
    seed: int,
    base_xy_provider: Any | None = None,
) -> StationPoseSampler:
    """按 episode seed 构建每次访问重抽 XY 的 station sampler。"""

    return StationPoseSampler.from_file(
        config.station_base_pose_file,
        config=config,
        seed=int(seed),
        base_xy_provider=base_xy_provider,
        station_aliases=dict(STATION_ALIASES),
    )


def build_navigation_config(
    settings: dict[str, Any],
    *,
    manual_base_path_file: Path,
    station_base_pose_file: Path,
) -> NavigationConfig:
    """将任务导航 YAML 转换为通用 NavigationConfig 并拒绝未知字段。"""

    values = dict(settings)
    for key in _NAV_RANDOMIZATION_KEYS:
        values.pop(key, None)
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


class TurnaroundPathDriver(BaseDriverAdapter):
    """跨站时原地转 180±5°，再锁 yaw 平移到本次采样的终点 XY。"""

    def __init__(
        self,
        env: Any,
        obs: dict[str, Any],
        *,
        config: NavigationConfig,
        sampler: StationPoseSampler,
        randomization: StationRandomization | None = None,
    ) -> None:
        super().__init__(env, obs, nav_config=config)
        self._station_sampler = sampler
        self._randomization = randomization or StationRandomization()
        self._turn_count = 0

    def plan_and_move(self, *args: Any, **kwargs: Any) -> Any:
        target_xy = np.asarray(args[0] if args else kwargs.get("target_xy"), dtype=float).reshape(-1)
        station_name = str(kwargs.get("station_name", ""))
        purpose = kwargs.get("purpose")
        holding = kwargs.get("holding")
        executor = self._manual_path_executor()
        dest_yaw = None
        delta = None
        if (
            executor is not None
            and target_xy.size >= 2
            and np.all(np.isfinite(target_xy[:2]))
        ):
            dest_index, _ = executor.target_index_for(
                station_name, purpose=purpose, holding=holding
            )
            if dest_index is not None:
                if executor.current_index is None:
                    self.refresh_obs()
                    executor.current_index, _ = executor.nearest_index(self._obs)
                dest_xy = target_xy[:2].astype(float)
                if int(dest_index) != int(executor.current_index):
                    current_yaw = float(self.current_base_yaw())
                    dest_yaw, delta = sample_turnaround_yaw(
                        current_yaw,
                        seed=int(self._station_sampler.selector.seed),
                        draw_index=self._turn_count,
                        randomization=self._randomization,
                    )
                    self._turn_count += 1
                    kwargs["use_sampled_terminal_pose"] = True
                    if args:
                        args = (dest_xy, dest_yaw, *args[2:])
                    else:
                        kwargs["target_xy"] = dest_xy
                        kwargs["target_yaw"] = dest_yaw
                nominal = executor.samples[int(dest_index)]
                executor.samples[int(dest_index)] = ManualBasePathSample(
                    index=int(dest_index),
                    xy=dest_xy,
                    yaw=float(
                        dest_yaw if dest_yaw is not None else nominal.yaw
                    ),
                    name=nominal.name,
                )
        result = super().plan_and_move(*args, **kwargs)
        if delta is not None:
            details = dict(getattr(result, "details", {}) or {})
            details["turnaround"] = {
                "draw_index": int(self._turn_count - 1),
                "delta_rad": float(delta),
                "target_xy": np.asarray(target_xy[:2], dtype=float).tolist(),
                "target_yaw": None if dest_yaw is None else float(dest_yaw),
            }
            result.details = details
        return result


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    config: NavigationConfig,
    seed: int,
    base_driver: Any | None = None,
    station_sampler: StationPoseSampler | None = None,
    randomization: StationRandomization | None = None,
) -> NavigationAssembly:
    """创建 WA-01 使用的 Base driver、station sampler 和导航控制器。"""

    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    resolved_randomization = randomization or resolve_station_randomization()
    if config.use_station_base_poses:
        sampler: Any = station_sampler or build_station_sampler(config, seed=int(seed))
        driver = base_driver or TurnaroundPathDriver(
            env,
            obs,
            config=config,
            sampler=sampler,
            randomization=resolved_randomization,
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
        driver = base_driver or BaseDriverAdapter(env, obs, nav_config=config)
    pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    if isinstance(sampler, StationPoseSampler):
        sampler.base_xy_provider = pose_provider
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
    "StationPoseSampler",
    "StationRandomization",
    "TurnaroundPathDriver",
    "build_navigation_assembly",
    "build_navigation_config",
    "build_station_sampler",
    "resolve_station_randomization",
    "sample_initial_yaw",
    "sample_turnaround_yaw",
    "station_specs",
]
