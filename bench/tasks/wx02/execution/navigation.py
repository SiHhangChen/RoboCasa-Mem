"""WX-02 station mapping and mobile Base assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from bench.execution.manipulation import (
    evaluate_object_transport,
    get_object_hold_status,
    get_object_transport_snapshot,
)
from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
    RoboCasaNavigationAdapter,
    StationPathPlanner,
    StationPoseSampler,
    station_specs_from_candidates,
)
from bench.execution.navigation.env_adapter import fixture_xy, object_xy, safe_name
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler


STATION_ALIASES: dict[Any, str] = {
    "sink": "cleaning_tool_station",
    "counter": "cleaning_tool_station",
    "tool_origin": "cleaning_tool_station",
    "cleaning_tool": "cleaning_tool_station",
    "cleaning_tool_region": "cleaning_tool_station",
    "dining_table": "dining_table_station",
    "dining_counter": "dining_table_station",
    "primary_cup": "dining_table_station",
    "secondary_cup": "dining_table_station",
    "object_origin": "dining_table_station",
    "original_area": "dining_table_station",
}


@dataclass(frozen=True)
class NavigationAssembly:
    """Concrete navigation objects used by the WX-02 bundle."""

    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


class HoldingVerifiedNavigationController:
    """Reject navigation that leaves a declared held object behind."""

    def __init__(
        self,
        controller: NavigationController,
        env: Any,
        *,
        max_end_effector_distance: float,
        max_relative_offset_drift: float,
        minimum_motion_ratio: float,
    ) -> None:
        self._controller = controller
        self._env = env
        self._max_end_effector_distance = float(max_end_effector_distance)
        self._max_relative_offset_drift = float(max_relative_offset_drift)
        self._minimum_motion_ratio = float(minimum_motion_ratio)

    def navigate_to(self, station_name: str, **kwargs: Any):
        holding = kwargs.get("holding")
        before = (
            None
            if holding is None
            else get_object_transport_snapshot(self._env, str(holding))
        )
        result = self._controller.navigate_to(station_name, **kwargs)
        if holding is None:
            return result

        after = get_object_transport_snapshot(self._env, str(holding))
        transport = evaluate_object_transport(
            before or {},
            after,
            minimum_end_effector_motion=0.0,
            minimum_motion_ratio=self._minimum_motion_ratio,
            max_relative_offset_drift=self._max_relative_offset_drift,
        )
        hold = get_object_hold_status(
            self._env,
            str(holding),
            max_end_effector_distance=self._max_end_effector_distance,
        )
        checks = {
            "navigation": bool(result.success),
            "object_followed_end_effector": bool(transport.get("pass", False)),
            "object_still_physically_held": bool(hold.get("held", False)),
        }
        result.details["holding_verification"] = {
            "object_name": str(holding),
            "transport": transport,
            "final_hold": hold,
            "checks": checks,
            "pass": bool(all(checks.values())),
        }
        if not bool(all(checks.values())):
            result.success = False
            result.failure_code = result.failure_code or "held_object_transport_failed"
            result.details["pass"] = False
            result.details["reason"] = result.failure_code
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._controller, name)


def build_navigation_config(
    settings: Mapping[str, Any],
    *,
    manual_base_path_file: Path,
    station_base_pose_file: Path,
) -> NavigationConfig:
    """Convert strict task YAML into the shared NavigationConfig."""

    values = dict(settings)
    config = replace(
        NavigationConfig(),
        kinematic_base_navigation=bool(values.pop("kinematic_base_navigation", False)),
        base_action_frame=str(values.pop("base_action_frame", "world")),
        base_translation_action_sign=float(
            values.pop("base_translation_action_sign", 1.0)
        ),
        base_tolerance=float(values.pop("base_tolerance", 0.025)),
        base_yaw_tolerance=float(values.pop("base_yaw_tolerance", 0.06)),
        base_max_steps=int(values.pop("base_max_steps", 1500)),
        manual_base_path_enabled=bool(values.pop("manual_base_path_enabled", True)),
        manual_base_path_file=Path(manual_base_path_file),
        manual_base_path_start_index=int(values.pop("manual_base_path_start_index", 0)),
        manual_base_path_xy_tolerance=float(
            values.pop("manual_base_path_xy_tolerance", 0.030)
        ),
        manual_base_path_yaw_tolerance=float(
            values.pop("manual_base_path_yaw_tolerance", 0.06)
        ),
        base_single_axis_tracking=bool(values.pop("base_single_axis_tracking", False)),
        base_terminal_yaw_distance=float(
            values.pop("base_terminal_yaw_distance", 0.05)
        ),
        base_snap_to_target_after_success=bool(
            values.pop("base_snap_to_target_after_success", False)
        ),
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
        station_nav_require_footprint=bool(
            values.pop("station_nav_require_footprint", False)
        ),
        station_base_pose_file=Path(station_base_pose_file),
        use_station_base_poses=bool(values.pop("use_station_base_poses", True)),
    )
    if values:
        raise ValueError(
            "unknown WX-02 navigation settings: " + ", ".join(sorted(values))
        )
    return config


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Derive task station anchors from the current fixtures and objects."""

    tool_xy = object_xy(env, "cleaning_tool")
    primary_xy = object_xy(env, "primary_cup")
    return {
        "cleaning_tool_station": {
            "station_id": "cleaning_tool_station",
            "fixture": getattr(env, "dining_table", None),
            "anchor_xy": (
                tool_xy
                if tool_xy is not None
                else fixture_xy(getattr(env, "dining_table", None))
            ),
            "source_fixture": safe_name(getattr(env, "dining_table", None)),
            "source_object_or_phase": "cleaning_tool",
        },
        "dining_table_station": {
            "station_id": "dining_table_station",
            "fixture": getattr(env, "dining_table", None),
            "anchor_xy": (
                primary_xy
                if primary_xy is not None
                else fixture_xy(getattr(env, "dining_table", None))
            ),
            "source_fixture": safe_name(getattr(env, "dining_table", None)),
            "source_object_or_phase": "primary_cup,secondary_cup,original_areas",
        },
    }


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    config: NavigationConfig,
    seed: int,
    base_driver: Any | None = None,
    holding_max_end_effector_distance: float = 0.10,
    holding_max_relative_offset_drift: float = 0.02,
    holding_minimum_motion_ratio: float = 0.70,
) -> NavigationAssembly:
    """Create the adapter, sampler, optional planner and navigation controller."""

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
            aliases={
                key: value
                for key, value in STATION_ALIASES.items()
                if isinstance(key, str)
            },
        )
        sampler = SimpleStationSampler(
            stations=generated,
            seeds=SeedStream(base_seed=int(seed), namespace="wx_02_navigation"),
        )
    planner = None
    if not config.manual_base_path_enabled and config.use_station_astar_paths:
        planner = StationPathPlanner(
            obstacles=adapter.build_fixture_obstacles, config=config
        )
    raw_controller = NavigationController(
        station_sampler=sampler,
        base_driver=driver,
        path_planner=planner,
        base_pose_provider=pose_provider,
    )
    controller = HoldingVerifiedNavigationController(
        raw_controller,
        env,
        max_end_effector_distance=holding_max_end_effector_distance,
        max_relative_offset_drift=holding_max_relative_offset_drift,
        minimum_motion_ratio=holding_minimum_motion_ratio,
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
    "HoldingVerifiedNavigationController",
    "STATION_ALIASES",
    "build_navigation_assembly",
    "build_navigation_config",
    "station_specs",
]
