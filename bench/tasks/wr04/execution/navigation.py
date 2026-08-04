"""WR-04 station mapping and mobile-base assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
    RoboCasaNavigationAdapter,
    StationPathPlanner,
    StationPoseSampler,
    station_specs_from_candidates,
)
from bench.execution.navigation.base_motion import qpos_for_controller, step_base_and_hold_arm
from bench.execution.navigation.station_poses import yaw_to_quat_xyzw
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.tasks.wr04.domain.contracts import HOME_REGIONS, ITEM_ID
from bench.tasks.wr04.execution.stations import station_aliases, station_specs


@dataclass(frozen=True)
class NavigationAssembly:
    """Concrete navigation objects used by WR-04."""

    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


class ObjectRelativeShelfStationSampler:
    """Follow shelf pick or sampled return XY with calibrated yaw/torso."""

    def __init__(
        self,
        delegate: StationPoseSampler,
        env: Any,
        *,
        return_target_xy_by_region: Mapping[str, tuple[float, float]] | None = None,
    ):
        self._delegate = delegate
        self._env = env
        self._return_target_xy_by_region = {
            str(region): _finite_xy(target_xy, label=f"return target for {region}")
            for region, target_xy in dict(return_target_xy_by_region or {}).items()
        }
        self._summary_by_station: dict[str, dict[str, Any]] = {}

    def sample(self, station_name: str, *, purpose: str | None = None) -> Pose:
        pose = self._delegate.sample(station_name, purpose=purpose)
        summary = dict(self._delegate.last_summary(station_name) or {})
        sample = dict(summary.get("sample") or {})
        settings = sample.get("object_relative_xy")
        if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
            self._summary_by_station[station_name] = summary
            return pose
        required_purpose = str(settings.get("purpose", "shelf_pick"))
        actual_purpose = str(purpose or "")
        home_region = str(settings.get("home_region") or "")
        follows_return_target = bool(
            actual_purpose == "home_return"
            and home_region in self._return_target_xy_by_region
        )
        if actual_purpose != required_purpose and not follows_return_target:
            self._summary_by_station[station_name] = summary
            return pose

        object_name = self._object_for_home_region(home_region)
        reference_xy = (
            np.array(self._return_target_xy_by_region[home_region], dtype=float)
            if follows_return_target
            else _finite_xy(
                self._env._obj_pos(object_name),
                label=f"current object position for {object_name}",
            )
        )
        base_minus_object_xy = _finite_xy(
            settings.get("base_minus_object_xy"),
            label=f"base-minus-object offset for {station_name}",
        )
        target_xy = reference_xy + base_minus_object_xy
        adjusted = Pose(
            position=(
                float(target_xy[0]),
                float(target_xy[1]),
                float(pose.position[2]),
            ),
            quaternion=pose.quaternion,
            torso_height=pose.torso_height,
        )
        summary["target_xy"] = target_xy.astype(float).tolist()
        summary["object_relative_xy"] = {
            "enabled": True,
            "purpose": actual_purpose,
            "home_region": home_region,
            "object_name": object_name,
            "reference_kind": (
                "sampled_return_target" if follows_return_target else "current_object"
            ),
            "object_xy": reference_xy.astype(float).tolist(),
            "base_minus_object_xy": base_minus_object_xy.astype(float).tolist(),
            "reference_base_xy": [
                float(pose.position[0]),
                float(pose.position[1]),
            ],
            "adjusted_base_xy": target_xy.astype(float).tolist(),
        }
        self._summary_by_station[station_name] = summary
        return adjusted

    def last_summary(self, station_name: str) -> dict[str, Any] | None:
        summary = self._summary_by_station.get(station_name)
        if summary is not None:
            return summary
        return self._delegate.last_summary(station_name)

    def include_points(self) -> list[np.ndarray]:
        return self._delegate.include_points()

    def reset_choices(self) -> None:
        self._delegate.reset_choices()
        self._summary_by_station.clear()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def _object_for_home_region(self, home_region: str) -> str:
        if str(home_region) != str(getattr(self._env, "demo_original_level", "")):
            raise ValueError(f"WR-04 home-return station level mismatch: {home_region!r}")
        return ITEM_ID


class StationHeightBaseDriver:
    """Apply the sampled station torso height after Base XY/yaw navigation."""

    def __init__(self, delegate: Any, env: Any, *, torso_mover: Any = None):
        self._delegate = delegate
        self._env = env
        self._torso_mover = torso_mover or move_station_torso_to_height

    def plan_and_move(
        self,
        target_xy: np.ndarray,
        target_yaw: float | None,
        *,
        holding: str | None = None,
        planning_summary: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        result = self._delegate.plan_and_move(
            target_xy,
            target_yaw,
            holding=holding,
            planning_summary=planning_summary,
            **kwargs,
        )
        if not bool(result.success):
            return result

        torso_height = _station_torso_height(planning_summary)
        try:
            torso = self._torso_mover(
                self._env,
                torso_height,
                holding=holding,
                post_step_callback=getattr(
                    self._delegate, "_post_step_callback", None
                ),
            )
        except Exception as exc:
            result.success = False
            result.failure_code = f"station torso adjustment failed: {exc}"
            result.details["station_torso"] = {
                "applied": False,
                "requested_height": torso_height,
                "reason": "station_torso_adjustment_failed",
                "error": str(exc),
            }
            return result

        result.details["station_torso"] = torso
        refresh = getattr(self._delegate, "refresh_obs", None)
        if callable(refresh) and bool(torso.get("applied", False)):
            refresh()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def move_station_torso_to_height(
    env: Any,
    target_height: float | None,
    *,
    holding: str | None = None,
    tolerance: float = 0.002,
    max_steps: int = 240,
    post_step_callback: Any = None,
) -> dict[str, Any]:
    """Reach one calibrated station torso height through normal env steps."""

    robot = env.robots[0]
    composite = robot.composite_controller
    torso = composite.part_controllers.get("torso")
    current = (
        None
        if torso is None
        else float(qpos_for_controller(env, torso).reshape(-1)[0])
    )
    if target_height is None:
        return {
            "applied": False,
            "reason": "station_has_no_torso_height",
            "requested_height": None,
            "initial_height": current,
            "actual_height": current,
            "steps": 0,
        }
    target = float(target_height)
    if not np.isfinite(target):
        raise ValueError("station torso_height must be finite")
    if torso is None:
        raise RuntimeError("station requests a torso height but no torso exists")
    if tolerance <= 0.0 or max_steps <= 0:
        raise ValueError("station torso tolerance and max_steps must be positive")

    joint_indexes = np.asarray(torso.joint_index, dtype=int).reshape(-1)
    if joint_indexes.size < 1:
        raise RuntimeError("torso controller does not expose a joint index")
    joint_range = np.asarray(
        env.sim.model.jnt_range[int(joint_indexes[0])], dtype=float
    ).reshape(-1)
    if (
        joint_range.size < 2
        or not np.all(np.isfinite(joint_range[:2]))
        or target < float(joint_range[0]) - 1e-9
        or target > float(joint_range[1]) + 1e-9
    ):
        raise ValueError(
            f"station torso_height={target:.4f}m is outside joint range "
            f"{joint_range[:2].tolist()}"
        )

    right = composite.part_controllers.get("right")
    if right is None:
        raise RuntimeError("robot does not expose the right-arm controller")
    base_start, base_end = composite._action_split_indexes["base"]
    base_action = np.zeros(int(base_end) - int(base_start), dtype=float)
    arm_qpos = qpos_for_controller(env, right)
    initial = float(current)
    actual = initial
    executed_steps = 0
    for _ in range(int(max_steps)):
        error = target - actual
        if abs(error) <= float(tolerance):
            break
        torso_action = float(np.clip(error / 0.05, -1.0, 1.0))
        step_base_and_hold_arm(
            env,
            arm_qpos=arm_qpos,
            base_action=base_action,
            base_mode=1.0,
            gripper_action=1.0 if holding else -1.0,
            torso_action=torso_action,
            post_step_callback=post_step_callback,
        )
        executed_steps += 1
        actual = float(qpos_for_controller(env, torso).reshape(-1)[0])
    if abs(target - actual) > float(tolerance):
        raise RuntimeError(
            f"torso failed to reach {target:.4f}m; actual={actual:.4f}m"
        )
    return {
        "applied": True,
        "reason": "station_torso_height_reached",
        "requested_height": target,
        "initial_height": initial,
        "actual_height": actual,
        "error": actual - target,
        "tolerance": float(tolerance),
        "steps": int(executed_steps),
        "holding": holding,
        "joint_range": joint_range[:2].astype(float).tolist(),
    }


def _station_torso_height(
    planning_summary: dict[str, Any] | None,
) -> float | None:
    station_pose = dict((planning_summary or {}).get("station_pose") or {})
    value = station_pose.get("torso_height")
    return None if value is None else float(value)


def build_navigation_config(
    settings: dict[str, Any],
    *, manual_base_path_file: str | Path | None,
    station_base_pose_file: str | Path,
) -> NavigationConfig:
    values = dict(settings)
    config = replace(
        NavigationConfig(),
        kinematic_base_navigation=bool(values.pop("kinematic_base_navigation", False)),
        base_action_frame=str(values.pop("base_action_frame", "world")),
        base_translation_action_sign=float(values.pop("base_translation_action_sign", 1.0)),
        base_tolerance=float(values.pop("base_tolerance", 0.08)),
        base_yaw_tolerance=float(values.pop("base_yaw_tolerance", 0.08)),
        base_max_steps=int(values.pop("base_max_steps", 2000)),
        base_holding_drop_tolerance=float(values.pop("base_holding_drop_tolerance", 0.10)),
        base_profile_v_max=float(values.pop("base_profile_v_max", 0.35)),
        base_profile_a_max=float(values.pop("base_profile_a_max", 0.70)),
        base_profile_a_dec=float(values.pop("base_profile_a_dec", 0.65)),
        base_profile_v_min=float(values.pop("base_profile_v_min", 0.10)),
        base_profile_dock_v_max=float(values.pop("base_profile_dock_v_max", 0.22)),
        base_profile_dock_v_min=float(values.pop("base_profile_dock_v_min", 0.10)),
        base_profile_w_max=float(values.pop("base_profile_w_max", 0.50)),
        base_profile_alpha_max=float(values.pop("base_profile_alpha_max", 0.80)),
        base_profile_alpha_dec=float(values.pop("base_profile_alpha_dec", 0.70)),
        base_profile_w_min=float(values.pop("base_profile_w_min", 0.08)),
        manual_base_path_enabled=bool(values.pop("manual_base_path_enabled", False)),
        manual_base_path_file=(
            None if manual_base_path_file is None else Path(manual_base_path_file)
        ),
        manual_base_path_start_index=int(values.pop("manual_base_path_start_index", 0)),
        manual_base_path_xy_tolerance=float(values.pop("manual_base_path_xy_tolerance", 0.08)),
        manual_base_path_yaw_tolerance=float(values.pop("manual_base_path_yaw_tolerance", 0.08)),
        manual_base_path_segment_yaw_threshold=float(values.pop("manual_base_path_segment_yaw_threshold", 0.25)),
        manual_base_path_rotate_xy_tolerance=float(values.pop("manual_base_path_rotate_xy_tolerance", 0.18)),
        base_single_axis_tracking=bool(values.pop("base_single_axis_tracking", False)),
        use_station_astar_paths=bool(values.pop("use_station_astar_paths", False)),
        station_nav_use_docking=bool(values.pop("station_nav_use_docking", True)),
        station_base_pose_file=Path(station_base_pose_file),
        use_station_base_poses=bool(values.pop("use_station_base_poses", True)),
    )
    if values:
        raise ValueError("unknown WR-04 navigation settings: " + ", ".join(sorted(values)))
    return config


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    nav_config: NavigationConfig | None = None,
    seed: int = 0,
    base_driver: Any | None = None,
    return_target_xy_by_region: Mapping[str, tuple[float, float]] | None = None,
) -> NavigationAssembly:
    if nav_config is None:
        raise ValueError("WR-04 navigation config is required")
    config = nav_config
    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    raw_driver = (
        base_driver
        if base_driver is not None
        else BaseDriverAdapter(env, obs, nav_config=config)
    )
    driver = StationHeightBaseDriver(raw_driver, env)
    base_pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    sampler = _build_station_sampler(
        adapter,
        config,
        seed=seed,
        base_pose_provider=base_pose_provider,
        return_target_xy_by_region=return_target_xy_by_region,
    )
    planner = (
        StationPathPlanner(obstacles=adapter.build_fixture_obstacles, config=config)
        if bool(config.use_station_astar_paths)
        else None
    )
    controller = NavigationController(
        station_sampler=sampler,
        base_driver=driver,
        path_planner=planner,
        base_pose_provider=base_pose_provider,
    )
    return NavigationAssembly(
        controller=controller,
        env_adapter=adapter,
        station_sampler=sampler,
        path_planner=planner,
        base_driver=driver,
        config=config,
    )


def build_navigation_controller(
    env: Any,
    obs: dict[str, Any],
    *,
    nav_config: NavigationConfig | None = None,
    seed: int = 0,
    base_driver: Any | None = None,
    return_target_xy_by_region: Mapping[str, tuple[float, float]] | None = None,
) -> NavigationController:
    return build_navigation_assembly(
        env,
        obs,
        nav_config=nav_config,
        seed=seed,
        base_driver=base_driver,
        return_target_xy_by_region=return_target_xy_by_region,
    ).controller


def _build_station_sampler(
    adapter: RoboCasaNavigationAdapter,
    config: NavigationConfig,
    *,
    seed: int,
    base_pose_provider: Any,
    return_target_xy_by_region: Mapping[str, tuple[float, float]] | None,
) -> Any:
    aliases = station_aliases()
    if bool(config.use_station_base_poses):
        return ObjectRelativeShelfStationSampler(
            StationPoseSampler.from_file(
                config.station_base_pose_file,
                config=config,
                seed=seed,
                base_xy_provider=base_pose_provider,
                station_aliases=aliases,
            ),
            adapter.env,
            return_target_xy_by_region=return_target_xy_by_region,
        )

    string_aliases = {
        alias: station_id
        for alias, station_id in aliases.items()
        if isinstance(alias, str)
    }
    generated_specs = station_specs_from_candidates(
        adapter.station_candidates(),
        aliases=string_aliases,
    )
    _apply_task_calibrated_station_specs(generated_specs, adapter.env, string_aliases)
    return SimpleStationSampler(
        stations=generated_specs,
        seeds=SeedStream(base_seed=int(seed), namespace="wr_02_navigation"),
    )


def _apply_task_calibrated_station_specs(
    specs: dict[str, StationSpec],
    env: Any,
    aliases: dict[str, str],
) -> None:
    """Add task-calibrated shelf poses when station files are disabled."""
    for region_name in HOME_REGIONS:
        try:
            pose = _pose_xyz(env.home_region_center(region_name))
        except Exception:
            pose = None
        if pose is not None:
            specs[f"{region_name}_station"] = _single_pose_station(
                f"{region_name}_station",
                (pose[0], pose[1] - 0.52),
                yaw=np.pi / 2.0,
                source=f"wr_02_layout32_{region_name}_center",
            )

    for alias, station_id in aliases.items():
        if station_id in specs:
            specs[str(alias)] = specs[station_id]


def _single_pose_station(
    name: str,
    xy: tuple[float, float],
    *,
    yaw: float,
    source: str,
) -> StationSpec:
    return StationSpec(
        name=name,
        base_candidates=[
            Pose(
                position=(float(xy[0]), float(xy[1]), 0.0),
                quaternion=yaw_to_quat_xyzw(float(yaw)),
            )
        ],
        metadata={"source": source},
    )


def _pose_xyz(value: Any) -> tuple[float, float, float] | None:
    try:
        arr = list(value)
    except Exception:
        return None
    if len(arr) < 3:
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _finite_xy(value: Any, *, label: str) -> np.ndarray:
    try:
        xy = np.asarray(value, dtype=float).reshape(-1)[:2]
    except Exception as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if xy.size != 2 or not np.all(np.isfinite(xy)):
        raise ValueError(f"invalid {label}: {value!r}")
    return xy.astype(float)


__all__ = [
    "ObjectRelativeShelfStationSampler",
    "NavigationAssembly",
    "StationHeightBaseDriver",
    "build_navigation_assembly",
    "build_navigation_config",
    "build_navigation_controller",
    "move_station_torso_to_height",
]
