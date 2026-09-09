"""WA-02 station mapping and mobile-base controller assembly."""

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
from bench.execution.navigation.base_motion import qpos_for_controller, step_base_and_hold_arm
from bench.execution.navigation.env_adapter import fixture_xy, object_xy, safe_name
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.tasks.wa02.domain.contracts import CUP_IDS, SERVICE_SLOTS


CUP_PICK_STATION = "cup_pick_station"
CUP_A_PICK_STATION = "cup_A_pick_station"
CUP_B_PICK_STATION = "cup_B_pick_station"
SINK_FILL_STATION = "sink_fill_station"
SERVICE_AREA_STATION = "service_area_station"
SERVICE_AREA_SLOT_2_STATION = "service_area_slot_2_station"
SERVICE_AREA_CENTER_STATION = "service_area_center_station"


@dataclass(frozen=True)
class NavigationAssembly:
    """The task-local pieces around the shared navigation controller."""

    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


class StationHeightBaseDriver:
    """Apply each calibrated station's torso height after Base navigation."""

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

        target_height = _station_torso_height(planning_summary)
        try:
            torso = self._torso_mover(
                self._env,
                target_height,
                holding=holding,
                post_step_callback=getattr(
                    self._delegate,
                    "_post_step_callback",
                    None,
                ),
            )
        except Exception as exc:
            result.success = False
            result.failure_code = f"station torso adjustment failed: {exc}"
            result.details["station_torso"] = {
                "applied": False,
                "requested_height": target_height,
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
    """Move the lift column normally while holding arm and Base position."""

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


def station_aliases() -> dict[Any, str]:
    """Map workflow anchors to their task-owned station IDs."""

    return {
        CUP_PICK_STATION: CUP_PICK_STATION,
        CUP_A_PICK_STATION: CUP_A_PICK_STATION,
        CUP_B_PICK_STATION: CUP_B_PICK_STATION,
        SINK_FILL_STATION: SINK_FILL_STATION,
        SERVICE_AREA_STATION: SERVICE_AREA_STATION,
        SERVICE_AREA_SLOT_2_STATION: SERVICE_AREA_SLOT_2_STATION,
        SERVICE_AREA_CENTER_STATION: SERVICE_AREA_CENTER_STATION,
        ("service_area", "slot_2"): SERVICE_AREA_SLOT_2_STATION,
        "service_area:slot_2": SERVICE_AREA_SLOT_2_STATION,
        "cup_A": CUP_A_PICK_STATION,
        "cup_B": CUP_B_PICK_STATION,
        "sink_basin": SINK_FILL_STATION,
        "sink": SINK_FILL_STATION,
        "service_area": SERVICE_AREA_STATION,
        "service_counter": SERVICE_AREA_STATION,
    }


def build_navigation_config(
    settings: dict[str, Any],
    *,
    manual_base_path_file: Path,
    station_base_pose_file: Path,
) -> NavigationConfig:
    """Convert the bounded YAML navigation surface into ``NavigationConfig``."""

    values = dict(settings)
    config = replace(
        NavigationConfig(),
        kinematic_base_navigation=bool(values.pop("kinematic_base_navigation", False)),
        base_action_frame=str(values.pop("base_action_frame", "world")),
        base_translation_action_sign=float(values.pop("base_translation_action_sign", 1.0)),
        base_tolerance=float(values.pop("base_tolerance", 0.08)),
        base_yaw_tolerance=float(values.pop("base_yaw_tolerance", 0.08)),
        base_max_steps=int(values.pop("base_max_steps", 1500)),
        manual_base_path_enabled=bool(values.pop("manual_base_path_enabled", True)),
        manual_base_path_file=Path(manual_base_path_file),
        manual_base_path_start_index=int(values.pop("manual_base_path_start_index", 1)),
        manual_base_path_xy_tolerance=float(
            values.pop("manual_base_path_xy_tolerance", 0.12)
        ),
        manual_base_path_yaw_tolerance=float(
            values.pop("manual_base_path_yaw_tolerance", 0.08)
        ),
        manual_base_path_segment_yaw_threshold=float(
            values.pop("manual_base_path_segment_yaw_threshold", 0.05)
        ),
        base_velocity_profile_enabled=bool(
            values.pop("base_velocity_profile_enabled", True)
        ),
        base_profile_v_max=float(values.pop("base_profile_v_max", 0.65)),
        base_profile_a_max=float(values.pop("base_profile_a_max", 1.6)),
        base_profile_a_dec=float(values.pop("base_profile_a_dec", 1.2)),
        base_profile_v_min=float(values.pop("base_profile_v_min", 0.22)),
        base_profile_w_max=float(values.pop("base_profile_w_max", 1.2)),
        base_profile_alpha_max=float(values.pop("base_profile_alpha_max", 3.0)),
        base_profile_alpha_dec=float(values.pop("base_profile_alpha_dec", 2.5)),
        base_profile_w_min=float(values.pop("base_profile_w_min", 0.20)),
        base_single_axis_tracking=bool(values.pop("base_single_axis_tracking", False)),
        base_snap_to_target_after_success=bool(
            values.pop("base_snap_to_target_after_success", False)
        ),
        use_station_astar_paths=bool(values.pop("use_station_astar_paths", False)),
        station_nav_use_docking=bool(values.pop("station_nav_use_docking", False)),
        station_nav_dock_standoff=float(values.pop("station_nav_dock_standoff", 0.0)),
        station_nav_require_footprint=bool(
            values.pop("station_nav_require_footprint", False)
        ),
        station_base_pose_file=Path(station_base_pose_file),
        use_station_base_poses=bool(values.pop("use_station_base_poses", False)),
    )
    if values:
        raise ValueError("unknown WA-02 navigation settings: " + ", ".join(sorted(values)))
    return config


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Read live scene anchors for non-manual navigation fallback only."""

    counter = getattr(env, "counter", None)
    sink = getattr(env, "sink", None)
    cup_points = {cup_id: object_xy(env, cup_id) for cup_id in CUP_IDS}
    cup_xy = _first_available_xy(
        _mean_xy(list(cup_points.values())),
        fixture_xy(counter),
    )
    water_xy = _first_available_xy(
        _xy_or_none(_water_site_position(env)),
        fixture_xy(sink),
    )
    service_points = [
        _xy_or_none(_service_slot_position(env, slot)) for slot in SERVICE_SLOTS
    ]
    service_xy = _first_available_xy(_mean_xy(service_points), fixture_xy(counter))
    return {
        CUP_PICK_STATION: {
            "station_id": CUP_PICK_STATION,
            "fixture": counter,
            "anchor_xy": cup_xy,
            "source_fixture": safe_name(counter),
            "source_object_or_phase": "cup_A,cup_B",
            "stand_off": 0.16,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        CUP_A_PICK_STATION: {
            "station_id": CUP_A_PICK_STATION,
            "fixture": counter,
            "anchor_xy": _first_available_xy(cup_points.get("cup_A"), cup_xy),
            "source_fixture": safe_name(counter),
            "source_object_or_phase": "cup_A",
            "stand_off": 0.16,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        CUP_B_PICK_STATION: {
            "station_id": CUP_B_PICK_STATION,
            "fixture": counter,
            "anchor_xy": _first_available_xy(cup_points.get("cup_B"), cup_xy),
            "source_fixture": safe_name(counter),
            "source_object_or_phase": "cup_B",
            "stand_off": 0.16,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        SINK_FILL_STATION: {
            "station_id": SINK_FILL_STATION,
            "fixture": sink or counter,
            "anchor_xy": water_xy,
            "source_fixture": safe_name(sink or counter),
            "source_object_or_phase": "sink_water_site",
            "stand_off": 0.24,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        SERVICE_AREA_STATION: {
            "station_id": SERVICE_AREA_STATION,
            "fixture": counter,
            "anchor_xy": service_xy,
            "source_fixture": safe_name(counter),
            "source_object_or_phase": "neutral_service_slots",
            "stand_off": 0.16,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        SERVICE_AREA_CENTER_STATION: {
            "station_id": SERVICE_AREA_CENTER_STATION,
            "fixture": counter,
            # Fixed calibrated midpoint between slot_1 and slot_2.
            "anchor_xy": service_xy,
            "source_fixture": safe_name(counter),
            "source_object_or_phase": "service_slots_midpoint",
            "stand_off": 0.16,
            "tangential_offsets": (0.0,),
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
    """Assemble calibrated or generated station navigation without previous WA2 code."""

    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    raw_driver = base_driver or BaseDriverAdapter(env, obs, nav_config=config)
    driver = StationHeightBaseDriver(raw_driver, env)
    pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    aliases = station_aliases()
    if config.use_station_base_poses:
        if config.station_base_pose_file is None:
            raise ValueError("WA-02 station poses require station_base_pose_file")
        sampler: Any = StationPoseSampler.from_file(
            config.station_base_pose_file,
            config=config,
            seed=int(seed),
            base_xy_provider=pose_provider,
            station_aliases=aliases,
        )
    else:
        sampler = SimpleStationSampler(
            stations=station_specs_from_candidates(
                adapter.station_candidates(),
                aliases={key: value for key, value in aliases.items() if isinstance(key, str)},
            ),
            seeds=SeedStream(base_seed=int(seed), namespace="wa_02_navigation"),
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


def _water_site_position(env: Any) -> np.ndarray | None:
    try:
        return np.asarray(env.water_site_position(), dtype=float)
    except Exception:
        return None


def _service_slot_position(env: Any, slot: str) -> np.ndarray | None:
    try:
        return np.asarray(env.service_slot_position(slot), dtype=float)
    except Exception:
        return None


def _xy_or_none(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array[:2])):
        return None
    return array[:2]


def _mean_xy(values: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return np.mean(np.vstack(valid), axis=0)


def _first_available_xy(*values: np.ndarray | None) -> np.ndarray | None:
    """Return the first coordinate without evaluating arrays as booleans."""

    return next((value for value in values if value is not None), None)


__all__ = [
    "CUP_A_PICK_STATION",
    "CUP_B_PICK_STATION",
    "CUP_PICK_STATION",
    "NavigationAssembly",
    "SERVICE_AREA_STATION",
    "SERVICE_AREA_SLOT_2_STATION",
    "SERVICE_AREA_CENTER_STATION",
    "SINK_FILL_STATION",
    "StationHeightBaseDriver",
    "build_navigation_assembly",
    "build_navigation_config",
    "move_station_torso_to_height",
    "station_aliases",
    "station_specs",
]
