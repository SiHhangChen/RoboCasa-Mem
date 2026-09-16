"""TM02 station mapping over the user-calibrated path with circle + relative turn sampling."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
    RoboCasaNavigationAdapter,
)
from bench.execution.navigation.manual_base_path import ManualBasePathSample
from bench.execution.navigation.station_poses import (
    quat_xyzw_to_yaw,
    yaw_to_quat_xyzw,
)
from bench.execution.planning import NavigationConfig
from bench.execution.planning.nav_algo import normalize_angle
from bench.execution.sampling.circle_sampling import (
    sample_uniform_disk,
    sample_uniform_yaw,
)
from bench.execution.sampling.rng import SeedStream


WATER_BUTTON_STATION = "water_button_station"
CABINET_DOOR_STATION = "cabinet_door_station"
CAN_RETRIEVAL_STATION = "can_retrieval_station"
CAN_PLACE_STATION = "can_place_station"
COMPLETION_BUTTON_STATION = "completion_button_station"
STATION_IDS = (
    WATER_BUTTON_STATION,
    CABINET_DOOR_STATION,
    CAN_RETRIEVAL_STATION,
    CAN_PLACE_STATION,
    COMPLETION_BUTTON_STATION,
)

# Indices that use relative_turn (angle sampling around nominal delta).
# These are the primary steering/turning points in the calibrated loop.  The
# five perturbations are sampled jointly so their sum remains zero, preserving
# the measured loop closure just like TM01.
TM02_TURN_INDICES = (1, 3, 6, 8, 10)


@dataclass
class EpisodePathPoseSampler:
    """Sample every TM02 path vertex once per episode (circle XY + relative yaw)."""

    path_file: Path
    seed: int
    station_targets: dict[str, Any]
    route_poses: dict[int, Pose]
    route_summaries: dict[int, dict[str, Any]]
    _summary_by_station: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path, *, seed: int) -> "EpisodePathPoseSampler":
        path_value = Path(path)
        payload = json.loads(path_value.read_text(encoding="utf-8"))
        route_poses, route_summaries = sample_episode_path_poses(
            payload,
            seed=int(seed),
        )
        return cls(
            path_file=path_value,
            seed=int(seed),
            station_targets=dict(payload.get("station_targets", {})),
            route_poses=route_poses,
            route_summaries=route_summaries,
        )

    def sample(self, station_name: str, *, purpose: str | None = None) -> Pose:
        del purpose
        if station_name not in self.station_targets:
            raise KeyError(f"TM02 path has no station target {station_name!r}")
        index = _target_index(self.station_targets[station_name])
        try:
            pose = self.route_poses[index]
        except KeyError as exc:
            raise ValueError(
                f"TM02 station {station_name!r} targets unsampled path index {index}"
            ) from exc
        self._summary_by_station[station_name] = {
            "station_name": station_name,
            "station_id": station_name,
            "station_pose_selection": {
                "policy": "episode_circle_uniform",
                "episode_stable": True,
                "seed": int(self.seed),
            },
            **dict(self.route_summaries[index]),
        }
        return pose

    def last_summary(self, station_name: str) -> dict[str, Any] | None:
        return self._summary_by_station.get(station_name)

    def reset_choices(self) -> None:
        self._summary_by_station.clear()


class SampledPathBaseDriver(BaseDriverAdapter):
    """Install the episode-owned TM02 waypoint poses into the manual path executor."""

    def __init__(
        self,
        env: Any,
        obs: dict[str, Any],
        *,
        config: NavigationConfig,
        sampler: EpisodePathPoseSampler,
    ) -> None:
        super().__init__(env, obs, nav_config=config)
        self._station_sampler = sampler
        self._sampled_executor: Any = None
        self._waypoint_summaries: dict[int, dict[str, Any]] = {}

    def _manual_path_executor(self) -> Any:
        executor = super()._manual_path_executor()
        if executor is None or executor is self._sampled_executor:
            return executor
        for index, pose in self._station_sampler.route_poses.items():
            if not 0 <= index < len(executor.samples):
                raise ValueError(f"TM02 sampled waypoint {index} is outside the path")
            nominal = executor.samples[index]
            executor.samples[index] = ManualBasePathSample(
                index=index,
                xy=np.asarray(pose.position[:2], dtype=float),
                yaw=quat_xyzw_to_yaw(pose.quaternion),
                name=nominal.name,
            )
            self._waypoint_summaries[index] = {
                "path_index": index,
                "nominal_xy": nominal.xy.tolist(),
                "nominal_yaw": nominal.yaw,
                **dict(self._station_sampler.route_summaries[index]),
            }
        self._sampled_executor = executor
        return executor

    def plan_and_move(self, *args: Any, **kwargs: Any) -> Any:
        result = super().plan_and_move(*args, **kwargs)
        result.details["sampled_path_waypoints"] = [
            dict(self._waypoint_summaries[index])
            for index in result.details.get("route_indices", ())
            if index in self._waypoint_summaries
        ]
        return result


@dataclass(frozen=True)
class NavigationAssembly:
    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: EpisodePathPoseSampler
    path_planner: None
    base_driver: Any
    config: NavigationConfig


def build_navigation_config(
    settings: dict[str, Any],
    *,
    manual_base_path_file: Path,
) -> NavigationConfig:
    """Make the measured manual path authoritative for TM02."""

    values = dict(settings)
    allowed = {field.name for field in fields(NavigationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown TM02 navigation settings: " + ", ".join(unknown))
    values.update(
        {
            "manual_base_path_enabled": True,
            "manual_base_path_file": Path(manual_base_path_file),
            "station_base_pose_file": None,
            "use_station_base_poses": False,
            "use_station_astar_paths": False,
        }
    )
    values.setdefault("manual_base_path_start_index", 0)
    return replace(NavigationConfig(), **values)


def sample_episode_path_poses(
    payload: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[int, Pose], dict[int, dict[str, Any]]]:
    """Resolve one reproducible set of sampled TM02 path poses (XY circle + yaw)."""

    raw_samples = list(payload.get("samples", ()))
    samples: dict[int, dict[str, Any]] = {}
    for fallback, raw in enumerate(raw_samples):
        item = dict(raw)
        index = int(item.get("index", fallback))
        if index in samples:
            raise ValueError(f"duplicate TM02 sampled path index {index}")
        samples[index] = item
    if sorted(samples) != list(range(len(samples))):
        raise ValueError("TM02 sampled path indexes must be contiguous from zero")

    seeds = SeedStream(base_seed=int(seed), namespace="tm02_navigation")
    sampled_xy: dict[int, np.ndarray] = {}
    for index, item in samples.items():
        circle = item.get("circle")
        if not isinstance(circle, Mapping):
            raise ValueError(f"TM02 path sample {index} requires a circle")
        sampled_xy[index] = sample_uniform_disk(
            circle.get("center_xy", ()),
            float(circle.get("radius_m", -1.0)),
            seed=seeds.seed("waypoint", index, "xy"),
            salt=0x544D3032,
        )

    turn_offsets = _sample_closed_turn_offsets(
        samples,
        seed=seeds.seed("main_loop_turns"),
    )
    resolved_yaw: dict[int, float] = {}
    resolving: set[int] = set()

    def yaw_for(index: int) -> float:
        if index in resolved_yaw:
            return resolved_yaw[index]
        if index in resolving:
            raise ValueError(f"TM02 path yaw dependency cycle at sample {index}")
        if index not in samples:
            raise ValueError(f"TM02 path yaw source {index} is unavailable")
        resolving.add(index)
        item = samples[index]
        turn = item.get("relative_turn")
        inherited_from = item.get("inherit_yaw_from")
        if isinstance(turn, Mapping):
            source_index = int(turn["source_index"])
            nominal = float(turn["nominal_angle_rad"])
            half_range = float(turn["angle_half_range_rad"])
            if index in turn_offsets:
                offset = turn_offsets[index]
            else:
                sampled = sample_uniform_yaw(
                    nominal,
                    half_range,
                    seed=seeds.seed("turn", index),
                    salt=0x5455524E,
                )
                offset = normalize_angle(sampled - nominal)
            yaw = normalize_angle(yaw_for(source_index) + nominal + offset)
        elif inherited_from is not None:
            yaw = yaw_for(int(inherited_from))
        else:
            circle = item["circle"]
            yaw = normalize_angle(float(circle.get("base_yaw", item.get("yaw"))))
        resolving.remove(index)
        resolved_yaw[index] = yaw
        return yaw

    poses: dict[int, Pose] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for index, item in samples.items():
        yaw = yaw_for(index)
        circle = dict(item["circle"])
        summary: dict[str, Any] = {
            "path_index": int(index),
            "sample_name": item.get("name"),
            "route_role": "station_circle",
            "target_xy": sampled_xy[index].astype(float).tolist(),
            "target_yaw": float(yaw),
            "circle": {
                "center_xy": [float(value) for value in circle["center_xy"]],
                "radius_m": float(circle["radius_m"]),
            },
        }
        turn = item.get("relative_turn")
        if isinstance(turn, Mapping):
            nominal = float(turn["nominal_angle_rad"])
            # Recompute offset from resolved yaws for auditability
            source_idx = int(turn["source_index"])
            offset = normalize_angle(yaw - yaw_for(source_idx) - nominal)
            summary.update(
                {
                    "route_role": "relative_turn_circle",
                    "source_index": source_idx,
                    "nominal_turn_angle": nominal,
                    "turn_angle_half_range": float(turn["angle_half_range_rad"]),
                    "sampled_turn_offset": float(offset),
                    "sampled_turn_angle": nominal + float(offset),
                    "closed_loop_constraint": index in TM02_TURN_INDICES,
                }
            )
        elif item.get("inherit_yaw_from") is not None:
            summary.update(
                {
                    "route_role": "inherited_yaw_circle",
                    "yaw_inherited_from_path_index": int(item["inherit_yaw_from"]),
                }
            )
        poses[index] = Pose(
            position=(float(sampled_xy[index][0]), float(sampled_xy[index][1]), 0.0),
            quaternion=yaw_to_quat_xyzw(yaw),
        )
        summaries[index] = summary
    return poses, summaries


def _sample_closed_turn_offsets(
    samples: Mapping[int, Mapping[str, Any]],
    *,
    seed: int,
) -> dict[int, float]:
    """Sample bounded turn noise whose sum is zero around TM02's loop."""

    half_ranges: list[float] = []
    for index in TM02_TURN_INDICES:
        turn = samples.get(index, {}).get("relative_turn")
        if not isinstance(turn, Mapping):
            raise ValueError(f"TM02 loop turn {index} lacks relative_turn")
        value = float(turn["angle_half_range_rad"])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("TM02 loop turn ranges must be positive")
        half_ranges.append(value)
    if not np.allclose(half_ranges, half_ranges[0]):
        raise ValueError("TM02 loop turns must share one half-range")

    half_range = half_ranges[0]
    rng = np.random.default_rng(int(seed))
    for _ in range(10000):
        values = rng.uniform(-half_range, half_range, size=len(TM02_TURN_INDICES))
        values -= float(np.mean(values))
        if float(np.max(np.abs(values))) <= half_range:
            return dict(zip(TM02_TURN_INDICES, values.astype(float).tolist()))
    raise RuntimeError("TM02 could not sample bounded closed-loop turn offsets")


def station_specs_from_manual_path(path: str | Path) -> dict[str, StationSpec]:
    """Resolve workflow stations from named path targets (used for diagnostics)."""

    path_value = Path(path)
    payload = json.loads(path_value.read_text(encoding="utf-8"))
    if str(payload.get("route_mode", "")).strip().lower() != "spatial_graph":
        raise ValueError("TM02 manual base path must use route_mode='spatial_graph'")

    samples: dict[int, dict[str, Any]] = {}
    for fallback_index, raw in enumerate(payload.get("samples", ())):
        item = dict(raw)
        index = int(item.get("index", fallback_index))
        if index in samples:
            raise ValueError(f"duplicate TM02 manual base sample index {index}")
        samples[index] = item
    if sorted(samples) != list(range(len(samples))):
        raise ValueError("TM02 manual base sample indexes must be contiguous from zero")

    targets = dict(payload.get("station_targets", {}))
    result: dict[str, StationSpec] = {}
    for station_id in STATION_IDS:
        if station_id not in targets:
            raise KeyError(f"TM02 manual base path has no target {station_id!r}")
        sample_index = _target_index(targets[station_id])
        if sample_index not in samples:
            raise ValueError(
                f"TM02 station {station_id!r} targets missing sample {sample_index}"
            )
        sample = samples[sample_index]
        xy = np.asarray(sample.get("xy", ()), dtype=float).reshape(-1)
        yaw = float(sample.get("yaw"))
        if xy.size != 2 or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"TM02 station {station_id!r} has an invalid pose")
        result[station_id] = StationSpec(
            name=station_id,
            base_candidates=[
                Pose(
                    position=(float(xy[0]), float(xy[1]), 0.0),
                    quaternion=yaw_to_quat_xyzw(yaw),
                )
            ],
            description="TM02 user-measured manual-path terminal pose",
            metadata={
                "source": str(path_value),
                "sample_index": sample_index,
                "sample_name": sample.get("name"),
                "yaw": yaw,
            },
        )
    return result


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Expose live task anchors for probe diagnostics."""

    cabinet = getattr(env, "storage_cabinet", None)
    cabinet_xy = _xy(getattr(cabinet, "pos", None))
    try:
        water_xy = _xy(env.water_button_position("cold"))
    except Exception:
        water_xy = None
    place_xy = _xy(getattr(env, "can_place_region_world_xyz", None))
    return {
        WATER_BUTTON_STATION: {
            "station_id": WATER_BUTTON_STATION,
            "fixture": getattr(env, "sink", None),
            "anchor_xy": water_xy,
            "source_fixture": "sink",
            "source_object_or_phase": "cold_water_button",
        },
        CABINET_DOOR_STATION: {
            "station_id": CABINET_DOOR_STATION,
            "fixture": cabinet,
            "anchor_xy": cabinet_xy,
            "source_fixture": "tm02_can_cabinet",
            "source_object_or_phase": "sliding_door",
        },
        CAN_RETRIEVAL_STATION: {
            "station_id": CAN_RETRIEVAL_STATION,
            "fixture": cabinet,
            "anchor_xy": cabinet_xy,
            "source_fixture": "tm02_can_cabinet",
            "source_object_or_phase": "can_retrieval",
        },
        CAN_PLACE_STATION: {
            "station_id": CAN_PLACE_STATION,
            "fixture": getattr(env, "cabinet_counter", None),
            "anchor_xy": place_xy,
            "source_fixture": "counter_1_left_group_1",
            "source_object_or_phase": "can_place_region",
        },
        COMPLETION_BUTTON_STATION: {
            "station_id": COMPLETION_BUTTON_STATION,
            "fixture": getattr(env, "sink", None),
            "anchor_xy": _xy(
                getattr(env, "completion_button_world_xyz", None)
            ),
            "source_fixture": "sink",
            "source_object_or_phase": "water_button_left_completion_button",
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
    """Bind workflow station names to the measured spatial-graph executor with episode sampling."""

    if not config.manual_base_path_enabled or config.manual_base_path_file is None:
        raise ValueError("TM02 requires its measured manual base path")
    sampler = EpisodePathPoseSampler.from_file(
        config.manual_base_path_file,
        seed=int(seed),
    )
    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    driver = base_driver or SampledPathBaseDriver(
        env,
        obs,
        config=config,
        sampler=sampler,
    )
    pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    controller = NavigationController(
        station_sampler=sampler,
        base_driver=driver,
        path_planner=None,
        base_pose_provider=pose_provider,
    )
    return NavigationAssembly(controller, adapter, sampler, None, driver, config)


def _target_index(value: Any) -> int:
    if isinstance(value, dict):
        raw = value.get("index", value.get("target_index"))
    elif isinstance(value, list):
        if not value:
            raise ValueError("TM02 station target index list cannot be empty")
        raw = value[0]
    else:
        raw = value
    return int(raw)


def _xy(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float).reshape(-1)[:2]
    except Exception:
        return None
    if result.size != 2 or not np.all(np.isfinite(result)):
        return None
    return result


__all__ = [
    "CABINET_DOOR_STATION",
    "CAN_PLACE_STATION",
    "CAN_RETRIEVAL_STATION",
    "COMPLETION_BUTTON_STATION",
    "EpisodePathPoseSampler",
    "NavigationAssembly",
    "SampledPathBaseDriver",
    "STATION_IDS",
    "TM02_TURN_INDICES",
    "WATER_BUTTON_STATION",
    "build_navigation_assembly",
    "build_navigation_config",
    "sample_episode_path_poses",
    "station_specs",
    "station_specs_from_manual_path",
]
