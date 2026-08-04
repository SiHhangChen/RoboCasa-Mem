"""TM02 station mapping over the user-calibrated nine-point path."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.navigation import BaseDriverAdapter, NavigationController, RoboCasaNavigationAdapter
from bench.execution.navigation.station_poses import yaw_to_quat_xyzw
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler


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


@dataclass(frozen=True)
class NavigationAssembly:
    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: SimpleStationSampler
    path_planner: None
    base_driver: Any
    config: NavigationConfig


def build_navigation_config(
    settings: dict[str, Any],
    *,
    manual_base_path_file: Path,
) -> NavigationConfig:
    """Make the nine-point measured path authoritative for TM02."""

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


def station_specs_from_manual_path(path: str | Path) -> dict[str, StationSpec]:
    """Resolve the four workflow stations from named path targets."""

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
    """Bind workflow station names to the measured spatial-graph executor."""

    if not config.manual_base_path_enabled or config.manual_base_path_file is None:
        raise ValueError("TM02 requires its measured manual base path")
    sampler = SimpleStationSampler(
        stations=station_specs_from_manual_path(config.manual_base_path_file),
        seeds=SeedStream(base_seed=int(seed), namespace="tm02_navigation"),
        mode="first_valid",
    )
    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    driver = base_driver or BaseDriverAdapter(env, obs, nav_config=config)
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
    "NavigationAssembly",
    "STATION_IDS",
    "WATER_BUTTON_STATION",
    "build_navigation_assembly",
    "build_navigation_config",
    "station_specs",
    "station_specs_from_manual_path",
]
