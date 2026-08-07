"""TM04 navigation assembly over TM02's measured layout-15 graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation import BaseDriverAdapter, NavigationController, RoboCasaNavigationAdapter
from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.navigation.station_poses import yaw_to_quat_xyzw
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.tasks.tm02.execution.navigation import (
    build_navigation_config as build_tm02_navigation_config,
    station_specs as tm02_station_specs,
    station_specs_from_manual_path as tm02_specs_from_path,
)


SUGAR_STATION = "sugar_station"
SAUCEPAN_STATION = "saucepan_station"
CAN_PLACE_STATION = "can_place_station"


@dataclass(frozen=True)
class NavigationAssembly:
    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: SimpleStationSampler
    path_planner: None
    base_driver: Any
    config: NavigationConfig


def build_navigation_config(
    settings: dict[str, Any], *, manual_base_path_file: Path
) -> NavigationConfig:
    return build_tm02_navigation_config(
        settings,
        manual_base_path_file=manual_base_path_file,
    )


def station_specs_from_manual_path(path: str | Path) -> dict[str, StationSpec]:
    specs = dict(tm02_specs_from_path(path))
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = {
        int(item.get("index", fallback)): dict(item)
        for fallback, item in enumerate(payload.get("samples", ()))
    }
    targets = dict(payload.get("station_targets", {}))
    for name in (CAN_PLACE_STATION, SUGAR_STATION, SAUCEPAN_STATION):
        raw_target = targets[name]
        sample_index = int(
            raw_target.get("index", raw_target.get("target_index"))
            if isinstance(raw_target, dict)
            else raw_target
        )
        sample = samples[sample_index]
        xy = np.asarray(sample["xy"], dtype=float)
        yaw = float(sample["yaw"])
        specs[name] = StationSpec(
            name=name,
            base_candidates=[
                Pose(
                    position=(float(xy[0]), float(xy[1]), 0.0),
                    quaternion=yaw_to_quat_xyzw(yaw),
                )
            ],
            description="TM04 measured layout-15 task station",
            metadata={
                "source": str(path),
                "sample_index": sample_index,
                "sample_name": sample.get("name"),
                "yaw": yaw,
            },
        )
    return specs


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    specs = dict(tm02_station_specs(env))
    specs[CAN_PLACE_STATION] = {
        "station_id": CAN_PLACE_STATION,
        "fixture": getattr(env, "counter", None),
        "anchor_xy": np.asarray(env.can_place_region_world_xyz, dtype=float)[:2],
        "source_fixture": "island_island_group",
        "source_object_or_phase": "sink_right_can_place_region",
    }
    specs[SUGAR_STATION] = {
        "station_id": SUGAR_STATION,
        "fixture": getattr(env, "sugar_counter", None),
        "anchor_xy": _object_xy(env, getattr(env, "sugar_plate_name", "sugar_plate")),
        "source_fixture": "layout15_stove_counter",
        "source_object_or_phase": "sugar_plate",
    }
    specs[SAUCEPAN_STATION] = {
        "station_id": SAUCEPAN_STATION,
        "fixture": getattr(env, "stove", None),
        "anchor_xy": _object_xy(env, getattr(env, "hot_chocolate_name", "hot_chocolate_saucepan")),
        "source_fixture": "layout15_stove",
        "source_object_or_phase": "hot_chocolate_saucepan",
    }
    return specs


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    config: NavigationConfig,
    seed: int,
    base_driver: Any | None = None,
) -> NavigationAssembly:
    if not config.manual_base_path_enabled or config.manual_base_path_file is None:
        raise ValueError("TM04 requires the layout-15 manual base path")
    sampler = SimpleStationSampler(
        stations=station_specs_from_manual_path(config.manual_base_path_file),
        seeds=SeedStream(base_seed=int(seed), namespace="tm04_navigation"),
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


def _object_xy(env: Any, name: str) -> np.ndarray | None:
    try:
        value = np.asarray(env.object_world_position(name), dtype=float).reshape(-1)
    except Exception:
        return None
    return value[:2] if value.size >= 2 and np.all(np.isfinite(value[:2])) else None


__all__ = [
    "NavigationAssembly", "SAUCEPAN_STATION", "SUGAR_STATION",
    "build_navigation_assembly", "build_navigation_config", "station_specs",
    "station_specs_from_manual_path",
]
