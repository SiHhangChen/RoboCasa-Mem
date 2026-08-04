"""TM03 six-station navigation over its task-owned spatial graph."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
    RoboCasaNavigationAdapter,
)
from bench.execution.navigation.station_poses import yaw_to_quat_xyzw
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.tasks.tm03.domain.contracts import (
    DRAWER_SIDE_COMPLETION_BUTTON,
    STOVE_SIDE_COMPLETION_BUTTON,
)


STOVE_STATION = "stove_station"
STOVE_SIDE_COMPLETION_STATION = "stove_side_completion_station"
DRAWER_SIDE_COMPLETION_STATION = "drawer_side_completion_station"
DRAWER_LEFT_STATION = "drawer_left_station"
DRAWER_RIGHT_STATION = "drawer_right_station"
TOASTER_OVEN_STATION = "toaster_oven_station"
STATION_IDS = (
    STOVE_STATION,
    STOVE_SIDE_COMPLETION_STATION,
    DRAWER_SIDE_COMPLETION_STATION,
    DRAWER_LEFT_STATION,
    DRAWER_RIGHT_STATION,
    TOASTER_OVEN_STATION,
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
    settings: dict[str, Any], *, manual_base_path_file: Path
) -> NavigationConfig:
    values = dict(settings)
    allowed = {field.name for field in fields(NavigationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown TM03 navigation settings: " + ", ".join(unknown))
    values.update(
        {
            "manual_base_path_enabled": True,
            "manual_base_path_file": Path(manual_base_path_file),
            "station_base_pose_file": None,
            "use_station_base_poses": False,
            "use_station_astar_paths": False,
        }
    )
    return replace(NavigationConfig(), **values)


def station_specs_from_manual_path(path: str | Path) -> dict[str, StationSpec]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if str(payload.get("route_mode", "")).lower() != "spatial_graph":
        raise ValueError("TM03 manual path must use route_mode='spatial_graph'")
    samples: dict[int, dict[str, Any]] = {}
    for fallback, raw in enumerate(payload.get("samples", ())):
        item = dict(raw)
        index = int(item.get("index", fallback))
        if index in samples:
            raise ValueError(f"duplicate TM03 manual path sample {index}")
        samples[index] = item
    if sorted(samples) != list(range(len(samples))):
        raise ValueError("TM03 manual path samples must be contiguous from zero")
    targets = dict(payload.get("station_targets") or {})
    result: dict[str, StationSpec] = {}
    for station_id in STATION_IDS:
        if station_id not in targets:
            raise KeyError(f"TM03 manual path has no target {station_id!r}")
        index = _target_index(targets[station_id])
        sample = samples[index]
        xy = np.asarray(sample.get("xy", ()), dtype=float).reshape(-1)
        yaw = float(sample.get("yaw"))
        if xy.size != 2 or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"TM03 station {station_id!r} has an invalid pose")
        result[station_id] = StationSpec(
            name=station_id,
            base_candidates=[
                Pose(
                    position=(float(xy[0]), float(xy[1]), 0.0),
                    quaternion=yaw_to_quat_xyzw(yaw),
                )
            ],
            description="TM03 measured manual-path terminal pose",
            metadata={
                "source": str(source),
                "sample_index": index,
                "sample_name": sample.get("name"),
                "yaw": yaw,
            },
        )
    return result


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    stove = getattr(env, "stove", None)
    toaster_oven = getattr(env, "toaster_oven", None)
    left = env.cabinet_for_drawer("drawer_left")
    right = env.cabinet_for_drawer("drawer_right")
    return {
        STOVE_STATION: _fixture_station(STOVE_STATION, stove, "front_right"),
        STOVE_SIDE_COMPLETION_STATION: {
            **_fixture_station(
                STOVE_SIDE_COMPLETION_STATION,
                stove,
                STOVE_SIDE_COMPLETION_BUTTON,
            ),
            "anchor_xy": _button_xy(env, STOVE_SIDE_COMPLETION_BUTTON),
        },
        DRAWER_SIDE_COMPLETION_STATION: {
            **_fixture_station(
                DRAWER_SIDE_COMPLETION_STATION,
                left,
                DRAWER_SIDE_COMPLETION_BUTTON,
            ),
            "anchor_xy": _button_xy(env, DRAWER_SIDE_COMPLETION_BUTTON),
        },
        DRAWER_LEFT_STATION: _fixture_station(
            DRAWER_LEFT_STATION, left, "drawer_left"
        ),
        DRAWER_RIGHT_STATION: _fixture_station(
            DRAWER_RIGHT_STATION, right, "drawer_right"
        ),
        TOASTER_OVEN_STATION: _fixture_station(
            TOASTER_OVEN_STATION, toaster_oven, "time_knob"
        ),
    }


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    config: NavigationConfig,
    seed: int,
    base_driver: Any | None = None,
) -> NavigationAssembly:
    if not config.manual_base_path_enabled or config.manual_base_path_file is None:
        raise ValueError("TM03 requires its task-owned manual spatial graph")
    sampler = SimpleStationSampler(
        stations=station_specs_from_manual_path(config.manual_base_path_file),
        seeds=SeedStream(base_seed=int(seed), namespace="tm03_navigation"),
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


def _fixture_station(
    station_id: str, fixture: Any, phase: str
) -> dict[str, Any]:
    return {
        "station_id": station_id,
        "fixture": fixture,
        "anchor_xy": _xy(getattr(fixture, "pos", None)),
        "source_fixture": getattr(fixture, "name", None),
        "source_object_or_phase": phase,
    }


def _target_index(value: Any) -> int:
    if isinstance(value, dict):
        value = value.get("index", value.get("target_index"))
    elif isinstance(value, list):
        value = value[0]
    return int(value)


def _xy(value: Any) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=float).reshape(-1)[:2]
    except Exception:
        return None
    return result if result.size == 2 and np.all(np.isfinite(result)) else None


def _button_xy(env: Any, name: str) -> np.ndarray | None:
    try:
        return _xy(env.completion_button_position(name))
    except Exception:
        return None


__all__ = [
    "DRAWER_LEFT_STATION",
    "DRAWER_RIGHT_STATION",
    "DRAWER_SIDE_COMPLETION_STATION",
    "TOASTER_OVEN_STATION",
    "NavigationAssembly",
    "STATION_IDS",
    "STOVE_SIDE_COMPLETION_STATION",
    "STOVE_STATION",
    "build_navigation_assembly",
    "build_navigation_config",
    "station_specs",
    "station_specs_from_manual_path",
]
