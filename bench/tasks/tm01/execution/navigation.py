"""TM01 station poses and manual-path mobile-base assembly."""

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
from bench.tasks.tm01.domain.contracts import (
    DRAWER_NAMES,
    DRAWER_SIDE_COMPLETION_BUTTON,
    STOVE_SIDE_COMPLETION_BUTTON,
)


STOVE_STATION = "stove_station"
STOVE_SIDE_COMPLETION_STATION = "stove_side_completion_station"
DRAWER_SIDE_COMPLETION_STATION = "drawer_side_completion_station"
DRAWER_STATION = "drawer_station"
STATION_IDS = (
    STOVE_STATION,
    STOVE_SIDE_COMPLETION_STATION,
    DRAWER_SIDE_COMPLETION_STATION,
    DRAWER_STATION,
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
    """Make the measured manual path authoritative for TM01 navigation."""

    values = dict(settings)
    allowed = {field.name for field in fields(NavigationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown TM01 navigation settings: " + ", ".join(unknown))
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
    """Load the five operation poses from the path's named station targets."""

    path_value = Path(path)
    payload = json.loads(path_value.read_text(encoding="utf-8"))
    if str(payload.get("route_mode", "")).strip().lower() != "spatial_graph":
        raise ValueError("TM01 manual base path must use route_mode='spatial_graph'")

    raw_samples = list(payload.get("samples", ()))
    samples: dict[int, dict[str, Any]] = {}
    for fallback_index, raw in enumerate(raw_samples):
        item = dict(raw)
        index = int(item.get("index", fallback_index))
        if index in samples:
            raise ValueError(f"duplicate TM01 manual base sample index {index}")
        samples[index] = item
    if sorted(samples) != list(range(len(samples))):
        raise ValueError("TM01 manual base sample indexes must be contiguous from zero")

    targets = dict(payload.get("station_targets", {}))
    result: dict[str, StationSpec] = {}
    for station_id in STATION_IDS:
        if station_id not in targets:
            raise KeyError(f"TM01 manual base path has no target {station_id!r}")
        sample_index = _target_index(targets[station_id])
        if sample_index not in samples:
            raise ValueError(
                f"TM01 station {station_id!r} targets missing sample {sample_index}"
            )
        sample = samples[sample_index]
        xy = np.asarray(sample.get("xy", ()), dtype=float).reshape(-1)
        yaw = float(sample.get("yaw"))
        if xy.size != 2 or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(
                f"TM01 station {station_id!r} has an invalid measured pose"
            )
        result[station_id] = StationSpec(
            name=station_id,
            base_candidates=[
                Pose(
                    position=(float(xy[0]), float(xy[1]), 0.0),
                    quaternion=yaw_to_quat_xyzw(yaw),
                )
            ],
            description="TM01 user-measured manual-path terminal pose",
            metadata={
                "source": str(path_value),
                "sample_index": int(sample_index),
                "sample_name": sample.get("name"),
                "yaw": yaw,
            },
        )
    return result


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Expose task-owned anchors for diagnostics and obstacle inspection."""

    stove = getattr(env, "stove", None)
    left = env.cabinet_for_drawer("drawer_left")
    center = env.cabinet_for_drawer("drawer_right")
    return {
        STOVE_STATION: {
            "station_id": STOVE_STATION,
            "fixture": stove,
            "anchor_xy": _xy(getattr(stove, "pos", None)),
            "source_fixture": "stove",
            "source_object_or_phase": "front_right",
        },
        STOVE_SIDE_COMPLETION_STATION: {
            "station_id": STOVE_SIDE_COMPLETION_STATION,
            "fixture": stove,
            "anchor_xy": _button_xy(env, STOVE_SIDE_COMPLETION_BUTTON),
            "source_fixture": "stove",
            "source_object_or_phase": STOVE_SIDE_COMPLETION_BUTTON,
        },
        DRAWER_SIDE_COMPLETION_STATION: {
            "station_id": DRAWER_SIDE_COMPLETION_STATION,
            "fixture": left,
            "anchor_xy": _button_xy(env, DRAWER_SIDE_COMPLETION_BUTTON),
            "source_fixture": getattr(left, "name", "drawer_left"),
            "source_object_or_phase": DRAWER_SIDE_COMPLETION_BUTTON,
        },
        DRAWER_STATION: {
            "station_id": DRAWER_STATION,
            "fixture": center,
            "anchor_xy": _xy(getattr(center, "pos", None)),
            "source_fixture": getattr(center, "name", "drawer_right"),
            "source_object_or_phase": "compact_drawer_group",
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
    """Bind workflow station names to the measured spatial graph executor."""

    if not config.manual_base_path_enabled or config.manual_base_path_file is None:
        raise ValueError("TM01 requires its measured manual base path")
    sampler = SimpleStationSampler(
        stations=station_specs_from_manual_path(config.manual_base_path_file),
        seeds=SeedStream(base_seed=int(seed), namespace="tm01_navigation"),
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
            raise ValueError("TM01 station target index list cannot be empty")
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


def _button_xy(env: Any, name: str) -> np.ndarray | None:
    try:
        return _xy(env.completion_button_position(name))
    except Exception:
        return None


__all__ = [
    "DRAWER_STATION",
    "DRAWER_SIDE_COMPLETION_STATION",
    "NavigationAssembly",
    "STATION_IDS",
    "STOVE_SIDE_COMPLETION_STATION",
    "STOVE_STATION",
    "build_navigation_assembly",
    "build_navigation_config",
    "station_specs",
    "station_specs_from_manual_path",
]
