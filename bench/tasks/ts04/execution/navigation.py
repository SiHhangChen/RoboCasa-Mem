"""TS04 fixture stations and mobile-base controller assembly."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
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
from bench.execution.navigation.env_adapter import fixture_xy, safe_name
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.tasks.ts04.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts04.domain.geometry import station_id_for_fixture


COMPLETION_BUTTON_STATION = "completion_button_station"
STATION_IDS = tuple(station_id_for_fixture(name) for name in FIXTURE_NAMES) + (
    COMPLETION_BUTTON_STATION,
)
STATION_ALIASES: dict[Any, str] = {
    fixture: station_id_for_fixture(fixture) for fixture in FIXTURE_NAMES
}
STATION_ALIASES.update({station: station for station in STATION_IDS})
STATION_ALIASES.update(
    {
        "cabinet": station_id_for_fixture(FIXTURE_NAMES[1]),
        "storage_cabinet": station_id_for_fixture(FIXTURE_NAMES[1]),
        "completion_button": COMPLETION_BUTTON_STATION,
    }
)


@dataclass(frozen=True)
class NavigationAssembly:
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
    """Convert YAML settings to the generic navigation dataclass."""

    values = dict(settings)
    allowed = {field.name for field in fields(NavigationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown TS04 navigation settings: " + ", ".join(unknown))
    values["station_base_pose_file"] = Path(station_base_pose_file)
    values["manual_base_path_file"] = Path(manual_base_path_file)
    return replace(NavigationConfig(), **values)


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Build one independently anchored station for every drawer."""

    result: dict[str, dict[str, Any]] = {}
    fixture_names = tuple(getattr(env, "fixture_names", FIXTURE_NAMES))
    for fixture_name in fixture_names:
        fixture = env.cabinet_for_slot(fixture_name)
        anchor = _slot_anchor_xy(env, fixture_name)
        fallback = fixture_xy(fixture)
        result[station_id_for_fixture(fixture_name)] = {
            "station_id": station_id_for_fixture(fixture_name),
            "fixture": fixture,
            "anchor_xy": anchor if anchor is not None else fallback,
            "nominal_xy": _fixture_front_xy(fixture),
            "source_fixture": safe_name(fixture),
            "source_object_or_phase": fixture_name,
            "stand_off": 0.75,
            "tangential_offsets": (0.0,),
        }
    reference_fixture_name = fixture_names[1]
    center_fixture = env.cabinet_for_slot(reference_fixture_name)
    try:
        button_xy = np.asarray(
            env.completion_button_position(), dtype=float
        ).reshape(-1)[:2]
    except Exception:
        button_xy = _slot_anchor_xy(env, reference_fixture_name)
    result[COMPLETION_BUTTON_STATION] = {
        "station_id": COMPLETION_BUTTON_STATION,
        "fixture": center_fixture,
        "anchor_xy": button_xy,
        "nominal_xy": _fixture_front_xy(center_fixture),
        "source_fixture": "completion_button",
        "source_object_or_phase": "completion_button",
        "stand_off": 0.75,
        "tangential_offsets": (0.0,),
    }
    return result


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    config: NavigationConfig,
    seed: int,
    base_driver: Any | None = None,
) -> NavigationAssembly:
    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    driver = base_driver or BaseDriverAdapter(env, obs, nav_config=config)
    pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    fixture_names = tuple(getattr(env, "fixture_names", FIXTURE_NAMES))
    station_aliases = _station_aliases(fixture_names)
    if config.use_station_base_poses:
        if config.station_base_pose_file is None:
            raise ValueError("TS04 use_station_base_poses requires station poses")
        sampler: Any = StationPoseSampler.from_file(
            config.station_base_pose_file,
            config=config,
            seed=int(seed),
            base_xy_provider=pose_provider,
            station_aliases=station_aliases,
        )
    else:
        generated = station_specs_from_candidates(
            adapter.station_candidates(),
            aliases={
                key: value
                for key, value in station_aliases.items()
                if isinstance(key, str)
            },
        )
        sampler = SimpleStationSampler(
            stations=generated,
        seeds=SeedStream(base_seed=int(seed), namespace="ts04_navigation"),
        )
    planner = None
    if not config.manual_base_path_enabled and config.use_station_astar_paths:
        planner = StationPathPlanner(
            obstacles=adapter.build_fixture_obstacles,
            config=config,
        )
    controller = NavigationController(
        station_sampler=sampler,
        base_driver=driver,
        path_planner=planner,
        base_pose_provider=pose_provider,
    )
    return NavigationAssembly(controller, adapter, sampler, planner, driver, config)


def _station_aliases(fixture_names: tuple[str, ...]) -> dict[Any, str]:
    station_ids = tuple(station_id_for_fixture(name) for name in fixture_names)
    aliases: dict[Any, str] = {
        fixture: station_id_for_fixture(fixture) for fixture in fixture_names
    }
    aliases.update({station: station for station in station_ids})
    aliases.update(
        {
            "cabinet": station_id_for_fixture(fixture_names[1]),
            "storage_cabinet": station_id_for_fixture(fixture_names[1]),
            "completion_button": COMPLETION_BUTTON_STATION,
            COMPLETION_BUTTON_STATION: COMPLETION_BUTTON_STATION,
        }
    )
    return aliases


def _slot_anchor_xy(env: Any, fixture_name: str) -> np.ndarray | None:
    try:
        return np.asarray(env.slot_pose(fixture_name), dtype=float).reshape(-1)[:2]
    except Exception:
        return None


def _fixture_front_xy(fixture: Any) -> np.ndarray | None:
    try:
        return np.asarray(fixture.front_pose(distance=0.75), dtype=float).reshape(-1)[
            :2
        ]
    except Exception:
        return None


__all__ = [
    "NavigationAssembly",
    "COMPLETION_BUTTON_STATION",
    "STATION_ALIASES",
    "STATION_IDS",
    "build_navigation_assembly",
    "build_navigation_config",
    "station_specs",
]
