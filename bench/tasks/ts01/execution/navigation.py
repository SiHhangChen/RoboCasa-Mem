"""TS01 station mapping and mobile-base assembly."""

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
from bench.execution.navigation.env_adapter import fixture_xy, object_xy, safe_name
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.tasks.ts01.domain.contracts import (
    COMPLETION_BUTTON_NAME,
    RECEPTACLE_NAME,
    SUGAR_PLATE_NAME,
)


SUGAR_STATION = "sugar_station"
SAUCEPAN_STATION = "saucepan_station"
COMPLETION_BUTTON_STATION = "completion_button_station"
SUGAR_STATION_STAND_OFF = 0.16


@dataclass(frozen=True)
class NavigationAssembly:
    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


def station_aliases(sugar_names: tuple[str, ...] = ()) -> dict[Any, str]:
    aliases: dict[Any, str] = {
        SUGAR_STATION: SUGAR_STATION,
        SAUCEPAN_STATION: SAUCEPAN_STATION,
        COMPLETION_BUTTON_STATION: COMPLETION_BUTTON_STATION,
        COMPLETION_BUTTON_NAME: COMPLETION_BUTTON_STATION,
        "hot_chocolate": SAUCEPAN_STATION,
        RECEPTACLE_NAME: SAUCEPAN_STATION,
        "saucepan": SAUCEPAN_STATION,
        SUGAR_PLATE_NAME: SUGAR_STATION,
    }
    aliases.update({str(name): SUGAR_STATION for name in sugar_names})
    return aliases


def station_id_for_target(target_name: str) -> str:
    name = str(target_name)
    if name.startswith("sugar") and name[5:].isdigit():
        return SUGAR_STATION
    return str(station_aliases().get(name, name))


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    counter = getattr(env, "counter", None)
    stove = getattr(env, "stove", None)
    sugar_anchor = object_xy(env, SUGAR_PLATE_NAME)
    saucepan_anchor = object_xy(env, RECEPTACLE_NAME)
    completion_button_anchor = _completion_button_xy(env)
    return {
        SUGAR_STATION: {
            "station_id": SUGAR_STATION,
            "fixture": counter,
            "anchor_xy": sugar_anchor if sugar_anchor is not None else fixture_xy(counter),
            "source_fixture": safe_name(counter),
            "source_object_or_phase": SUGAR_PLATE_NAME,
            "stand_off": SUGAR_STATION_STAND_OFF,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        SAUCEPAN_STATION: {
            "station_id": SAUCEPAN_STATION,
            "fixture": stove,
            "anchor_xy": saucepan_anchor if saucepan_anchor is not None else fixture_xy(stove),
            "source_fixture": safe_name(stove),
            "source_object_or_phase": RECEPTACLE_NAME,
            "stand_off": 0.30,
            "tangential_offsets": (-0.08, -0.04, 0.0, 0.04, 0.08),
        },
        COMPLETION_BUTTON_STATION: {
            "station_id": COMPLETION_BUTTON_STATION,
            "fixture": counter,
            "anchor_xy": (
                completion_button_anchor
                if completion_button_anchor is not None
                else fixture_xy(counter)
            ),
            "source_fixture": safe_name(counter),
            "source_object_or_phase": COMPLETION_BUTTON_NAME,
            "stand_off": 0.30,
            "tangential_offsets": (-0.06, -0.03, 0.0, 0.03, 0.06),
        },
    }


def _completion_button_xy(env: Any) -> np.ndarray | None:
    try:
        position = np.asarray(env.completion_button_position(), dtype=float).reshape(-1)
    except Exception:
        return None
    if position.size < 2 or not np.all(np.isfinite(position[:2])):
        return None
    return position[:2]


def build_navigation_config(
    settings: dict[str, Any], *, manual_base_path_file: Path
) -> NavigationConfig:
    values = dict(settings)
    allowed = {field.name for field in fields(NavigationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError("unknown TS01 navigation settings: " + ", ".join(unknown))
    values["manual_base_path_file"] = Path(manual_base_path_file)
    values["station_base_pose_file"] = None
    return replace(NavigationConfig(), **values)


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
    aliases = station_aliases(tuple(getattr(env, "sugar_names", ())))
    if config.use_station_base_poses:
        if config.station_base_pose_file is None:
            raise ValueError("TS01 use_station_base_poses requires station_base_pose_file")
        sampler: Any = StationPoseSampler.from_file(
            config.station_base_pose_file,
            config=config,
            seed=int(seed),
            base_xy_provider=pose_provider,
            station_aliases=aliases,
        )
    else:
        generated = station_specs_from_candidates(
            adapter.station_candidates(),
            aliases={key: value for key, value in aliases.items() if isinstance(key, str)},
        )
        sampler = SimpleStationSampler(
            stations=generated,
            seeds=SeedStream(base_seed=int(seed), namespace="ts01_navigation"),
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
    return NavigationAssembly(controller, adapter, sampler, planner, driver, config)


__all__ = [
    "NavigationAssembly",
    "COMPLETION_BUTTON_STATION",
    "SAUCEPAN_STATION",
    "SUGAR_STATION",
    "build_navigation_assembly",
    "build_navigation_config",
    "station_aliases",
    "station_id_for_target",
    "station_specs",
]
