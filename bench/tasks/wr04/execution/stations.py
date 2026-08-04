"""WR-04 station aliases and fixture-derived station specs."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.navigation.env_adapter import (
    fixture_xy,
    safe_name,
    station_navigation_inspection as generic_station_navigation_inspection,
)
from bench.tasks.wr04.domain.contracts import HOME_REGIONS


STATION_ALIASES: dict[Any, str] = {
    "shelf_lower": "shelf_lower_station",
    "shelf_lower_station": "shelf_lower_station",
    "shelf_upper": "shelf_upper_station",
    "shelf_upper_station": "shelf_upper_station",
    "sink": "sink_place_station",
    "sink_station": "sink_place_station",
    "sink_place_station": "sink_place_station",
    ("ketchup", "sink_place"): "sink_place_station",
    ("ketchup", "sink_pick"): "sink_place_station",
}

STATION_IDS = (
    "shelf_lower_station",
    "shelf_upper_station",
    "sink_place_station",
)
SINK_STATION_STAND_OFF = 0.24


def station_aliases() -> dict[Any, str]:
    return dict(STATION_ALIASES)


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Return task-owned station specs for fixture-edge candidate generation."""
    storage_fixture = getattr(env, "item_shelf", None)
    sink = getattr(env, "sink", None)
    sink_counter = getattr(env, "sink_counter", None)
    specs: dict[str, dict[str, Any]] = {}

    storage_xy = fixture_xy(storage_fixture)
    for region_name in HOME_REGIONS:
        home_xy = _home_region_xy(env, region_name)
        specs[f"{region_name}_station"] = {
            "station_id": f"{region_name}_station",
            "fixture": storage_fixture,
            "anchor_xy": home_xy if home_xy is not None else storage_xy,
            "source_fixture": safe_name(storage_fixture),
            "source_object_or_phase": region_name,
        }

    specs["sink_place_station"] = {
        "station_id": "sink_place_station",
        "fixture": sink_counter,
        "anchor_xy": fixture_xy(sink),
        "source_fixture": safe_name(sink),
        "source_object_or_phase": "sink_left_right_staging",
        "stand_off": SINK_STATION_STAND_OFF,
    }
    return specs


def station_navigation_inspection(
    env: Any,
    obs: dict[str, Any],
    *,
    resolution: float = 0.04,
    inflation: float = 0.35,
    stand_off: float = 0.50,
) -> dict[str, Any]:
    return generic_station_navigation_inspection(
        env,
        obs,
        station_specs(env),
        resolution=resolution,
        inflation=inflation,
        stand_off=stand_off,
        expected_station_ids=STATION_IDS,
        expected_candidate_count=3,
    )


def _home_region_xy(env: Any, region_name: str) -> np.ndarray | None:
    try:
        return _pose_xy(env.home_region_center(region_name))
    except Exception:
        pass
    if str(region_name) == str(getattr(env, "demo_original_level", "")):
        pose = object_xy(env, "ketchup")
        if pose is not None:
            return pose
    return None


def _pose_xy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.array(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return None
    return arr[:2]


__all__ = [
    "STATION_ALIASES",
    "STATION_IDS",
    "SINK_STATION_STAND_OFF",
    "station_aliases",
    "station_navigation_inspection",
    "station_specs",
]
