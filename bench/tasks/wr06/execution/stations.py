"""WR-06 station aliases and fixture-derived station specs.

Shelf stations follow the lower/upper object-relative pattern; the plate
staging uses the episode-decided side.  A ketchup bottle placed at the right
plate slot uses the WR-05-validated ``ketchup_plate_right_station`` (mirrored
left-slot base geometry) instead of a mustard-style right station.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.navigation.env_adapter import (
    fixture_xy,
    object_xy,
    safe_name,
    station_navigation_inspection as generic_station_navigation_inspection,
)
from bench.tasks.wr06.domain.contracts import HOME_REGIONS, ITEM_ID


STATION_ALIASES: dict[Any, str] = {
    "shelf_lower": "shelf_lower_station",
    "shelf_lower_station": "shelf_lower_station",
    "shelf_upper": "shelf_upper_station",
    "shelf_upper_station": "shelf_upper_station",
    "plate": "plate_left_station",
    "plate_station": "plate_left_station",
    "plate_left": "plate_left_station",
    "plate_left_station": "plate_left_station",
    "plate_left_pick_station": "plate_left_pick_station",
    "plate_right": "ketchup_plate_right_station",
    "ketchup_plate_right_station": "ketchup_plate_right_station",
    "plate_right_pick_station": "plate_right_pick_station",
}

STATION_IDS = (
    "shelf_lower_station",
    "shelf_upper_station",
    "plate_left_station",
    "plate_left_pick_station",
    "ketchup_plate_right_station",
    "plate_right_pick_station",
)
PLATE_STATION_STAND_OFF = 0.24


def station_aliases() -> dict[Any, str]:
    return dict(STATION_ALIASES)


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Return task-owned station specs for fixture-edge candidate generation."""
    storage_fixture = getattr(env, "item_shelf", None)
    meal_fixture = getattr(env, "meal_counter", None)
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

    plate_xy = object_xy(env, "plate")
    plate_anchor = plate_xy if plate_xy is not None else fixture_xy(meal_fixture)
    specs["plate_left_station"] = {
        "station_id": "plate_left_station",
        "fixture": meal_fixture,
        "anchor_xy": plate_anchor,
        "source_fixture": safe_name(meal_fixture),
        "source_object_or_phase": "plate_left_staging",
        "stand_off": PLATE_STATION_STAND_OFF,
    }
    specs["plate_left_pick_station"] = {
        "station_id": "plate_left_pick_station",
        "fixture": meal_fixture,
        "anchor_xy": plate_anchor,
        "source_fixture": safe_name(meal_fixture),
        "source_object_or_phase": "plate_left_wa02_side_pick",
        "stand_off": PLATE_STATION_STAND_OFF,
    }
    specs["ketchup_plate_right_station"] = {
        "station_id": "ketchup_plate_right_station",
        "fixture": meal_fixture,
        "anchor_xy": plate_anchor,
        "source_fixture": safe_name(meal_fixture),
        "source_object_or_phase": "ketchup_plate_right_staging",
        "stand_off": PLATE_STATION_STAND_OFF,
    }
    specs["plate_right_pick_station"] = {
        "station_id": "plate_right_pick_station",
        "fixture": meal_fixture,
        "anchor_xy": plate_anchor,
        "source_fixture": safe_name(meal_fixture),
        "source_object_or_phase": "plate_right_wa02_side_pick",
        "stand_off": PLATE_STATION_STAND_OFF,
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
        expected_candidate_count=6,
    )


def _home_region_xy(env: Any, region_name: str) -> np.ndarray | None:
    try:
        return _pose_xy(env.home_region_center(region_name))
    except Exception:
        pass
    if str(region_name) == str(getattr(env, "demo_original_level", "")):
        pose = object_xy(env, ITEM_ID)
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
    "PLATE_STATION_STAND_OFF",
    "station_aliases",
    "station_navigation_inspection",
    "station_specs",
]
