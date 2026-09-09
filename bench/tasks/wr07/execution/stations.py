"""WR-07 station aliases for the two sugar-plate work areas."""

from __future__ import annotations

from typing import Any

from bench.execution.navigation.env_adapter import (
    fixture_xy,
    safe_name,
    station_navigation_inspection as generic_station_navigation_inspection,
)
from bench.tasks.wr07.domain.contracts import SOURCE_SLOTS

SOURCE_PLATE_STATION = "counter_pick_station"
TARGET_PLATE_STATION = "plate_station"
STATION_IDS = (SOURCE_PLATE_STATION, TARGET_PLATE_STATION)

STATION_ALIASES: dict[Any, str] = {
    "source_plates": SOURCE_PLATE_STATION,
    "source_plate_station": SOURCE_PLATE_STATION,
    "counter_zones": SOURCE_PLATE_STATION,
    "counter_zone_left": SOURCE_PLATE_STATION,
    "counter_zone_center": SOURCE_PLATE_STATION,
    "counter_zone_right": SOURCE_PLATE_STATION,
    SOURCE_PLATE_STATION: SOURCE_PLATE_STATION,
    **{slot: SOURCE_PLATE_STATION for slot in SOURCE_SLOTS},
    **{f"{slot}_station": SOURCE_PLATE_STATION for slot in SOURCE_SLOTS},
    "target_plate": TARGET_PLATE_STATION,
    "target_plate_station": TARGET_PLATE_STATION,
    TARGET_PLATE_STATION: TARGET_PLATE_STATION,
    "target_plates": TARGET_PLATE_STATION,
}


def station_aliases() -> dict[Any, str]:
    return dict(STATION_ALIASES)


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    """Return fixture anchors used when calibrated station poses are disabled."""

    source_fixture = getattr(env, "counter_zones_fixture", None)
    target_fixture = getattr(env, "meal_counter", None)
    return {
        SOURCE_PLATE_STATION: {
            "station_id": SOURCE_PLATE_STATION,
            "fixture": source_fixture,
            "anchor_xy": fixture_xy(source_fixture),
            "source_fixture": safe_name(source_fixture),
            "source_object_or_phase": "source_colored_sugar_plates",
        },
        TARGET_PLATE_STATION: {
            "station_id": TARGET_PLATE_STATION,
            "fixture": target_fixture,
            "anchor_xy": fixture_xy(target_fixture),
            "source_fixture": safe_name(target_fixture),
            "source_object_or_phase": "target_colored_sugar_plates",
        },
    }


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
        expected_candidate_count=len(STATION_IDS),
    )


__all__ = [
    "SOURCE_PLATE_STATION",
    "TARGET_PLATE_STATION",
    "STATION_ALIASES",
    "STATION_IDS",
    "station_aliases",
    "station_navigation_inspection",
    "station_specs",
]
