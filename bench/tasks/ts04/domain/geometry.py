"""Pure geometry and naming helpers for the two TS04 drawers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bench.tasks.ts04.domain.contracts import FIXTURE_NAMES


DEFAULT_DRAWER_SPACING = 0.50
DEFAULT_DRAWER_WORLD_YAW_DEG = 180.0
DEFAULT_DRAWER_WORLD_XYZS = (
    (4.10, -2.37, 1.1363333333333332),
    (4.60, -2.37, 1.1363333333333332),
    (5.10, -2.37, 1.1363333333333332),
)
MINIMUM_DRAWER_SEPARATION = 0.25


def normalize_drawer_world_xyzs(
    value: Any = DEFAULT_DRAWER_WORLD_XYZS,
    *,
    expected_count: int = len(FIXTURE_NAMES),
) -> tuple[tuple[float, float, float], ...]:
    """Validate and normalize one world XYZ per fixture."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("drawer_world_xyzs must be a sequence of three XYZ values")
    rows = tuple(tuple(float(component) for component in row) for row in value)
    count = int(expected_count)
    if count < 2:
        raise ValueError("drawer_world_xyzs requires at least two drawers")
    if len(rows) != count or any(len(row) != 3 for row in rows):
        raise ValueError(f"drawer_world_xyzs must contain exactly {count} XYZ values")
    array = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("drawer_world_xyzs must be finite")
    separation = minimum_drawer_separation(rows, expected_count=count)
    if separation < MINIMUM_DRAWER_SEPARATION:
        raise ValueError(
            "TS04 drawers must be spatially separated by at least "
            f"{MINIMUM_DRAWER_SEPARATION:.2f} m"
        )
    return rows


def minimum_drawer_separation(
    drawer_world_xyzs: Sequence[Sequence[float]],
    *,
    expected_count: int = len(FIXTURE_NAMES),
) -> float:
    """Return the minimum pairwise XY distance between drawer centers."""

    points = np.asarray(drawer_world_xyzs, dtype=float)
    count = int(expected_count)
    if points.shape != (count, 3) or not np.all(np.isfinite(points)):
        raise ValueError(f"expected a finite {count}x3 drawer position matrix")
    return min(
        float(np.linalg.norm(points[left, :2] - points[right, :2]))
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )


def station_id_for_fixture(fixture_name: str) -> str:
    """Map a task fixture identifier to its dedicated navigation station."""

    name = str(fixture_name)
    if name not in FIXTURE_NAMES:
        raise ValueError(f"unknown TS04 fixture: {fixture_name!r}")
    return f"{name}_station"


__all__ = [
    "DEFAULT_DRAWER_SPACING",
    "DEFAULT_DRAWER_WORLD_XYZS",
    "DEFAULT_DRAWER_WORLD_YAW_DEG",
    "MINIMUM_DRAWER_SEPARATION",
    "minimum_drawer_separation",
    "normalize_drawer_world_xyzs",
    "station_id_for_fixture",
]
