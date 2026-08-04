"""Pure WR-01 geometry validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.tasks.wr03.domain.contracts import CABINET_LEVELS, ITEM_IDS


# Validated validated ``cabinet_slot_0`` volume in the slide-cabinet frame.
STORAGE_INTERIOR_LOCAL_CENTER = (-0.225, -0.18, 0.0)
STORAGE_INTERIOR_HALF_EXTENTS = (0.18, 0.09, 0.16)


def validate_item_level_mapping(mapping: Mapping[str, Any]) -> dict[str, str]:
    """Validate a bijective level -> item mapping without simulator access."""

    result = {str(level): str(mapping[level]) for level in CABINET_LEVELS}
    if set(result) != set(CABINET_LEVELS) or set(result.values()) != set(ITEM_IDS):
        raise ValueError("WR-01 requires a bijective lower/upper item assignment")
    return result


def finite_xyz(value: Any, *, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain three finite coordinates")
    return float(array[0]), float(array[1]), float(array[2])


def level_world_xyz(
    lower_xyz: Any,
    *,
    upper_z_offset: float,
) -> dict[str, tuple[float, float, float]]:
    lower = finite_xyz(lower_xyz, label="cabinet.world_xyz")
    offset = float(upper_z_offset)
    if not np.isfinite(offset) or offset <= 0.0:
        raise ValueError("cabinet.upper_z_offset must be positive and finite")
    return {
        "lower": lower,
        "upper": (lower[0], lower[1], lower[2] + offset),
    }


def point_in_storage_interior(local_xyz: Any, *, margin: float = 0.035) -> bool:
    """Return whether a fixture-local point is inside previous cabinet slot 0."""

    point = np.asarray(finite_xyz(local_xyz, label="storage local point"), dtype=float)
    resolved_margin = float(margin)
    if not np.isfinite(resolved_margin) or resolved_margin < 0.0:
        raise ValueError("storage containment margin must be finite and non-negative")
    center = np.asarray(STORAGE_INTERIOR_LOCAL_CENTER, dtype=float)
    half_extents = np.asarray(STORAGE_INTERIOR_HALF_EXTENTS, dtype=float)
    return bool(np.all(np.abs(point - center) <= half_extents + resolved_margin))


__all__ = [
    "STORAGE_INTERIOR_HALF_EXTENTS",
    "STORAGE_INTERIOR_LOCAL_CENTER",
    "finite_xyz",
    "level_world_xyz",
    "point_in_storage_interior",
    "validate_item_level_mapping",
]
