"""Utilities for anchored world target pose calibration."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AnchoredWorldTarget:
    """A target represented as an offset from an anchor point."""

    anchor_world: np.ndarray
    target_world: np.ndarray
    offset_world: np.ndarray
    height: float


def vector3(value: Any, *, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Parse a 3-vector from JSON-like values or comma-separated text."""
    if value is None:
        return np.array(default, dtype=float)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        if len(parts) != 3:
            return np.array(default, dtype=float)
        try:
            arr = np.array([float(part) for part in parts], dtype=float)
        except ValueError:
            return np.array(default, dtype=float)
    else:
        try:
            arr = np.array(value, dtype=float).reshape(3)
        except Exception:
            return np.array(default, dtype=float)
    if not np.all(np.isfinite(arr)):
        return np.array(default, dtype=float)
    return arr


def anchored_target_from_operation(
    operation: MutableMapping[str, Any],
    *,
    anchor_world: np.ndarray,
    offset_key: str,
    height_key: str | None = None,
) -> AnchoredWorldTarget:
    """Build a calibrated world target from an operation payload."""
    anchor = np.array(anchor_world, dtype=float).reshape(3)
    offset = vector3(operation.get(offset_key))
    height = float(operation.get(height_key, 0.0)) if height_key else float(offset[2])
    target = anchor + np.array([offset[0], offset[1], height], dtype=float)
    return AnchoredWorldTarget(
        anchor_world=anchor,
        target_world=target,
        offset_world=offset,
        height=height,
    )


def update_operation_from_target(
    operation: MutableMapping[str, Any],
    *,
    anchor_world: np.ndarray,
    target_world: np.ndarray,
    offset_key: str,
    height_key: str | None = None,
) -> AnchoredWorldTarget:
    """Write a calibrated target back as an anchor-relative offset."""
    anchor = np.array(anchor_world, dtype=float).reshape(3)
    target = np.array(target_world, dtype=float).reshape(3)
    delta = target - anchor
    if height_key:
        offset = np.array([delta[0], delta[1], 0.0], dtype=float)
        height = float(delta[2])
        operation[offset_key] = offset.astype(float).tolist()
        operation[height_key] = height
    else:
        offset = delta.astype(float)
        height = float(delta[2])
        operation[offset_key] = offset.astype(float).tolist()
    return AnchoredWorldTarget(
        anchor_world=anchor,
        target_world=target,
        offset_world=offset,
        height=height,
    )


__all__ = [
    "AnchoredWorldTarget",
    "anchored_target_from_operation",
    "update_operation_from_target",
    "vector3",
]
