"""Pure TM04 sugar-containment geometry."""

from __future__ import annotations

from typing import Any

import numpy as np


def point_is_inside_cylindrical_region(
    point_world: Any,
    *,
    center_world: Any,
    rotation_world_from_local: Any,
    radius_m: float,
    half_height_m: float,
) -> bool:
    point = np.asarray(point_world, dtype=float).reshape(-1)
    center = np.asarray(center_world, dtype=float).reshape(-1)
    rotation = np.asarray(rotation_world_from_local, dtype=float)
    radius = float(radius_m)
    half_height = float(half_height_m)
    if point.size != 3 or center.size != 3 or rotation.shape != (3, 3):
        raise ValueError("TM04 containment requires 3D points and a 3x3 rotation")
    if not (
        np.all(np.isfinite(point))
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(rotation))
        and np.isfinite(radius)
        and np.isfinite(half_height)
    ):
        raise ValueError("TM04 containment inputs must be finite")
    if radius <= 0 or half_height <= 0:
        raise ValueError("TM04 containment dimensions must be positive")
    local = rotation.T @ (point - center)
    return bool(
        float(np.linalg.norm(local[:2])) <= radius + 1e-9
        and abs(float(local[2])) <= half_height + 1e-9
    )


__all__ = ["point_is_inside_cylindrical_region"]
