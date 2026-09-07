"""Deterministic uniform sampling inside bounded planar station circles."""

from __future__ import annotations

from typing import Any

import numpy as np


def sample_uniform_disk(
    center_xy: Any,
    radius_m: float,
    *,
    seed: int,
    salt: int,
) -> np.ndarray:
    """Return one reproducible point uniformly distributed over a disk."""

    center = np.asarray(center_xy, dtype=float).reshape(-1)
    if center.size < 2 or not np.all(np.isfinite(center[:2])):
        raise ValueError("circle center must contain two finite coordinates")
    radius = float(radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("circle radius must be finite and non-negative")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(salt)]))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    distance = radius * float(np.sqrt(rng.uniform(0.0, 1.0)))
    return center[:2] + distance * np.asarray(
        [np.cos(angle), np.sin(angle)], dtype=float
    )


def sample_uniform_yaw(
    center_yaw: float,
    half_range_rad: float,
    *,
    seed: int,
    salt: int,
) -> float:
    """Return one reproducible yaw around ``center_yaw``.

    The yaw draw uses its own salt so adding angular variation does not change
    the existing XY sequence sampled by :func:`sample_uniform_disk`.
    """

    center = float(center_yaw)
    half_range = float(half_range_rad)
    if not np.isfinite(center):
        raise ValueError("yaw center must be finite")
    if not np.isfinite(half_range) or not 0.0 <= half_range <= np.pi:
        raise ValueError("yaw half-range must be finite and in [0, pi]")
    if half_range == 0.0:
        return center
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(salt)]))
    return center + float(rng.uniform(-half_range, half_range))


__all__ = ["sample_uniform_disk", "sample_uniform_yaw"]
