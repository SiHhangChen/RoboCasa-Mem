"""Pure TS01 sugar-tray and saucepan release geometry."""

from __future__ import annotations

from typing import Any

import numpy as np


SUGAR_PLATE_MIN_SPACING_SPARSE = 0.08
SUGAR_PLATE_MIN_SPACING_DENSE = 0.045
SUGAR_PLATE_EDGE_MARGIN = 0.012
SUGAR_RELEASE_REGIONS_XY = {
    "A": (
        (-0.0460, 0.0000),
        (-0.0500, 0.0000),
        (-0.0540, 0.0000),
    ),
    "B": (
        (-0.0325, -0.0325),
        (-0.0354, -0.0354),
        (-0.0382, -0.0382),
    ),
    "C": (
        (0.0000, -0.0460),
        (0.0000, -0.0500),
        (0.0000, -0.0540),
    ),
}
SUGAR_RELEASE_REGION_NAMES = tuple(SUGAR_RELEASE_REGIONS_XY)
GRASP_SAFE_SLOTS = np.array(
    [
        (-0.86, -0.35),
        (0.0, -0.35),
        (0.86, -0.35),
        (-0.43, 0.35),
        (0.43, 0.35),
    ],
    dtype=float,
)


def sugar_plate_min_spacing(count: int) -> float:
    """Return the guaranteed center spacing for N cubes."""

    return (
        SUGAR_PLATE_MIN_SPACING_SPARSE
        if int(count) <= 3
        else SUGAR_PLATE_MIN_SPACING_DENSE
    )


def validate_sugar_plate_offsets_xy(
    offsets_xy: Any,
    *,
    count: int,
    safe_half_extents_xy: Any,
) -> np.ndarray:
    """Validate that complete cubes fit and remain separated on the tray."""

    offsets = np.asarray(offsets_xy, dtype=float).reshape(-1, 2)
    safe = np.asarray(safe_half_extents_xy, dtype=float).reshape(-1)
    if offsets.shape != (int(count), 2):
        raise ValueError(f"expected {int(count)} sugar offsets, got shape {offsets.shape}")
    if safe.size != 2 or not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(safe)):
        raise ValueError("sugar offsets and tray bounds must be finite")
    if np.any(safe <= 0.0):
        raise ValueError("tray bounds must be positive")
    if np.any(np.abs(offsets) > safe + 1e-9):
        raise ValueError(f"sugar offset lies outside safe tray bounds {safe.tolist()}")
    minimum = sugar_plate_min_spacing(int(count))
    for left in range(len(offsets)):
        for right in range(left + 1, len(offsets)):
            distance = float(np.linalg.norm(offsets[left] - offsets[right]))
            if distance + 1e-9 < minimum:
                raise ValueError(
                    f"sugar offsets are only {distance:.6f} m apart; minimum is {minimum:.6f} m"
                )
    return offsets


def sample_sugar_plate_offsets_xy(
    *,
    count: int,
    rng: Any,
    safe_half_extents_xy: Any,
) -> tuple[tuple[float, float], ...]:
    """Sample reproducible grasp-safe positions inside a tray."""

    safe = np.asarray(safe_half_extents_xy, dtype=float).reshape(2)
    if int(count) < 1 or int(count) > len(GRASP_SAFE_SLOTS):
        raise ValueError(f"total sugar count must be between 1 and {len(GRASP_SAFE_SLOTS)}, got {count}")
    last_error: ValueError | None = None
    for _ in range(256):
        slot_indices = np.asarray(rng.permutation(len(GRASP_SAFE_SLOTS)), dtype=int)[: int(count)]
        jitter = np.asarray(
            rng.uniform((-0.025, -0.03), (0.025, 0.03), size=(int(count), 2)),
            dtype=float,
        )
        offsets = (GRASP_SAFE_SLOTS[slot_indices] + jitter) * safe
        try:
            checked = validate_sugar_plate_offsets_xy(
                offsets,
                count=int(count),
                safe_half_extents_xy=safe,
            )
        except ValueError as exc:
            last_error = exc
            continue
        return tuple(tuple(float(value) for value in row) for row in checked)
    raise ValueError(f"could not sample {int(count)} separated sugar positions after 256 attempts") from last_error


def sample_sugar_plate_offsets_for_seed(
    *,
    count: int,
    seed: int,
    safe_half_extents_xy: Any,
) -> tuple[tuple[float, float], ...]:
    """Sample tray offsets from an isolated episode seed stream."""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x54524159]))
    return sample_sugar_plate_offsets_xy(
        count=count,
        rng=rng,
        safe_half_extents_xy=safe_half_extents_xy,
    )


def sugar_release_region_offsets_xy(region_name: str) -> tuple[tuple[float, float], ...]:
    """Return the three fixed XY candidates in one reusable region."""

    name = str(region_name).strip().upper()
    if name not in SUGAR_RELEASE_REGIONS_XY:
        raise ValueError(
            f"unknown sugar release region {region_name!r}; "
            f"expected one of {SUGAR_RELEASE_REGION_NAMES}"
        )
    return SUGAR_RELEASE_REGIONS_XY[name]


def sugar_release_offset_is_inside(
    offset_xy: Any,
    *,
    max_radius_m: float,
) -> bool:
    """Check an XY candidate against the configured saucepan interior radius."""

    offset = np.asarray(offset_xy, dtype=float).reshape(-1)
    radius = float(max_radius_m)
    return bool(
        offset.size == 2
        and np.all(np.isfinite(offset))
        and np.isfinite(radius)
        and radius > 0.0
        and float(np.linalg.norm(offset)) <= radius + 1e-12
    )


def point_is_inside_cylindrical_region(
    point_world: Any,
    *,
    center_world: Any,
    rotation_world_from_local: Any,
    radius_m: float,
    half_height_m: float,
) -> bool:
    """Return whether a world-space point lies inside a finite cylinder.

    TS01 uses this with the saucepan's MuJoCo ``reg_int`` cylinder and each
    sugar cube's body center.  Deliberately checking geometry instead of
    direct object contact lets a cube resting on another cube count as being
    in the hot chocolate.
    """

    point = np.asarray(point_world, dtype=float).reshape(-1)
    center = np.asarray(center_world, dtype=float).reshape(-1)
    rotation = np.asarray(rotation_world_from_local, dtype=float)
    radius = float(radius_m)
    half_height = float(half_height_m)
    if point.size != 3 or center.size != 3 or rotation.shape != (3, 3):
        raise ValueError("cylindrical containment requires 3D points and a 3x3 rotation")
    if not (
        np.all(np.isfinite(point))
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(rotation))
        and np.isfinite(radius)
        and np.isfinite(half_height)
    ):
        raise ValueError("cylindrical containment inputs must be finite")
    if radius <= 0.0 or half_height <= 0.0:
        raise ValueError("cylinder radius and half-height must be positive")

    point_local = rotation.T @ (point - center)
    radial_distance = float(np.linalg.norm(point_local[:2]))
    return bool(
        radial_distance <= radius + 1e-9
        and abs(float(point_local[2])) <= half_height + 1e-9
    )


__all__ = [
    "GRASP_SAFE_SLOTS",
    "SUGAR_PLATE_EDGE_MARGIN",
    "SUGAR_PLATE_MIN_SPACING_DENSE",
    "SUGAR_PLATE_MIN_SPACING_SPARSE",
    "SUGAR_RELEASE_REGIONS_XY",
    "SUGAR_RELEASE_REGION_NAMES",
    "point_is_inside_cylindrical_region",
    "sample_sugar_plate_offsets_for_seed",
    "sample_sugar_plate_offsets_xy",
    "sugar_plate_min_spacing",
    "sugar_release_offset_is_inside",
    "sugar_release_region_offsets_xy",
    "validate_sugar_plate_offsets_xy",
]
