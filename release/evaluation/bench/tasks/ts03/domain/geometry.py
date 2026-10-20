"""Pure TS03 sugar-tray and saucepan release geometry."""

from __future__ import annotations

from typing import Any

import numpy as np


SUGAR_PLATE_MIN_SPACING_SPARSE = 0.05
SUGAR_PLATE_MIN_SPACING_DENSE = 0.045
SUGAR_PLATE_EDGE_MARGIN = 0.012
SUGAR_PLATE_SAMPLING_RADIUS_M = 0.005
SUGAR_CUBE_YAW_SALT = 0x59415733
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
# Normalized tray-local slot centers. Outer X used to sit at 0.86 of the safe
# half-extent (~10 cm), which put some cubes outside a comfortable Panda reach
# from the sugar station. Pull X inward; keep Y so N=3 still clears the 5 cm
# center-spacing floor after 5 mm disk jitter.
GRASP_SAFE_SLOTS = np.array(
    [
        (-0.64, -0.35),
        (0.0, -0.35),
        (0.64, -0.35),
        (-0.32, 0.35),
        (0.32, 0.35),
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
    sampling_radius_m: float = SUGAR_PLATE_SAMPLING_RADIUS_M,
) -> tuple[tuple[float, float], ...]:
    """Sample positions uniformly in disks around grasp-safe slot centers."""

    safe = np.asarray(safe_half_extents_xy, dtype=float).reshape(2)
    if not np.all(np.isfinite(safe)) or np.any(safe <= 0.0):
        raise ValueError("tray bounds must be finite and positive")
    radius = float(sampling_radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("sugar plate sampling radius must be finite and non-negative")
    if int(count) < 1 or int(count) > len(GRASP_SAFE_SLOTS):
        raise ValueError(f"total sugar count must be between 1 and {len(GRASP_SAFE_SLOTS)}, got {count}")
    last_error: ValueError | None = None
    for _ in range(256):
        slot_indices = np.asarray(rng.permutation(len(GRASP_SAFE_SLOTS)), dtype=int)[: int(count)]
        angles = rng.uniform(0.0, 2.0 * np.pi, size=int(count))
        distances = radius * np.sqrt(rng.uniform(0.0, 1.0, size=int(count)))
        jitter = distances[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        offsets = GRASP_SAFE_SLOTS[slot_indices] * safe + jitter
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
    sampling_radius_m: float = SUGAR_PLATE_SAMPLING_RADIUS_M,
) -> tuple[tuple[float, float], ...]:
    """Sample tray offsets from an isolated episode seed stream."""

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x54524159]))
    return sample_sugar_plate_offsets_xy(
        count=count,
        rng=rng,
        safe_half_extents_xy=safe_half_extents_xy,
        sampling_radius_m=sampling_radius_m,
    )


def sample_sugar_cube_yaws_for_seed(
    *,
    count: int,
    seed: int,
    half_range_rad: float = np.pi,
) -> tuple[float, ...]:
    """Sample world yaw independently of tray XY rejection sampling."""

    if int(count) < 1 or int(count) > len(GRASP_SAFE_SLOTS):
        raise ValueError(f"sugar count must be between 1 and {len(GRASP_SAFE_SLOTS)}")
    half_range = float(half_range_rad)
    if not np.isfinite(half_range) or not 0.0 <= half_range <= np.pi:
        raise ValueError("sugar cube yaw half-range must be finite and in [0, pi]")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), SUGAR_CUBE_YAW_SALT]))
    return tuple(float(yaw) for yaw in rng.uniform(-half_range, half_range, size=int(count)))


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
    "SUGAR_PLATE_SAMPLING_RADIUS_M",
    "SUGAR_PLATE_MIN_SPACING_DENSE",
    "SUGAR_PLATE_MIN_SPACING_SPARSE",
    "SUGAR_RELEASE_REGIONS_XY",
    "SUGAR_RELEASE_REGION_NAMES",
    "point_is_inside_cylindrical_region",
    "sample_sugar_cube_yaws_for_seed",
    "sample_sugar_plate_offsets_for_seed",
    "sample_sugar_plate_offsets_xy",
    "sugar_plate_min_spacing",
    "sugar_release_offset_is_inside",
    "sugar_release_region_offsets_xy",
    "validate_sugar_plate_offsets_xy",
]
