from __future__ import annotations

import numpy as np
import pytest

from bench.tasks.ts03.config import load_task_config
from bench.tasks.ts03.domain.geometry import (
    GRASP_SAFE_SLOTS,
    SUGAR_PLATE_SAMPLING_RADIUS_M,
    sample_sugar_cube_yaws_for_seed,
    sample_sugar_plate_offsets_for_seed,
    sugar_plate_min_spacing,
    validate_sugar_plate_offsets_xy,
)


SAFE_HALF_EXTENTS = np.asarray([0.11564878202215918, 0.06039817852215919])


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("spacing", [0.049, 0.05, 0.06])
def test_minimum_center_spacing_is_five_centimeters(count, spacing):
    assert sugar_plate_min_spacing(count) == 0.05
    offsets = [(0.0, 0.0), (spacing, 0.0)]
    if count == 3:
        offsets.append((-0.1, 0.0))
    if spacing < 0.05:
        with pytest.raises(ValueError, match="minimum is 0.050000 m"):
            validate_sugar_plate_offsets_xy(
                offsets, count=count, safe_half_extents_xy=SAFE_HALF_EXTENTS,
            )
    else:
        validate_sugar_plate_offsets_xy(
            offsets, count=count, safe_half_extents_xy=SAFE_HALF_EXTENTS,
        )


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_disk_samples_are_bounded_separated_and_reproducible(count):
    positions = set()
    centers = GRASP_SAFE_SLOTS * SAFE_HALF_EXTENTS
    for seed in range(128):
        samples = sample_sugar_plate_offsets_for_seed(
            count=count, seed=seed, safe_half_extents_xy=SAFE_HALF_EXTENTS,
        )
        assert samples == sample_sugar_plate_offsets_for_seed(
            count=count, seed=seed, safe_half_extents_xy=SAFE_HALF_EXTENTS,
        )
        offsets = validate_sugar_plate_offsets_xy(
            samples, count=count, safe_half_extents_xy=SAFE_HALF_EXTENTS,
        )
        distances = np.linalg.norm(offsets[:, None, :] - centers[None, :, :], axis=2)
        assert np.all(distances.min(axis=1) <= SUGAR_PLATE_SAMPLING_RADIUS_M + 1e-12)
        assert np.all(distances.min(axis=1) > 0.0)
        assert len(set(distances.argmin(axis=1))) == count
        positions.add(samples)
    assert len(positions) == 128


def test_disk_sampling_is_uniform_in_area_not_radius():
    safe = np.asarray([0.2, 0.2])
    centers = GRASP_SAFE_SLOTS * safe
    deltas = []
    for seed in range(2048):
        point = np.asarray(sample_sugar_plate_offsets_for_seed(
            count=1, seed=seed, safe_half_extents_xy=safe,
        )[0])
        center = centers[np.linalg.norm(centers - point, axis=1).argmin()]
        deltas.append((point - center) / SUGAR_PLATE_SAMPLING_RADIUS_M)
    deltas = np.asarray(deltas)
    assert np.mean(np.sum(deltas ** 2, axis=1)) == pytest.approx(0.5, abs=0.03)
    np.testing.assert_allclose(deltas.mean(axis=0), 0.0, atol=0.04)
    assert np.max(np.abs(deltas[:, 0])) > 0.9
    assert np.max(np.abs(deltas[:, 1])) > 0.9


def test_yaws_are_reproducible_independent_and_cover_full_rotation():
    draws = []
    for seed in range(128):
        yaws = sample_sugar_cube_yaws_for_seed(count=3, seed=seed)
        assert yaws == sample_sugar_cube_yaws_for_seed(count=3, seed=seed)
        assert yaws[:2] == sample_sugar_cube_yaws_for_seed(count=2, seed=seed)
        assert len(set(yaws)) == 3
        assert all(-np.pi <= yaw < np.pi for yaw in yaws)
        sample_sugar_plate_offsets_for_seed(
            count=3, seed=seed, safe_half_extents_xy=SAFE_HALF_EXTENTS,
        )
        assert yaws == sample_sugar_cube_yaws_for_seed(count=3, seed=seed)
        draws.extend(yaws)
    assert min(draws) < -3.0
    assert max(draws) > 3.0
    assert len(set(draws)) == len(draws)


def test_zero_radius_and_zero_yaw_range_disable_variation():
    samples = sample_sugar_plate_offsets_for_seed(
        count=3, seed=7, safe_half_extents_xy=SAFE_HALF_EXTENTS, sampling_radius_m=0.0,
    )
    centers = GRASP_SAFE_SLOTS * SAFE_HALF_EXTENTS
    for point in samples:
        assert np.linalg.norm(centers - point, axis=1).min() == 0.0
    assert sample_sugar_cube_yaws_for_seed(count=3, seed=7, half_range_rad=0.0) == (0.0,) * 3


@pytest.mark.parametrize("radius", [-0.001, float("nan"), float("inf")])
def test_invalid_disk_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="sampling radius"):
        sample_sugar_plate_offsets_for_seed(
            count=2, seed=0, safe_half_extents_xy=SAFE_HALF_EXTENTS, sampling_radius_m=radius,
        )


@pytest.mark.parametrize("half_range", [-0.1, np.pi + 0.1, float("nan"), float("inf")])
def test_invalid_yaw_range_is_rejected(half_range):
    with pytest.raises(ValueError, match="yaw half-range"):
        sample_sugar_cube_yaws_for_seed(count=2, seed=0, half_range_rad=half_range)


def test_production_config_enables_one_centimeter_disks_and_full_yaw():
    objects = load_task_config().initial_state.objects
    assert objects["sugar_plate_sampling_radius_m"] == 0.005
    assert objects["sugar_cube_yaw_half_range_deg"] == 180.0
