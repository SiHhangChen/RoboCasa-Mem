from __future__ import annotations

import numpy as np
import pytest

from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wx01.domain.geometry import sample_wipe_offsets


def test_production_wipe_angles_stay_in_reliable_left_right_sector() -> None:
    for seed in range(200):
        episode = TaskCatalog.resolve(EpisodeRequest(task_id="WX-01", seed=seed))
        metadata = dict(episode.facts["wipe_metadata"])

        assert metadata["angle_range_deg"] == [60.0, 120.0]
        assert 60.0 <= float(metadata["angle_deg"]) <= 120.0
        direction = np.asarray(metadata["direction_xy"], dtype=float)
        assert abs(float(direction[1])) >= abs(float(direction[0]))


def test_wipe_sampling_remains_seeded_and_bidirectional() -> None:
    settings = {
        "seed_offset": 2000,
        "half_length": 0.07,
        "pass_count": 3,
        "angle_range_deg": [60.0, 120.0],
        "angle_offset_deg": 0.0,
    }

    offsets, metadata = sample_wipe_offsets(episode_seed=17, settings=settings)
    repeated_offsets, repeated_metadata = sample_wipe_offsets(
        episode_seed=17, settings=settings
    )

    assert offsets == repeated_offsets
    assert metadata == repeated_metadata
    assert len(offsets) == 4
    np.testing.assert_allclose(offsets[0], offsets[2])
    np.testing.assert_allclose(offsets[1], offsets[3])
    np.testing.assert_allclose(offsets[0], -np.asarray(offsets[1]))
    assert np.linalg.norm(np.asarray(offsets[0], dtype=float)) == pytest.approx(0.07)


@pytest.mark.parametrize(
    "angle_range",
    ([60.0], [120.0, 60.0], [-1.0, 60.0], [60.0, 181.0]),
)
def test_wipe_sampling_rejects_invalid_angle_ranges(angle_range: list[float]) -> None:
    with pytest.raises(ValueError, match="angle_range_deg"):
        sample_wipe_offsets(
            episode_seed=0,
            settings={"angle_range_deg": angle_range},
        )
