"""TS01 randomization keeps count decisions and supports dense trays."""

from dataclasses import asdict

import numpy as np
import pytest

from bench.tasks.ts01.config import load_task_config, resolve_action_config
from bench.tasks.ts01.domain.episode import K_SALT, N_SALT, resolve_episode_plan
from bench.tasks.ts01.domain.geometry import (
    GRASP_SAFE_SLOTS, sample_sugar_plate_offsets_for_seed,
    sample_sugar_cube_yaws_for_seed, validate_sugar_plate_offsets_xy,
)


def test_count_policy_and_appearance_streams_are_independent():
    for seed in range(100):
        plan = resolve_episode_plan(seed=seed)
        expected_k = int(np.random.default_rng(np.random.SeedSequence([seed, K_SALT])).choice((1, 2, 3)))
        expected_n = int(np.random.default_rng(np.random.SeedSequence([seed, N_SALT])).integers(expected_k, 6))
        assert (plan.target_sugar_count, plan.total_sugar_count) == (expected_k, expected_n)
        original = resolve_episode_plan(
            seed=seed, saucepan_material_candidates=("original",),
            sugar_plate_material_candidates=("original",), sugar_cube_material_candidates=("original",),
        )
        for key, value in asdict(plan).items():
            if key not in {"saucepan_material", "sugar_plate_material", "sugar_cube_material"}:
                assert value == asdict(original)[key]


@pytest.mark.parametrize("count", range(1, 6))
def test_all_tray_counts_fit_real_scene_bounds(count):
    safe = np.array((0.11564878202215918, 0.06039817852215919))
    centers = GRASP_SAFE_SLOTS * safe
    for seed in range(200):
        offsets = np.array(sample_sugar_plate_offsets_for_seed(count=count, seed=seed, safe_half_extents_xy=safe))
        validate_sugar_plate_offsets_xy(offsets, count=count, safe_half_extents_xy=safe)
        assert np.all(np.min(np.linalg.norm(offsets[:, None, :] - centers[None, :, :], axis=2), axis=1) <= 0.005 + 1e-9)
        assert len(sample_sugar_cube_yaws_for_seed(count=count, seed=seed)) == count


def test_production_config_enables_station_navigation_and_alignment():
    config = load_task_config()
    plan = resolve_episode_plan(seed=11, target_sugar_count=3, total_sugar_count=5, **{
        key: value for key, value in config.episode.items()
        if key not in {"target_sugar_count", "total_sugar_count"}
    })
    actions = resolve_action_config(episode=plan, operations=config.operations)
    assert config.navigation["use_station_base_poses"]
    assert not config.navigation["manual_base_path_enabled"]
    assert config.path(config.resources["station_poses"]).is_file()
    assert actions.sugar_pick_metadata["align_preshape_finger_center"]
    assert len(actions.sugar_delivery_metadata_by_index) == 3
    assert plan.saucepan_type == "Saucepan002"


def test_k_above_three_is_rejected():
    with pytest.raises(ValueError):
        resolve_episode_plan(seed=0, target_sugar_count=4, total_sugar_count=5)
