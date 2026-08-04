from __future__ import annotations

from dataclasses import asdict

import pytest

from bench.tasks.ts01.domain.episode import (
    resolve_episode_plan as resolve_ts01_episode_plan,
)
from bench.tasks.ts03.config import load_task_config, resolve_action_config
from bench.tasks.ts03.domain.contracts import profile_for_seed
from bench.tasks.ts03.domain.episode import resolve_episode_plan
from bench.tasks.ts03.domain.geometry import SUGAR_RELEASE_REGION_NAMES
from bench.tasks.ts03.execution.geometry import build_saucepan_release_candidate_groups


def test_ts03_seeded_sampling_matches_ts01_physical_policy() -> None:
    for seed in range(32):
        profile = profile_for_seed(seed)
        actual = resolve_episode_plan(seed=seed)
        expected = resolve_ts01_episode_plan(
            seed=seed,
            target_sugar_count=profile.target_sugar_count,
            total_sugar_count=profile.total_sugar_count,
            max_total_sugar_count=3,
            max_horizon=2800,
        )

        assert asdict(actual) == asdict(expected)


def test_ts03_release_orders_are_seeded_permutations_accepted_by_execution() -> None:
    config = load_task_config()
    observed_orders: set[tuple[str, ...]] = set()

    for seed in range(16):
        episode = resolve_episode_plan(seed=seed)
        actions = resolve_action_config(episode=episode, operations=config.operations)

        for metadata in actions.sugar_delivery_metadata_by_index.values():
            order = tuple(metadata["release_region_order"])
            observed_orders.add(order)
            assert set(order) == set(SUGAR_RELEASE_REGION_NAMES)
            groups = build_saucepan_release_candidate_groups(
                (0.0, 0.0, 0.0),
                region_order=order,
                release_regions_xy=metadata["release_regions_xy"],
                yaw_offsets_deg=metadata["placement_yaw_offsets_deg"],
                max_radius_m=metadata["release_target_max_radius_m"],
            )
            assert len(groups) == len(SUGAR_RELEASE_REGION_NAMES)

    assert len(observed_orders) > 1


def test_ts03_explicit_target_order_and_seed_validation_match_ts01() -> None:
    actual = resolve_episode_plan(
        seed=7,
        profile_id="k2_n3",
        target_sugar_names="sugar2;sugar0",
    )
    expected = resolve_ts01_episode_plan(
        seed=7,
        target_sugar_count=2,
        total_sugar_count=3,
        max_total_sugar_count=3,
        target_sugar_names="sugar2;sugar0",
        max_horizon=2800,
    )
    assert asdict(actual) == asdict(expected)

    with pytest.raises(ValueError, match="seed must be non-negative"):
        resolve_episode_plan(seed=-1, profile_id="k1_n2")
