from __future__ import annotations

import numpy as np

from bench.evaluation.expert_prefix import (
    ExpertPrefixArtifact,
    load_expert_prefix,
    write_expert_prefix,
)


def test_expert_prefix_round_trip_and_target_slicing(tmp_path) -> None:
    actions = np.arange(5 * 13, dtype=np.float32).reshape(5, 13)
    artifact = ExpertPrefixArtifact(
        task_id="WA-05",
        seed=1050,
        episode_identity="episode",
        expert_plan_version="1",
        stage_boundary_steps={"cup_filled": 2, "cup_staged": 4, "success": 5},
        actions=actions,
        success=True,
    )
    write_expert_prefix(tmp_path, artifact)
    loaded = load_expert_prefix(tmp_path, 1050)
    np.testing.assert_array_equal(loaded.actions, actions)
    np.testing.assert_array_equal(
        loaded.prefix_for_target(("cup_filled", "cup_staged", "success"), "success"),
        actions[:4],
    )


def test_expert_prefix_rejects_non_monotonic_boundaries() -> None:
    actions = np.zeros((5, 13), dtype=np.float32)
    try:
        ExpertPrefixArtifact(
            task_id="WA-05",
            seed=1050,
            episode_identity="episode",
            expert_plan_version="1",
            stage_boundary_steps={"cup_filled": 4, "cup_staged": 2},
            actions=actions,
            success=False,
        )
    except ValueError as exc:
        assert "monotonic" in str(exc)
    else:
        raise AssertionError("non-monotonic boundaries must be rejected")
