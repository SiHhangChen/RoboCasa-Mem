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
    assert loaded.stage_order == ("cup_filled", "cup_staged", "success")
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


def test_expert_prefix_round_trip_preserves_explicit_stage_order(tmp_path) -> None:
    artifact = ExpertPrefixArtifact(
        task_id="WA-01",
        seed=1050,
        episode_identity="episode",
        expert_plan_version="1",
        stage_boundary_steps={
            "sugar": 2,
            "one_ready": 4,
            "ready": 6,
            "selection": 8,
        },
        actions=np.zeros((9, 13), dtype=np.float32),
        success=True,
        stage_order=("sugar", "one_ready", "ready", "selection"),
    )

    write_expert_prefix(tmp_path, artifact)
    loaded = load_expert_prefix(tmp_path, artifact.seed)
    assert loaded.stage_order == artifact.stage_order
    assert loaded.stage_boundary_steps == artifact.stage_boundary_steps


def test_invalid_prefix_can_record_only_the_boundaries_it_reached() -> None:
    artifact = ExpertPrefixArtifact(
        task_id="WA-01",
        seed=1050,
        episode_identity="episode",
        expert_plan_version="1",
        stage_boundary_steps={"sugar": 2},
        actions=np.zeros((3, 13), dtype=np.float32),
        success=False,
        stage_order=("sugar",),
        failure_reason="expert_plan_failed",
    )

    assert artifact.stage_order == ("sugar",)


def test_expert_prefix_allows_an_initial_state_boundary() -> None:
    artifact = ExpertPrefixArtifact(
        task_id="TS02",
        seed=1050,
        episode_identity="episode",
        expert_plan_version="1",
        stage_boundary_steps={"before_target_drawer": 0, "target_found": 3},
        actions=np.zeros((5, 13), dtype=np.float32),
        success=True,
    )

    np.testing.assert_array_equal(
        artifact.actions[: artifact.stage_boundary_steps["before_target_drawer"]],
        np.empty((0, 13), dtype=np.float32),
    )
