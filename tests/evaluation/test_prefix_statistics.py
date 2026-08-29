from __future__ import annotations

import numpy as np

from bench.evaluation.expert_prefix import ExpertPrefixArtifact
from bench.evaluation.prefix_statistics import summarize_expert_prefixes


def _artifact(seed: int, a: int, b: int, c: int) -> ExpertPrefixArtifact:
    return ExpertPrefixArtifact(
        task_id="WA-05",
        seed=seed,
        episode_identity=f"episode-{seed}",
        expert_plan_version="expert-v1",
        stage_boundary_steps={"A": a, "B": a + b, "C": a + b + c},
        actions=np.zeros((a + b + c + 1, 13), dtype=np.float32),
        success=True,
        provenance={"git_revision": "revision"},
    )


def test_prefix_summary_computes_stage_lengths_and_conservative_p95() -> None:
    artifacts = [
        _artifact(seed, a=10 + seed, b=20 + seed, c=30 + seed)
        for seed in range(1, 21)
    ]
    summary = summarize_expert_prefixes(
        artifacts,
        stage_order=("A", "B", "C"),
        expected_seeds=range(1, 21),
    )

    assert summary["ready_for_budget_freeze"] is True
    assert summary["prefix_validity"] == 1.0
    assert summary["stages"]["A"]["p95_higher"] == 30
    assert summary["stages"]["B"]["p95_higher"] == 40
    assert summary["stages"]["C"]["p95_higher"] == 50
    assert summary["recommended_budgets"] == {"A": 60, "B": 80, "C": 100}


def test_prefix_summary_reports_missing_and_invalid_seeds() -> None:
    valid = _artifact(1, 10, 20, 30)
    invalid = _artifact(2, 10, 20, 30)
    invalid = ExpertPrefixArtifact(
        task_id=invalid.task_id,
        seed=invalid.seed,
        episode_identity=invalid.episode_identity,
        expert_plan_version=invalid.expert_plan_version,
        stage_boundary_steps=invalid.stage_boundary_steps,
        actions=invalid.actions,
        success=False,
        failure_reason="expert_failed",
        provenance=invalid.provenance,
    )

    summary = summarize_expert_prefixes(
        (valid, invalid),
        stage_order=("A", "B", "C"),
        expected_seeds=(1, 2, 3),
    )

    assert summary["ready_for_budget_freeze"] is False
    assert summary["prefix_validity"] == 1 / 3
    assert summary["invalid_seeds"] == [2]
    assert summary["missing_seeds"] == [3]
