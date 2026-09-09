"""Statistics for frozen expert-prefix action trajectories."""

from __future__ import annotations

from collections.abc import Mapping
import math
from statistics import mean, median
from typing import Any, Sequence

from bench.evaluation.expert_prefix import ExpertPrefixArtifact


def summarize_expert_prefixes(
    artifacts: Sequence[ExpertPrefixArtifact],
    *,
    stage_order: Sequence[str],
    stage_segments: Mapping[str, tuple[str | None, str]] | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarize valid prefix lengths and propose conservative stage budgets."""

    stages = tuple(str(value) for value in stage_order)
    if not stages or len(stages) != len(set(stages)):
        raise ValueError("stage_order must contain unique stage ids")
    segments = _stage_segments(stages, stage_segments)
    items = tuple(artifacts)
    artifact_seeds = [artifact.seed for artifact in items]
    if len(artifact_seeds) != len(set(artifact_seeds)):
        raise ValueError("expert prefix artifacts must have unique seeds")
    expected = (
        tuple(artifact_seeds)
        if expected_seeds is None
        else tuple(int(seed) for seed in expected_seeds)
    )
    if len(expected) != len(set(expected)):
        raise ValueError("expected_seeds must be unique")
    expected_set = set(expected)
    unexpected = sorted(set(artifact_seeds) - expected_set)
    missing = sorted(expected_set - set(artifact_seeds))
    invalid = sorted(
        artifact.seed
        for artifact in items
        if artifact.seed in expected_set and not artifact.success
    )
    valid = [
        artifact
        for artifact in items
        if artifact.seed in expected_set and artifact.success
    ]

    lengths: dict[str, list[int]] = {stage_id: [] for stage_id in stages}
    for artifact in valid:
        for stage_id in stages:
            start_stage, end_stage = segments[stage_id]
            try:
                start = (
                    0
                    if start_stage is None
                    else int(artifact.stage_boundary_steps[start_stage])
                )
                end = int(artifact.stage_boundary_steps[end_stage])
            except KeyError as exc:
                raise ValueError(
                    f"seed {artifact.seed} is missing a boundary for stage {stage_id!r}"
                ) from exc
            if start < 0 or end <= start or end > artifact.actions.shape[0]:
                raise ValueError(
                    f"seed {artifact.seed} has invalid segment "
                    f"{stage_id}=({start_stage!r}:{start}, {end_stage!r}:{end})"
                )
            lengths[stage_id].append(end - start)

    stage_payloads = {
        stage_id: _length_statistics(values)
        for stage_id, values in lengths.items()
    }
    attempted = len(expected)
    valid_count = len(valid)
    ready = (
        attempted > 0
        and valid_count == attempted
        and not invalid
        and not missing
        and not unexpected
    )
    return {
        "schema_version": "membench.expert_prefix_summary/v1",
        "stage_order": list(stages),
        "stage_segments": {
            stage_id: {
                "start_boundary": start,
                "end_boundary": end,
            }
            for stage_id, (start, end) in segments.items()
        },
        "attempted": attempted,
        "recorded": len(items),
        "valid": valid_count,
        "invalid_seeds": invalid,
        "missing_seeds": missing,
        "unexpected_seeds": unexpected,
        "prefix_validity": None if attempted == 0 else valid_count / attempted,
        "ready_for_budget_freeze": ready,
        "stages": stage_payloads,
        "recommended_budgets": {
            stage_id: payload["recommended_budget_2x_p95_higher"]
            for stage_id, payload in stage_payloads.items()
        },
        "provenance": {
            "task_ids": sorted({artifact.task_id for artifact in items}),
            "expert_plan_versions": sorted(
                {artifact.expert_plan_version for artifact in items}
            ),
            "git_revisions": sorted(
                {
                    str(artifact.provenance.get("git_revision"))
                    for artifact in items
                }
            ),
        },
    }


def _stage_segments(
    stages: tuple[str, ...],
    configured: Mapping[str, tuple[str | None, str]] | None,
) -> dict[str, tuple[str | None, str]]:
    if configured is None:
        return {
            stage_id: (None if index == 0 else stages[index - 1], stage_id)
            for index, stage_id in enumerate(stages)
        }
    if set(configured) != set(stages):
        raise ValueError("stage_segments keys must match stage_order")
    segments: dict[str, tuple[str | None, str]] = {}
    for stage_id in stages:
        value = configured[stage_id]
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("each stage segment must be a (start, end) tuple")
        start, end = value
        if start is not None and not str(start):
            raise ValueError("stage segment start must be non-empty or None")
        if not str(end):
            raise ValueError("stage segment end must be non-empty")
        segments[stage_id] = (
            None if start is None else str(start),
            str(end),
        )
    return segments


def _length_statistics(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p95_higher": None,
            "max": None,
            "recommended_budget_2x_p95_higher": None,
        }
    ordered = sorted(int(value) for value in values)
    p95 = ordered[math.ceil(0.95 * (len(ordered) - 1))]
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": float(median(ordered)),
        "mean": float(mean(ordered)),
        "p95_higher": p95,
        "max": ordered[-1],
        "recommended_budget_2x_p95_higher": 2 * p95,
    }


__all__ = ["summarize_expert_prefixes"]
