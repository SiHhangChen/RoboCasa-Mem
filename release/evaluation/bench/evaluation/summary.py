"""Aggregation of independently committed episode results."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from bench.evaluation.report_stages import (
    REPORT_STAGE_SUMMARY_SCHEMA,
    validate_report_stages,
)


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def aggregate_episode_results(
    results: Iterable[Mapping[str, Any]],
    expected_seeds: Iterable[int] | None = None,
) -> dict[str, Any]:
    items = list(results)
    duplicate_seeds: list[Any]
    if expected_seeds is None:
        aggregate_items = items
        planned = len(items)
        recorded = len(items)
        pending = 0
        unexpected = 0
        duplicate_seeds = _duplicate_result_seeds(items)
    else:
        expected = _unique_expected_seeds(expected_seeds)
        aggregate_items, unexpected, duplicate_seeds = _select_planned_results(
            items,
            expected,
        )
        planned = len(expected)
        recorded = len(aggregate_items)
        pending = planned - recorded

    counts = {"success": 0, "failure": 0, "error": 0, "cancelled": 0}
    failure_reasons: dict[str, int] = {}
    success_steps: list[int] = []
    latencies: list[float] = []
    for result in aggregate_items:
        status = str(result.get("status", "error"))
        counts[status] = counts.get(status, 0) + 1
        if status == "failure":
            reason = str(result.get("failure_reason", "unknown"))
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        if status == "success" and result.get("success_step") is not None:
            success_steps.append(int(result["success_step"]))
        latencies.extend(float(item) for item in result.get("inference_latency_ms", []))
    completed = counts["success"] + counts["failure"]
    interval = wilson_interval(counts["success"], completed)
    task_success_rate = counts["success"] / completed if completed else None
    completion_rate = recorded / planned if planned else None
    overall_success_rate = counts["success"] / planned if planned else None
    publishable = bool(
        expected_seeds is not None
        and planned > 0
        and pending == 0
        and unexpected == 0
        and not duplicate_seeds
        and completed == recorded
    )
    report_stage_summary = _aggregate_report_stages(aggregate_items)
    subtask_summary = _aggregate_subtask_schedulers(aggregate_items)
    if (
        report_stage_summary is not None
        and not report_stage_summary["complete"]
    ):
        publishable = False
    summary = {
        "episodes_total": len(aggregate_items),
        "counts": counts,
        "completed_task_episodes": completed,
        "success_rate": task_success_rate,
        "success_rate_ci95": list(interval) if completed else None,
        "success_step_mean": (
            sum(success_steps) / len(success_steps) if success_steps else None
        ),
        "inference_latency_ms_mean": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "planned": planned,
        "recorded": recorded,
        "pending": pending,
        "unexpected": unexpected,
        "completion_rate": completion_rate,
        "task_success_rate": task_success_rate,
        "overall_success_rate": overall_success_rate,
        "publishable": publishable,
        "duplicate_seeds": duplicate_seeds,
    }
    if report_stage_summary is not None:
        summary["report_stages"] = report_stage_summary
    if subtask_summary is not None:
        summary["subtask_scheduler"] = subtask_summary
    return summary


def _aggregate_report_stages(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    eligible = [
        result
        for result in results
        if result.get("status") in {"success", "failure"}
    ]
    reported: list[dict[str, Any]] = []
    for result in eligible:
        raw = result.get("report_stages")
        if raw is None:
            continue
        task_id = result.get("task_id")
        reported.append(
            validate_report_stages(
                raw,
                expected_task_id=(
                    str(task_id) if isinstance(task_id, str) else None
                ),
                expected_success=bool(result.get("success")),
            )
        )
    if not reported:
        return None

    first = reported[0]
    identity = (
        first["schema_version"],
        first["task_id"],
        first["profile_id"],
        tuple(first["order"]),
    )
    for report in reported[1:]:
        candidate = (
            report["schema_version"],
            report["task_id"],
            report["profile_id"],
            tuple(report["order"]),
        )
        if candidate != identity:
            raise ValueError(
                "cannot aggregate incompatible report-stage profiles: "
                f"{identity!r} != {candidate!r}"
            )

    stages: dict[str, dict[str, Any]] = {}
    for stage_id in first["order"]:
        verdicts = [report["stages"][stage_id] for report in reported]
        denominator = sum(int(verdict["applicable"]) for verdict in verdicts)
        passed = sum(
            int(verdict["passed"] is True)
            for verdict in verdicts
            if verdict["applicable"]
        )
        steps = [
            int(verdict["first_pass_step"])
            for verdict in verdicts
            if verdict["applicable"] and verdict["passed"] is True
        ]
        stages[stage_id] = {
            "passed": passed,
            "denominator": denominator,
            "rate": passed / denominator if denominator else None,
            "not_applicable": len(reported) - denominator,
            "first_pass_step_mean": (
                sum(steps) / len(steps) if steps else None
            ),
        }

    return {
        "schema_version": REPORT_STAGE_SUMMARY_SCHEMA,
        "task_id": first["task_id"],
        "profile_id": first["profile_id"],
        "order": list(first["order"]),
        "eligible_episodes": len(eligible),
        "reported_episodes": len(reported),
        "missing_episodes": len(eligible) - len(reported),
        "complete": len(eligible) == len(reported),
        "stages": stages,
    }


def _aggregate_subtask_schedulers(
    results: Iterable[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate optional oracle prompt-scheduler stage handoffs."""

    eligible = [
        result
        for result in results
        if result.get("status") in {"success", "failure"}
        and isinstance(result.get("episode_metadata"), Mapping)
        and isinstance(
            result["episode_metadata"].get("subtask_scheduler"), Mapping
        )
    ]
    if not eligible:
        return None

    scheduler_records = [
        result["episode_metadata"]["subtask_scheduler"] for result in eligible
    ]
    schedule_ids = {str(record.get("schedule_id", "unknown")) for record in scheduler_records}
    if len(schedule_ids) != 1:
        raise ValueError(
            "cannot aggregate incompatible subtask schedulers: "
            f"{sorted(schedule_ids)}"
        )
    orders = {
        tuple(str(item) for item in record.get("subtasks", ()))
        for record in scheduler_records
    }
    if len(orders) != 1 or not next(iter(orders), ()):
        raise ValueError("subtask scheduler records must declare one common non-empty order")
    order = next(iter(orders))
    stages: dict[str, dict[str, Any]] = {}
    for index, stage_id in enumerate(order):
        attempted = completed = 0
        completion_steps: list[int] = []
        for record in scheduler_records:
            current = str(record.get("current_subtask", ""))
            try:
                current_index = order.index(current)
            except ValueError:
                continue
            if current_index < index:
                continue
            attempted += 1
            transitions = record.get("transitions", ())
            stage_completed = False
            for transition in transitions:
                if not isinstance(transition, Mapping):
                    continue
                if transition.get("changed") and transition.get("previous_subtask") == stage_id:
                    stage_completed = True
                if transition.get("complete") and transition.get("current_subtask") == stage_id:
                    stage_completed = True
                if stage_completed and transition.get("environment_step") is not None:
                    completion_steps.append(int(transition["environment_step"]))
                    break
            if stage_completed:
                completed += 1
        stages[stage_id] = {
            "attempted": attempted,
            "completed": completed,
            "completion_rate": completed / attempted if attempted else None,
            "completion_step_mean": (
                sum(completion_steps) / len(completion_steps)
                if completion_steps
                else None
            ),
        }
    return {
        "schema_version": "membench.subtask_scheduler_summary/v1",
        "schedule_id": next(iter(schedule_ids)),
        "order": list(order),
        "eligible_episodes": len(eligible),
        "completed_episodes": sum(
            bool(record.get("finished", False)) for record in scheduler_records
        ),
        "failed_episodes": sum(
            bool(record.get("failed", False)) for record in scheduler_records
        ),
        "stages": stages,
    }


def _unique_expected_seeds(expected_seeds: Iterable[int]) -> tuple[int, ...]:
    unique: list[int] = []
    seen: set[tuple[type[Any], Any]] = set()
    for seed in expected_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Expected seed must be an integer: {seed!r}")
        identity = _seed_identity(seed)
        if identity not in seen:
            seen.add(identity)
            unique.append(seed)
    return tuple(unique)


def _select_planned_results(
    items: list[Mapping[str, Any]],
    expected_seeds: tuple[int, ...],
) -> tuple[list[Mapping[str, Any]], int, list[Any]]:
    expected = {_seed_identity(seed) for seed in expected_seeds}
    selected: list[Mapping[str, Any]] = []
    seen: set[tuple[type[Any], Any]] = set()
    unexpected: set[tuple[type[Any], Any]] = set()
    malformed_seed_results = 0
    duplicates: dict[tuple[type[Any], Any], Any] = {}
    for result in items:
        seed = result.get("seed", _MISSING)
        identity = _hashable_seed_identity(seed)
        if identity is None:
            malformed_seed_results += 1
            continue
        if identity in seen:
            duplicates.setdefault(identity, seed)
            continue
        seen.add(identity)
        if identity in expected:
            selected.append(result)
        else:
            unexpected.add(identity)
    return (
        selected,
        len(unexpected) + malformed_seed_results,
        _sorted_seed_values(duplicates.values()),
    )


def _duplicate_result_seeds(items: list[Mapping[str, Any]]) -> list[Any]:
    seen: set[tuple[type[Any], Any]] = set()
    duplicates: dict[tuple[type[Any], Any], Any] = {}
    for result in items:
        seed = result.get("seed", _MISSING)
        identity = _hashable_seed_identity(seed)
        if identity is None:
            continue
        if identity in seen:
            duplicates.setdefault(identity, seed)
        else:
            seen.add(identity)
    return _sorted_seed_values(duplicates.values())


def _hashable_seed_identity(value: Any) -> tuple[type[Any], Any] | None:
    if value is _MISSING:
        return None
    try:
        hash(value)
    except TypeError:
        return None
    return _seed_identity(value)


def _seed_identity(value: Any) -> tuple[type[Any], Any]:
    return (type(value), value)


def _sorted_seed_values(values: Iterable[Any]) -> list[Any]:
    items = list(values)
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=lambda value: (type(value).__name__, repr(value)))


_MISSING = object()
