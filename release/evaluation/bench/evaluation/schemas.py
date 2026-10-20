"""Versioned validation for persisted evaluation records.

The on-disk JSON files are a public benchmark boundary.  Keep validation here
so resume, verification, and aggregation agree on what a completed episode is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.evaluation.report_stages import validate_report_stages


EPISODE_RESULT_SCHEMA = "membench.episode_result/v1"
VALID_EPISODE_STATUSES = frozenset({"success", "failure", "error", "cancelled"})


def validate_episode_result(
    payload: Mapping[str, Any],
    *,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    """Return a normalized episode result or raise ``ValueError``.

    Records without ``schema_version`` are accepted as previous v0 inputs so an
    interrupted run created by the first evaluation implementation remains
    resumable.  Every newly written result is stamped as v1.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("episode result must be a mapping")
    result = dict(payload)
    schema = result.get("schema_version")
    if schema not in {None, EPISODE_RESULT_SCHEMA}:
        raise ValueError(f"unsupported episode result schema: {schema!r}")

    seed = _integer(result.get("seed"), "seed")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError(f"episode seed mismatch: expected {expected_seed}, got {seed}")

    status = result.get("status")
    if not isinstance(status, str) or status not in VALID_EPISODE_STATUSES:
        raise ValueError(
            "episode status must be one of " + ", ".join(sorted(VALID_EPISODE_STATUSES))
        )
    success = result.get("success")
    if not isinstance(success, bool):
        raise ValueError("episode success must be a boolean")
    if success != (status == "success"):
        raise ValueError(f"episode status={status!r} is inconsistent with success={success}")

    success_step = result.get("success_step")
    if success_step is not None:
        success_step = _integer(success_step, "success_step")
        if success_step <= 0:
            raise ValueError("success_step must be positive when present")
    if status == "success" and success_step is None:
        raise ValueError("successful episode must include success_step")

    for key in ("env_steps", "policy_calls"):
        if key in result and _integer(result[key], key) < 0:
            raise ValueError(f"{key} must be non-negative")
    for key in (
        "reward_sum",
        "duration_s",
        "policy_actions_returned",
        "policy_actions_consumed",
    ):
        if key in result:
            value = result[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric")
            if not np.isfinite(float(value)) or float(value) < 0 and key != "reward_sum":
                raise ValueError(f"{key} must be finite and non-negative")

    if status == "error" and not _nonempty_string(result.get("error")):
        if schema is None:
            result["error"] = "legacy_unspecified_error"
        else:
            raise ValueError("error episode must include a non-empty error")
    if status == "failure" and not _nonempty_string(result.get("failure_reason")):
        if schema is None:
            result["failure_reason"] = "unknown"
        else:
            raise ValueError("failure episode must include a non-empty failure_reason")

    report_stages = result.get("report_stages")
    if report_stages is not None:
        task_id = result.get("task_id")
        if not _nonempty_string(task_id):
            raise ValueError(
                "episode task_id must be present when report_stages is present"
            )
        result["report_stages"] = validate_report_stages(
            report_stages,
            expected_task_id=str(task_id),
            expected_success=success,
        )

    result["schema_version"] = EPISODE_RESULT_SCHEMA
    result["seed"] = seed
    result["status"] = status
    result["success"] = success
    result["success_step"] = success_step
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"episode {name} must be an integer")
    return int(value)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "EPISODE_RESULT_SCHEMA",
    "VALID_EPISODE_STATUSES",
    "validate_episode_result",
]
