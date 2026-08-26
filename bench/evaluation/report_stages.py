"""Public benchmark report-stage contract.

Task-local evaluators may expose richer internal stages through diagnostics.
This module validates the smaller, stable stage funnel persisted for benchmark
reporting and cross-task aggregation.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from bench.evaluation.report_profiles import get_report_stage_profile


REPORT_STAGES_SCHEMA = "membench.report_stages/v1"
REPORT_STAGE_SUMMARY_SCHEMA = "membench.report_stage_summary/v1"

_STAGE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_REPORT_KEYS = frozenset(
    {"schema_version", "task_id", "profile_id", "order", "stages"}
)
_VERDICT_KEYS = frozenset({"applicable", "passed", "first_pass_step"})


def validate_report_stages(
    payload: Mapping[str, Any],
    *,
    expected_task_id: str | None = None,
    expected_profile_id: str | None = None,
    expected_success: bool | None = None,
) -> dict[str, Any]:
    """Validate and normalize one episode's public report-stage verdicts."""

    if not isinstance(payload, Mapping):
        raise ValueError("report_stages must be a mapping")
    unknown = sorted(set(payload) - _REPORT_KEYS)
    missing = sorted(_REPORT_KEYS - set(payload))
    if unknown:
        raise ValueError(f"report_stages has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"report_stages is missing fields: {missing}")
    if payload.get("schema_version") != REPORT_STAGES_SCHEMA:
        raise ValueError(
            "unsupported report_stages schema: "
            f"{payload.get('schema_version')!r}"
        )

    task_id = _nonempty_string(payload.get("task_id"), "report_stages.task_id")
    profile_id = _nonempty_string(
        payload.get("profile_id"), "report_stages.profile_id"
    )
    if expected_task_id is not None and task_id != str(expected_task_id):
        raise ValueError(
            f"report_stages task mismatch: expected {expected_task_id!r}, "
            f"got {task_id!r}"
        )
    if expected_profile_id is not None and profile_id != str(expected_profile_id):
        raise ValueError(
            f"report_stages profile mismatch: expected {expected_profile_id!r}, "
            f"got {profile_id!r}"
        )

    try:
        profile = get_report_stage_profile(task_id, profile_id=profile_id)
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark report profile: {exc}") from exc

    raw_order = payload.get("order")
    if isinstance(raw_order, (str, bytes)) or not isinstance(raw_order, (list, tuple)):
        raise ValueError("report_stages.order must be a list of stage ids")
    order = [_stage_id(value) for value in raw_order]
    if not order:
        raise ValueError("report_stages.order must not be empty")
    if len(set(order)) != len(order):
        raise ValueError("report_stages.order must contain unique stage ids")
    if tuple(order) != profile.stage_ids:
        raise ValueError(
            f"report_stages.order does not match {profile.profile_id!r}: "
            f"expected {list(profile.stage_ids)!r}, got {order!r}"
        )

    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, Mapping):
        raise ValueError("report_stages.stages must be a mapping")
    if set(raw_stages) != set(order):
        raise ValueError(
            "report_stages.stages keys must exactly match report_stages.order"
        )

    stages: dict[str, dict[str, Any]] = {}
    prior_applicable_failed = False
    for stage_id in order:
        raw_verdict = raw_stages[stage_id]
        if not isinstance(raw_verdict, Mapping):
            raise ValueError(f"report stage {stage_id!r} must be a mapping")
        unknown_verdict = sorted(set(raw_verdict) - _VERDICT_KEYS)
        missing_verdict = sorted(_VERDICT_KEYS - set(raw_verdict))
        if unknown_verdict:
            raise ValueError(
                f"report stage {stage_id!r} has unknown fields: {unknown_verdict}"
            )
        if missing_verdict:
            raise ValueError(
                f"report stage {stage_id!r} is missing fields: {missing_verdict}"
            )

        applicable = raw_verdict.get("applicable")
        passed = raw_verdict.get("passed")
        first_pass_step = raw_verdict.get("first_pass_step")
        if not isinstance(applicable, bool):
            raise ValueError(
                f"report stage {stage_id!r}.applicable must be a boolean"
            )
        if applicable:
            if not isinstance(passed, bool):
                raise ValueError(
                    f"applicable report stage {stage_id!r}.passed must be a boolean"
                )
            if passed:
                if (
                    isinstance(first_pass_step, bool)
                    or not isinstance(first_pass_step, int)
                    or first_pass_step <= 0
                ):
                    raise ValueError(
                        f"passed report stage {stage_id!r} must have a positive "
                        "first_pass_step"
                    )
                if prior_applicable_failed:
                    raise ValueError(
                        f"report stage {stage_id!r} passed after an applicable "
                        "predecessor failed"
                    )
            elif first_pass_step is not None:
                raise ValueError(
                    f"failed report stage {stage_id!r} must not have first_pass_step"
                )
            prior_applicable_failed = prior_applicable_failed or not passed
        else:
            if stage_id not in profile.conditional_stage_ids:
                raise ValueError(
                    f"report stage {stage_id!r} is not conditional in "
                    f"profile {profile.profile_id!r}"
                )
            if passed is not None or first_pass_step is not None:
                raise ValueError(
                    f"non-applicable report stage {stage_id!r} must have "
                    "passed=null and first_pass_step=null"
                )

        stages[stage_id] = {
            "applicable": applicable,
            "passed": passed,
            "first_pass_step": first_pass_step,
        }

    success = stages[profile.success_stage_id]
    if not success["applicable"]:
        raise ValueError("report profile success stage must be applicable")
    if expected_success is not None and success["passed"] is not expected_success:
        raise ValueError(
            "report success stage is inconsistent with episode success: "
            f"expected {expected_success}, got {success['passed']}"
        )

    return {
        "schema_version": REPORT_STAGES_SCHEMA,
        "task_id": task_id,
        "profile_id": profile_id,
        "order": order,
        "stages": stages,
    }


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _stage_id(value: Any) -> str:
    if not isinstance(value, str) or not _STAGE_ID.fullmatch(value):
        raise ValueError(f"invalid report stage id: {value!r}")
    return value


__all__ = [
    "REPORT_STAGES_SCHEMA",
    "REPORT_STAGE_SUMMARY_SCHEMA",
    "validate_report_stages",
]
