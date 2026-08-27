"""Shared public report projection for drawer-search benchmark tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bench.evaluation.report_profiles import ReportStageProfile


DRAWER_SEARCH_STAGE_IDS = (
    "one_drawer_checked",
    "target_found",
    "search_stopped_after_target",
    "button_pressed",
    "correct_press_count",
)


def build_drawer_search_report(
    *,
    profile: ReportStageProfile,
    completed_fixture_history: Sequence[str],
    completed_fixture_steps: Mapping[str, int],
    target_fixture: str,
    target_found: bool,
    search_after_target_detected: bool,
    valid_button_press_steps: Sequence[int],
    success: bool,
) -> dict[str, Any]:
    """Build the monotonic final funnel from task-owned trajectory facts."""

    if profile.stage_ids != DRAWER_SEARCH_STAGE_IDS:
        raise ValueError(
            f"profile {profile.profile_id!r} is not a drawer-search profile"
        )
    history = [str(value) for value in completed_fixture_history]
    step_by_fixture = {
        str(name): int(step) for name, step in completed_fixture_steps.items()
    }
    first_fixture = history[0] if history else None
    one_checked = bool(first_fixture is not None)
    found = bool(
        one_checked
        and target_found
        and str(target_fixture) in history
    )
    stopped = bool(
        found
        and history[-1] == str(target_fixture)
        and not search_after_target_detected
    )
    button_steps = [int(step) for step in valid_button_press_steps]
    button_pressed = bool(stopped and button_steps)
    correct_press_count = bool(button_pressed and success)
    stages = {
        "one_drawer_checked": _verdict(
            one_checked,
            step_by_fixture.get(first_fixture) if first_fixture is not None else None,
        ),
        "target_found": _verdict(
            found,
            step_by_fixture.get(str(target_fixture)),
        ),
        "search_stopped_after_target": _verdict(
            stopped,
            step_by_fixture.get(str(target_fixture)),
        ),
        "button_pressed": _verdict(
            button_pressed,
            button_steps[0] if button_steps else None,
        ),
        "correct_press_count": _verdict(
            correct_press_count,
            button_steps[-1] if button_steps else None,
        ),
    }
    return {
        "schema_version": "membench.report_stages/v1",
        "task_id": profile.task_id,
        "profile_id": profile.profile_id,
        "order": list(profile.stage_ids),
        "stages": stages,
    }


def _verdict(passed: bool, step: Any) -> dict[str, Any]:
    return {
        "applicable": True,
        "passed": bool(passed),
        "first_pass_step": int(step) if passed and step is not None else None,
    }


__all__ = ["DRAWER_SEARCH_STAGE_IDS", "build_drawer_search_report"]
