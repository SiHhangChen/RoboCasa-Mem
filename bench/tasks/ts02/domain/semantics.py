"""Authoritative, simulator-free TS02 search-and-count state machine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.ts02.domain.contracts import (
    AXIS,
    FIXTURE_DISPLAY_NAMES,
    FIXTURE_NAMES,
)


@dataclass
class SemanticState:
    """History that cannot be recovered from the final closed-drawer image."""

    fixture_names: tuple[str, ...] = FIXTURE_NAMES
    target_fixture: str = FIXTURE_NAMES[0]
    checked_fixture_set: set[str] = field(default_factory=set)
    completed_fixture_history: list[str] = field(default_factory=list)
    open_attempt_history: list[str] = field(default_factory=list)
    inspect_history: list[str] = field(default_factory=list)
    close_attempt_history: list[str] = field(default_factory=list)
    active_fixture: str | None = None
    inspected_active_fixture: bool = False
    target_observed: bool = False
    target_found: bool = False
    duplicate_fixture_check: bool = False
    search_after_target_detected: bool = False
    invalid_open_detected: bool = False
    invalid_inspect_detected: bool = False
    invalid_close_detected: bool = False
    close_before_inspect_detected: bool = False
    completion_button_pressed: bool = False
    completion_button_attempted: bool = False
    completion_button_attempt_count: int = 0
    completion_button_press_count: int = 0
    invalid_completion_button_press: bool = False
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        self.fixture_names = tuple(str(name) for name in self.fixture_names)
        self.target_fixture = str(self.target_fixture)
        if len(self.fixture_names) < 2 or len(set(self.fixture_names)) != len(
            self.fixture_names
        ):
            raise ValueError("inspection semantic fixtures must contain unique names")
        if self.target_fixture not in self.fixture_names:
            raise ValueError("target_fixture must name one of the inspection fixtures")

    @property
    def num_checked(self) -> int:
        return len(self.checked_fixture_set)

    @property
    def required_button_presses(self) -> int:
        """The search length k, defined only after the target drawer is closed."""

        return len(self.completed_fixture_history) if self.target_found else 0

    @property
    def next_check_step(self) -> int | str:
        if self.failure_reason is not None:
            return "failed"
        if self.active_fixture is not None:
            return (
                "close_target_fixture"
                if self.target_observed
                else "close_active_fixture"
            )
        if self.target_found:
            if self.completion_button_pressed:
                return "done"
            return (
                f"press_completion_button_"
                f"{self.completion_button_press_count + 1}_of_"
                f"{self.required_button_presses}"
            )
        return self.num_checked + 1


@dataclass(frozen=True)
class PhysicalState:
    open_fraction_by_fixture: Mapping[str, float]


@dataclass(frozen=True)
class TransitionReport:
    primitive: str
    fixture: str
    checks: Mapping[str, bool]
    reason: str | None

    @property
    def passed(self) -> bool:
        return bool(all(self.checks.values()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "fixture": self.fixture,
            "checks": dict(self.checks),
            "pass": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SuccessReport:
    target_found: bool
    search_stopped_at_target: bool
    at_least_one_fixture_checked: bool
    no_duplicate_completion: bool
    no_active_fixture: bool
    all_fixtures_closed: bool
    no_invalid_open: bool
    no_invalid_inspect: bool
    no_invalid_close: bool
    no_duplicate_check: bool
    no_close_before_inspect: bool
    no_search_after_target: bool
    completion_button_pressed: bool
    completion_button_count_matches_search_count: bool
    no_invalid_completion_button_press: bool

    @property
    def success(self) -> bool:
        return bool(all(self.checks().values()))

    def checks(self) -> dict[str, bool]:
        return {name: bool(value) for name, value in self.__dict__.items()}


def open_precondition_checks(state: SemanticState, fixture: str) -> dict[str, bool]:
    name = str(fixture)
    return {
        "fixture_known": name in state.fixture_names,
        "no_active_fixture": state.active_fixture is None,
        "not_previously_checked": name not in state.checked_fixture_set,
        "target_not_previously_found": not state.target_found,
    }


def close_precondition_checks(state: SemanticState, fixture: str) -> dict[str, bool]:
    name = str(fixture)
    return {
        "fixture_known": name in state.fixture_names,
        "matches_active_fixture": state.active_fixture == name,
        "inspected_before_close": bool(state.inspected_active_fixture),
    }


def completion_button_precondition_checks(
    state: SemanticState,
    *,
    all_drawers_closed: bool,
) -> dict[str, bool]:
    """Accept one press while fewer than k physical presses are recorded."""

    history = state.completed_fixture_history
    return {
        "target_found": state.target_found,
        "target_is_last_checked": bool(history) and history[-1] == state.target_fixture,
        "at_least_one_fixture_checked": bool(history),
        "no_duplicate_completion": len(set(history)) == len(history),
        "no_active_fixture": state.active_fixture is None,
        "all_fixtures_closed": bool(all_drawers_closed),
        "button_presses_remaining": (
            state.completion_button_press_count < state.required_button_presses
        ),
    }


def record_open_attempt(
    state: SemanticState,
    fixture: str,
    *,
    drawer_open: bool,
    failure_reason: str | None = None,
) -> TransitionReport:
    """Commit an open event only after the drawer is physically open."""

    name = str(fixture)
    state.open_attempt_history.append(name)
    checks = {
        **open_precondition_checks(state, name),
        "drawer_open": bool(drawer_open),
    }
    reason = _open_failure_reason(checks, failure_reason)
    if all(checks.values()):
        state.active_fixture = name
        state.inspected_active_fixture = False
        state.target_observed = False
    else:
        state.invalid_open_detected = True
        if not checks["not_previously_checked"]:
            state.duplicate_fixture_check = True
        if not checks["target_not_previously_found"]:
            state.search_after_target_detected = True
        _latch_failure(state, reason)
    return TransitionReport(
        "open_fixture", name, checks, None if all(checks.values()) else reason
    )


def record_inspection_attempt(
    state: SemanticState,
    fixture: str,
    *,
    drawer_open: bool,
) -> TransitionReport:
    name = str(fixture)
    checks = {
        "fixture_known": name in state.fixture_names,
        "matches_active_fixture": state.active_fixture == name,
        "drawer_open": bool(drawer_open),
        "target_not_previously_found": not state.target_found,
    }
    reason: str | None = None
    if not checks["fixture_known"]:
        reason = "unknown_fixture"
    elif not checks["matches_active_fixture"]:
        reason = "inspect_without_matching_open"
    elif not checks["drawer_open"]:
        reason = "inspect_fixture_not_open"
    elif not checks["target_not_previously_found"]:
        reason = "search_continued_after_target_found"
    if all(checks.values()):
        state.inspect_history.append(name)
        state.inspected_active_fixture = True
        if name == state.target_fixture:
            state.target_observed = True
    else:
        state.invalid_inspect_detected = True
        if not checks["target_not_previously_found"]:
            state.search_after_target_detected = True
        _latch_failure(state, reason)
    return TransitionReport(
        "inspect_fixture",
        name,
        checks,
        None if all(checks.values()) else reason,
    )


def record_close_attempt(
    state: SemanticState,
    fixture: str,
    *,
    drawer_closed: bool,
    failure_reason: str | None = None,
) -> TransitionReport:
    """Mark a fixture checked only after open-inspect-close is complete."""

    name = str(fixture)
    state.close_attempt_history.append(name)
    checks = {
        **close_precondition_checks(state, name),
        "drawer_closed": bool(drawer_closed),
    }
    reason: str | None = None
    if not checks["fixture_known"]:
        reason = "unknown_fixture"
    elif not checks["matches_active_fixture"]:
        reason = "close_without_matching_open"
    elif not checks["inspected_before_close"]:
        reason = "close_before_inspect"
    elif not checks["drawer_closed"]:
        reason = failure_reason or "drawer_not_closed"
    if all(checks.values()):
        state.checked_fixture_set.add(name)
        state.completed_fixture_history.append(name)
        state.target_found = bool(
            name == state.target_fixture and state.target_observed
        )
        state.active_fixture = None
        state.inspected_active_fixture = False
        state.target_observed = False
    else:
        state.invalid_close_detected = True
        if not checks["inspected_before_close"]:
            state.close_before_inspect_detected = True
        _latch_failure(state, reason)
    return TransitionReport(
        "close_fixture",
        name,
        checks,
        None if all(checks.values()) else reason,
    )


def record_completion_button_attempt(
    state: SemanticState,
    *,
    all_drawers_closed: bool,
    button_physically_pressed: bool,
    failure_reason: str | None = None,
) -> TransitionReport:
    """Record one debounced physical press and finish exactly on press k."""

    state.completion_button_attempted = True
    state.completion_button_attempt_count += 1
    checks = {
        **completion_button_precondition_checks(
            state,
            all_drawers_closed=all_drawers_closed,
        ),
        "button_physically_pressed": bool(button_physically_pressed),
    }
    reason = _completion_button_failure_reason(checks, failure_reason)
    if button_physically_pressed:
        state.completion_button_press_count += 1
    if all(checks.values()):
        state.completion_button_pressed = bool(
            state.completion_button_press_count == state.required_button_presses
        )
    else:
        state.invalid_completion_button_press = True
        _latch_failure(state, reason)
    return TransitionReport(
        "press_completion_button",
        "completion_button",
        checks,
        None if all(checks.values()) else reason,
    )


def evaluate_success(
    state: SemanticState,
    physical: PhysicalState,
    *,
    closed_threshold: float = 0.08,
) -> SuccessReport:
    fractions = {
        str(name): float(value)
        for name, value in physical.open_fraction_by_fixture.items()
    }
    all_closed = all(
        fractions.get(name, float("inf")) <= float(closed_threshold)
        for name in state.fixture_names
    )
    history = state.completed_fixture_history
    target_last = bool(history) and history[-1] == state.target_fixture
    return SuccessReport(
        target_found=state.target_found,
        search_stopped_at_target=(
            state.target_found
            and target_last
            and not state.search_after_target_detected
        ),
        at_least_one_fixture_checked=bool(history),
        no_duplicate_completion=len(set(history)) == len(history),
        no_active_fixture=state.active_fixture is None,
        all_fixtures_closed=all_closed,
        no_invalid_open=not state.invalid_open_detected,
        no_invalid_inspect=not state.invalid_inspect_detected,
        no_invalid_close=not state.invalid_close_detected,
        no_duplicate_check=not state.duplicate_fixture_check,
        no_close_before_inspect=not state.close_before_inspect_detected,
        no_search_after_target=not state.search_after_target_detected,
        completion_button_pressed=state.completion_button_pressed,
        completion_button_count_matches_search_count=(
            state.target_found
            and state.completion_button_press_count == state.required_button_presses
        ),
        no_invalid_completion_button_press=(not state.invalid_completion_button_press),
    )


def failure_attribution(
    state: SemanticState,
    physical: PhysicalState,
    *,
    closed_threshold: float = 0.08,
) -> str:
    report = evaluate_success(state, physical, closed_threshold=closed_threshold)
    if report.success:
        return "success"
    if state.failure_reason is not None:
        return state.failure_reason
    if not report.target_found:
        return "target_not_found"
    if not report.all_fixtures_closed:
        return "fixture_left_open"
    if not report.completion_button_count_matches_search_count:
        return "completion_button_count_does_not_match_search_count"
    if not report.completion_button_pressed:
        return "completion_button_sequence_incomplete"
    return "low_level_or_unknown"


def diagnostics(
    state: SemanticState,
    physical: PhysicalState,
    *,
    closed_threshold: float = 0.08,
    fixture_display_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    report = evaluate_success(state, physical, closed_threshold=closed_threshold)
    fractions = {
        str(name): float(value)
        for name, value in physical.open_fraction_by_fixture.items()
    }
    return {
        "axis": AXIS,
        "fixture_names": list(state.fixture_names),
        "fixture_display_names": dict(
            FIXTURE_DISPLAY_NAMES
            if fixture_display_names is None
            else fixture_display_names
        ),
        "target_fixture": state.target_fixture,
        "target_observed": state.target_observed,
        "target_found": state.target_found,
        "checked_fixture_set": sorted(state.checked_fixture_set),
        "completed_fixture_history": list(state.completed_fixture_history),
        "open_attempt_history": list(state.open_attempt_history),
        "inspect_history": list(state.inspect_history),
        "close_attempt_history": list(state.close_attempt_history),
        "active_fixture": state.active_fixture,
        "inspected_active_fixture": bool(state.inspected_active_fixture),
        "current_slot_states": fractions,
        "num_checked": state.num_checked,
        "search_count": state.required_button_presses,
        "required_button_presses": state.required_button_presses,
        "next_check_step": state.next_check_step,
        "duplicate_fixture_check": state.duplicate_fixture_check,
        "search_after_target_detected": state.search_after_target_detected,
        "invalid_open_detected": state.invalid_open_detected,
        "invalid_inspect_detected": state.invalid_inspect_detected,
        "invalid_close_detected": state.invalid_close_detected,
        "close_before_inspect_detected": state.close_before_inspect_detected,
        "completion_button_pressed": state.completion_button_pressed,
        "completion_button_attempted": state.completion_button_attempted,
        "completion_button_attempt_count": state.completion_button_attempt_count,
        "completion_button_press_count": state.completion_button_press_count,
        "invalid_completion_button_press": state.invalid_completion_button_press,
        "fixture_left_open": not report.all_fixtures_closed,
        "failure_reason": state.failure_reason,
        "success_checks": report.checks(),
        "success": report.success,
        "failure_attribution": failure_attribution(
            state,
            physical,
            closed_threshold=closed_threshold,
        ),
    }


def _open_failure_reason(
    checks: Mapping[str, bool],
    supplied: str | None,
) -> str:
    if not checks["fixture_known"]:
        return "unknown_fixture"
    if not checks["no_active_fixture"]:
        return "open_while_fixture_active"
    if not checks["not_previously_checked"]:
        return "duplicate_fixture_check"
    if not checks["target_not_previously_found"]:
        return "search_continued_after_target_found"
    return supplied or "drawer_not_open"


def _completion_button_failure_reason(
    checks: Mapping[str, bool],
    supplied: str | None,
) -> str:
    if not checks["target_found"]:
        return "completion_button_pressed_before_target_found"
    if not checks["target_is_last_checked"]:
        return "target_not_last_checked"
    if not checks["at_least_one_fixture_checked"]:
        return "no_fixture_checked"
    if not checks["no_duplicate_completion"]:
        return "duplicate_fixture_completion_before_button"
    if not checks["no_active_fixture"]:
        return "completion_button_pressed_while_fixture_active"
    if not checks["all_fixtures_closed"]:
        return "completion_button_pressed_with_fixture_open"
    if not checks["button_presses_remaining"]:
        return "completion_button_pressed_too_many_times"
    if not checks["button_physically_pressed"]:
        return supplied or "completion_button_not_physically_pressed"
    return supplied or "invalid_completion_button_press"


def _latch_failure(state: SemanticState, reason: str | None) -> None:
    if state.failure_reason is None and reason is not None:
        state.failure_reason = str(reason)


__all__ = [
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "TransitionReport",
    "close_precondition_checks",
    "completion_button_precondition_checks",
    "diagnostics",
    "evaluate_success",
    "failure_attribution",
    "open_precondition_checks",
    "record_close_attempt",
    "record_completion_button_attempt",
    "record_inspection_attempt",
    "record_open_attempt",
]
