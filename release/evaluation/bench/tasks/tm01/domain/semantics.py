"""Authoritative simulator-free TM01 two-thread state machine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.tm01.domain.contracts import (
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    StageEvent,
)


LEGAL_EVENT_ROUTES = {
    (
        StageEvent.A1,
        StageEvent.B1,
        StageEvent.A2,
        StageEvent.B2,
        StageEvent.COMPLETION_BUTTON,
    ),
    (
        StageEvent.A1,
        StageEvent.B1,
        StageEvent.B2,
        StageEvent.A2,
        StageEvent.COMPLETION_BUTTON,
    ),
}
LEGAL_CORE_ROUTES = frozenset(route[:-1] for route in LEGAL_EVENT_ROUTES)
LEGAL_PREFIXES = frozenset(
    route[:length]
    for route in LEGAL_CORE_ROUTES
    for length in range(len(route) + 1)
)


@dataclass
class SemanticState:
    a1_complete: bool = False
    a2_complete: bool = False
    b1_complete: bool = False
    b2_complete: bool = False
    boiling_ready: bool = False
    checked_drawers: set[str] = field(default_factory=set)
    completed_drawer_history: list[str] = field(default_factory=list)
    active_drawer: str | None = None
    drawer_dwell_steps: int = 0
    drawer_inspected: bool = False
    open_attempt_history: list[str] = field(default_factory=list)
    inspect_history: list[str] = field(default_factory=list)
    close_attempt_history: list[str] = field(default_factory=list)
    invalid_open_detected: bool = False
    invalid_inspect_detected: bool = False
    invalid_close_detected: bool = False
    duplicate_drawer_check: bool = False
    close_before_inspect_detected: bool = False
    event_history: list[StageEvent] = field(default_factory=list)
    knob_on_event_count: int = 0
    knob_off_event_count: int = 0
    green_press_count: int = 0
    completion_button_attempted: bool = False
    completion_button_pressed: bool = False
    completion_button_id: str | None = None
    completion_button_attempt_history: list[str] = field(default_factory=list)
    completion_button_attempt_count: int = 0
    completion_button_press_count: int = 0
    invalid_completion_button_press: bool = False
    failure_reason: str | None = None
    failure_labels: list[str] = field(default_factory=list)
    absorbing_failure: bool = False

    @property
    def pending_a2(self) -> bool:
        return self.a1_complete and not self.a2_complete

    @property
    def pending_b2(self) -> bool:
        return self.b1_complete and not self.b2_complete

    @property
    def pending_mask(self) -> frozenset[str]:
        result: set[str] = set()
        if self.pending_a2:
            result.add("A2")
        if self.pending_b2:
            result.add("B2")
        return frozenset(result)

    def fail(self, label: str) -> None:
        value = str(label)
        if value not in self.failure_labels:
            self.failure_labels.append(value)
        if self.failure_reason is None:
            self.failure_reason = value
        self.absorbing_failure = True


@dataclass(frozen=True)
class PhysicalState:
    knob_qpos_rad: float
    target_burner_on: bool
    all_other_burners_off: bool
    drawer_open_fraction: Mapping[str, float]
    pot_stable_on_burner: bool


@dataclass(frozen=True)
class TransitionReport:
    event: str
    passed: bool
    reason: str | None
    checks: Mapping[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "pass": self.passed,
            "reason": self.reason,
            "checks": dict(self.checks),
        }


@dataclass(frozen=True)
class SuccessReport:
    legal_event_route: bool
    native_knob_on_once: bool
    native_knob_off_once: bool
    knob_off: bool
    burner_off: bool
    all_other_burners_off: bool
    both_drawers_checked_once: bool
    all_drawers_closed: bool
    no_active_drawer: bool
    pot_stable: bool
    either_completion_button_pressed: bool
    no_invalid_open: bool
    no_invalid_inspect: bool
    no_invalid_close: bool
    no_duplicate_check: bool
    no_close_before_inspect: bool
    completion_button_pressed_once: bool
    no_invalid_completion_button_press: bool
    no_absorbing_failure: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {name: bool(value) for name, value in self.__dict__.items()}


def record_a1(state: SemanticState) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "a1_not_complete": not state.a1_complete,
        "no_prior_events": not state.event_history,
        "no_drawer_active": state.active_drawer is None,
    }
    if all(checks.values()):
        state.a1_complete = True
        state.knob_on_event_count += 1
        state.event_history.append(StageEvent.A1)
        return TransitionReport("A1", True, None, checks)
    reason = "duplicate_A1" if state.a1_complete else "invalid_A1_order"
    state.fail(reason)
    return TransitionReport("A1", False, reason, checks)


def legal_next_events(history: Sequence[StageEvent]) -> frozenset[StageEvent]:
    prefix = tuple(history)
    if prefix not in LEGAL_PREFIXES:
        return frozenset()
    return frozenset(
        route[len(prefix)]
        for route in LEGAL_CORE_ROUTES
        if len(prefix) < len(route) and route[: len(prefix)] == prefix
    )


def open_precondition_checks(state: SemanticState, drawer: str) -> dict[str, bool]:
    name = str(drawer)
    expected = StageEvent.B1 if not state.b1_complete else StageEvent.B2
    return {
        "known_drawer": name in DRAWER_NAMES,
        "a1_complete": state.a1_complete,
        "no_absorbing_failure": not state.absorbing_failure,
        "no_active_drawer": state.active_drawer is None,
        "not_previously_checked": name not in state.checked_drawers,
        "drawer_event_is_legal_next": expected in legal_next_events(state.event_history),
    }


def close_precondition_checks(state: SemanticState, drawer: str) -> dict[str, bool]:
    name = str(drawer)
    return {
        "known_drawer": name in DRAWER_NAMES,
        "matches_active_drawer": state.active_drawer == name,
        "inspected_before_close": bool(state.drawer_inspected),
        "not_previously_checked": name not in state.checked_drawers,
    }


def completion_button_precondition_checks(
    state: SemanticState,
    *,
    all_drawers_closed: bool,
) -> dict[str, bool]:
    return {
        "a2_complete": state.a2_complete,
        "b2_complete": state.b2_complete,
        "no_active_drawer": state.active_drawer is None,
        "all_drawers_closed": bool(all_drawers_closed),
        "not_previously_pressed": not state.completion_button_pressed,
        "button_presses_remaining": state.completion_button_press_count < 1,
    }


def record_open_attempt(
    state: SemanticState,
    drawer: str,
    *,
    drawer_open: bool,
    failure_reason: str | None = None,
) -> TransitionReport:
    name = str(drawer)
    state.open_attempt_history.append(name)
    checks = {**open_precondition_checks(state, name), "drawer_open": bool(drawer_open)}
    if all(checks.values()):
        state.active_drawer = name
        state.drawer_dwell_steps = 0
        state.drawer_inspected = False
        return TransitionReport("open_drawer", True, None, checks)
    state.invalid_open_detected = True
    if not checks["not_previously_checked"]:
        state.duplicate_drawer_check = True
    reason = _open_failure_reason(checks, failure_reason)
    state.fail(reason)
    return TransitionReport("open_drawer", False, reason, checks)


def record_inspection_attempt(
    state: SemanticState,
    drawer: str,
    *,
    drawer_open: bool,
) -> TransitionReport:
    name = str(drawer)
    checks = {
        "known_drawer": name in DRAWER_NAMES,
        "no_absorbing_failure": not state.absorbing_failure,
        "matches_active_drawer": state.active_drawer == name,
        "drawer_open": bool(drawer_open),
        "not_previously_inspected": not state.drawer_inspected,
    }
    if all(checks.values()):
        state.inspect_history.append(name)
        state.drawer_inspected = True
        return TransitionReport("inspect_drawer", True, None, checks)
    state.invalid_inspect_detected = True
    reason = (
        "unknown_drawer"
        if not checks["known_drawer"]
        else "absorbing_failure"
        if not checks["no_absorbing_failure"]
        else "inspect_without_matching_open"
        if not checks["matches_active_drawer"]
        else "inspect_drawer_not_open"
        if not checks["drawer_open"]
        else "duplicate_inspection_attempt"
    )
    state.fail(reason)
    return TransitionReport("inspect_drawer", False, reason, checks)


def record_close_attempt(
    state: SemanticState,
    drawer: str,
    *,
    drawer_closed: bool,
    failure_reason: str | None = None,
) -> TransitionReport:
    name = str(drawer)
    state.close_attempt_history.append(name)
    event = StageEvent.B1 if not state.b1_complete else StageEvent.B2
    checks = {
        **close_precondition_checks(state, name),
        "drawer_closed": bool(drawer_closed),
        "event_is_legal_next": event in legal_next_events(state.event_history),
        "no_absorbing_failure": not state.absorbing_failure,
    }
    if all(checks.values()):
        state.checked_drawers.add(name)
        state.completed_drawer_history.append(name)
        state.active_drawer = None
        state.drawer_dwell_steps = 0
        state.drawer_inspected = False
        if event is StageEvent.B1:
            state.b1_complete = True
            state.boiling_ready = True
        else:
            state.b2_complete = True
        state.event_history.append(event)
        return TransitionReport(event.value, True, None, checks)
    state.invalid_close_detected = True
    if not checks["inspected_before_close"]:
        state.close_before_inspect_detected = True
    if not checks["not_previously_checked"]:
        state.duplicate_drawer_check = True
    reason = _close_failure_reason(checks, failure_reason)
    state.fail(reason)
    return TransitionReport(event.value, False, reason, checks)


def record_completion_button_attempt(
    state: SemanticState,
    *,
    button_id: str = COMPLETION_BUTTON_NAMES[0],
    all_drawers_closed: bool = True,
    button_physically_pressed: bool = True,
    failure_reason: str | None = None,
) -> TransitionReport:
    name = str(button_id)
    state.completion_button_attempted = True
    state.completion_button_attempt_count += 1
    state.green_press_count += 1
    state.completion_button_attempt_history.append(name)
    checks = {
        "known_completion_button": name in COMPLETION_BUTTON_NAMES,
        "no_absorbing_failure": not state.absorbing_failure,
        **completion_button_precondition_checks(
            state, all_drawers_closed=all_drawers_closed
        ),
        "button_physically_pressed": bool(button_physically_pressed),
    }
    if button_physically_pressed:
        state.completion_button_press_count += 1
    if all(checks.values()):
        state.completion_button_pressed = True
        state.completion_button_id = name
        state.event_history.append(StageEvent.COMPLETION_BUTTON)
        return TransitionReport("completion_button", True, None, checks)
    state.invalid_completion_button_press = True
    reason = _completion_failure_reason(checks, failure_reason)
    state.fail(reason)
    return TransitionReport("completion_button", False, reason, checks)


def start_drawer_cycle(state: SemanticState, drawer: str) -> TransitionReport:
    report = record_open_attempt(state, drawer, drawer_open=True)
    return TransitionReport("drawer_cycle_started", report.passed, report.reason, report.checks)


def mark_active_drawer_inspected(state: SemanticState) -> None:
    if state.active_drawer is not None:
        record_inspection_attempt(state, state.active_drawer, drawer_open=True)


def cancel_incomplete_drawer_cycle(state: SemanticState) -> None:
    if state.active_drawer is not None:
        record_close_attempt(
            state,
            state.active_drawer,
            drawer_closed=True,
            failure_reason="close_before_inspect",
        )


def record_drawer_completion(
    state: SemanticState,
    drawer: str,
) -> TransitionReport:
    return record_close_attempt(state, drawer, drawer_closed=True)


def record_a2(state: SemanticState) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "a1_complete": state.a1_complete,
        "b1_complete": state.b1_complete,
        "boiling_ready": state.boiling_ready,
        "a2_not_complete": not state.a2_complete,
        "no_active_drawer": state.active_drawer is None,
    }
    if all(checks.values()):
        state.a2_complete = True
        state.knob_off_event_count += 1
        state.event_history.append(StageEvent.A2)
        return TransitionReport("A2", True, None, checks)
    if state.a2_complete:
        reason = "duplicate_A2"
    elif state.active_drawer is not None:
        reason = "A2_during_active_drawer"
    elif not state.b1_complete:
        reason = "early_A2"
    else:
        reason = "invalid_A2_order"
    state.fail(reason)
    return TransitionReport("A2", False, reason, checks)


def record_completion_request(
    state: SemanticState,
    button_id: str,
) -> TransitionReport:
    return record_completion_button_attempt(
        state,
        button_id=button_id,
        all_drawers_closed=True,
        button_physically_pressed=True,
    )


def evaluate_success(
    state: SemanticState,
    physical: PhysicalState,
    *,
    off_commit_max_rad: float = 0.10,
    drawer_closed_threshold: float = 0.08,
) -> SuccessReport:
    return SuccessReport(
        legal_event_route=tuple(state.event_history) in LEGAL_EVENT_ROUTES,
        native_knob_on_once=state.knob_on_event_count == 1,
        native_knob_off_once=state.knob_off_event_count == 1,
        knob_off=float(physical.knob_qpos_rad) <= float(off_commit_max_rad),
        burner_off=not bool(physical.target_burner_on),
        all_other_burners_off=bool(physical.all_other_burners_off),
        both_drawers_checked_once=(
            state.checked_drawers == set(DRAWER_NAMES)
            and len(state.completed_drawer_history) == 2
            and len(set(state.completed_drawer_history)) == 2
        ),
        all_drawers_closed=all(
            float(physical.drawer_open_fraction.get(name, 1.0))
            <= float(drawer_closed_threshold)
            for name in DRAWER_NAMES
        ),
        no_active_drawer=state.active_drawer is None,
        pot_stable=bool(physical.pot_stable_on_burner),
        either_completion_button_pressed=state.completion_button_pressed,
        no_invalid_open=not state.invalid_open_detected,
        no_invalid_inspect=not state.invalid_inspect_detected,
        no_invalid_close=not state.invalid_close_detected,
        no_duplicate_check=not state.duplicate_drawer_check,
        no_close_before_inspect=not state.close_before_inspect_detected,
        completion_button_pressed_once=(
            state.completion_button_press_count == 1
            and state.completion_button_attempt_count == 1
        ),
        no_invalid_completion_button_press=not state.invalid_completion_button_press,
        no_absorbing_failure=not state.absorbing_failure,
    )


def failure_attribution(
    state: SemanticState,
    physical: PhysicalState,
    *,
    off_commit_max_rad: float = 0.10,
    drawer_closed_threshold: float = 0.08,
) -> str:
    report = evaluate_success(
        state,
        physical,
        off_commit_max_rad=off_commit_max_rad,
        drawer_closed_threshold=drawer_closed_threshold,
    )
    if report.success:
        return "success"
    if state.failure_reason is not None:
        return state.failure_reason
    if state.invalid_open_detected:
        return "invalid_drawer_open_attempt"
    if state.invalid_inspect_detected:
        return "invalid_drawer_inspection_attempt"
    if state.invalid_close_detected:
        return "invalid_drawer_close_attempt"
    if not report.both_drawers_checked_once:
        return "both_drawers_not_checked_exactly_once"
    if not report.all_drawers_closed:
        return "drawer_left_open"
    if not report.completion_button_pressed_once:
        return "completion_button_was_not_pressed_exactly_once"
    return "low_level_or_unknown"


def _open_failure_reason(
    checks: Mapping[str, bool], supplied: str | None
) -> str:
    if not checks["known_drawer"]:
        return "unknown_drawer"
    if not checks["a1_complete"]:
        return "action_before_A1"
    if not checks["no_active_drawer"]:
        return "overlapping_drawer_cycles"
    if not checks["not_previously_checked"]:
        return "duplicate_drawer_check"
    if not checks["drawer_event_is_legal_next"]:
        return "invalid_drawer_open_order"
    return supplied or "drawer_not_open"


def _close_failure_reason(
    checks: Mapping[str, bool], supplied: str | None
) -> str:
    if not checks["known_drawer"]:
        return "unknown_drawer"
    if not checks["matches_active_drawer"]:
        return "close_without_matching_open"
    if not checks["inspected_before_close"]:
        return "close_before_inspect"
    if not checks["event_is_legal_next"]:
        return "invalid_drawer_close_order"
    if not checks["no_absorbing_failure"]:
        return "absorbing_failure"
    return supplied or "drawer_not_closed"


def _completion_failure_reason(
    checks: Mapping[str, bool], supplied: str | None
) -> str:
    if not checks["known_completion_button"]:
        return "unknown_completion_button"
    if not checks["a2_complete"] or not checks["b2_complete"]:
        return "completion_button_pressed_before_task_complete"
    if not checks["no_active_drawer"]:
        return "completion_button_pressed_while_drawer_active"
    if not checks["all_drawers_closed"]:
        return "completion_button_pressed_with_drawer_open"
    if not checks["not_previously_pressed"]:
        return "duplicate_completion_button_press"
    if not checks["button_presses_remaining"]:
        return "completion_button_pressed_too_many_times"
    if not checks["no_absorbing_failure"]:
        return "absorbing_failure"
    return supplied or "completion_button_not_physically_pressed"


def diagnostics(state: SemanticState) -> dict[str, Any]:
    return {
        "a1_complete": state.a1_complete,
        "a2_complete": state.a2_complete,
        "b1_complete": state.b1_complete,
        "b2_complete": state.b2_complete,
        "boiling_ready": state.boiling_ready,
        "pending_A2": state.pending_a2,
        "pending_B2": state.pending_b2,
        "pending_mask": sorted(state.pending_mask),
        "checked_drawers": sorted(state.checked_drawers),
        "completed_drawer_history": list(state.completed_drawer_history),
        "active_drawer": state.active_drawer,
        "open_attempt_history": list(state.open_attempt_history),
        "inspect_history": list(state.inspect_history),
        "close_attempt_history": list(state.close_attempt_history),
        "invalid_open_detected": state.invalid_open_detected,
        "invalid_inspect_detected": state.invalid_inspect_detected,
        "invalid_close_detected": state.invalid_close_detected,
        "duplicate_drawer_check": state.duplicate_drawer_check,
        "close_before_inspect_detected": state.close_before_inspect_detected,
        "event_history": [event.value for event in state.event_history],
        "knob_on_event_count": state.knob_on_event_count,
        "knob_off_event_count": state.knob_off_event_count,
        "green_press_count": state.green_press_count,
        "completion_button_attempted": state.completion_button_attempted,
        "completion_button_pressed": state.completion_button_pressed,
        "completion_button_id": state.completion_button_id,
        "completion_button_attempt_history": list(
            state.completion_button_attempt_history
        ),
        "completion_button_attempt_count": state.completion_button_attempt_count,
        "completion_button_press_count": state.completion_button_press_count,
        "invalid_completion_button_press": state.invalid_completion_button_press,
        "failure_reason": state.failure_reason,
        "failure_labels": list(state.failure_labels),
        "absorbing_failure": state.absorbing_failure,
    }


__all__ = [
    "LEGAL_CORE_ROUTES",
    "LEGAL_EVENT_ROUTES",
    "LEGAL_PREFIXES",
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "TransitionReport",
    "cancel_incomplete_drawer_cycle",
    "close_precondition_checks",
    "completion_button_precondition_checks",
    "diagnostics",
    "evaluate_success",
    "failure_attribution",
    "legal_next_events",
    "mark_active_drawer_inspected",
    "record_a1",
    "record_a2",
    "record_close_attempt",
    "record_completion_button_attempt",
    "record_completion_request",
    "record_drawer_completion",
    "record_inspection_attempt",
    "record_open_attempt",
    "start_drawer_cycle",
]
