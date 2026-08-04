"""Simulator-free TM03 three-thread state machine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.tm03.domain.contracts import (
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    ROUTE_SPECS,
    StageEvent,
)


LEGAL_CORE_ROUTES = frozenset(spec.events for spec in ROUTE_SPECS)
LEGAL_EVENT_ROUTES = frozenset(
    (*route, StageEvent.COMPLETION_BUTTON) for route in LEGAL_CORE_ROUTES
)
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
    c1_complete: bool = False
    c2_complete: bool = False
    checked_drawers: set[str] = field(default_factory=set)
    completed_drawer_history: list[str] = field(default_factory=list)
    active_drawer: str | None = None
    drawer_dwell_steps: int = 0
    drawer_inspected: bool = False
    event_history: list[StageEvent] = field(default_factory=list)
    knob_on_event_count: int = 0
    knob_off_event_count: int = 0
    toaster_oven_start_event_count: int = 0
    toaster_oven_stop_event_count: int = 0
    toaster_oven_attempt_history: list[str] = field(default_factory=list)
    green_press_count: int = 0
    completion_button_attempted: bool = False
    completion_button_pressed: bool = False
    completion_button_id: str | None = None
    completion_button_attempt_history: list[str] = field(default_factory=list)
    failure_labels: list[str] = field(default_factory=list)
    absorbing_failure: bool = False

    @property
    def pending_a2(self) -> bool:
        return self.a1_complete and not self.a2_complete

    @property
    def pending_b2(self) -> bool:
        return self.b1_complete and not self.b2_complete

    @property
    def pending_c2(self) -> bool:
        return self.c1_complete and not self.c2_complete

    @property
    def pending_mask(self) -> frozenset[str]:
        result: set[str] = set()
        if self.pending_a2:
            result.add("A2")
        if self.pending_b2:
            result.add("B2")
        if self.pending_c2:
            result.add("C2")
        return frozenset(result)

    def fail(self, label: str) -> None:
        value = str(label)
        if value not in self.failure_labels:
            self.failure_labels.append(value)
        self.absorbing_failure = True


@dataclass(frozen=True)
class PhysicalState:
    knob_qpos_rad: float
    target_burner_on: bool
    all_other_burners_off: bool
    drawer_open_fraction: Mapping[str, float]
    pot_stable_on_burner: bool
    toaster_oven_on: bool
    toaster_oven_door_closed: bool
    toaster_oven_food_inside: bool


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
    toaster_oven_started_once: bool
    toaster_oven_stopped_once: bool
    knob_off: bool
    burner_off: bool
    toaster_oven_off: bool
    toaster_oven_door_closed: bool
    toaster_oven_food_inside: bool
    all_other_burners_off: bool
    both_drawers_checked_once: bool
    all_drawers_closed: bool
    no_active_drawer: bool
    pot_stable: bool
    either_completion_button_pressed: bool
    no_absorbing_failure: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {name: bool(value) for name, value in self.__dict__.items()}


def legal_next_events(history: Sequence[StageEvent]) -> frozenset[StageEvent]:
    prefix = tuple(history)
    if prefix not in LEGAL_PREFIXES:
        return frozenset()
    return frozenset(
        route[len(prefix)]
        for route in LEGAL_CORE_ROUTES
        if len(prefix) < len(route) and route[: len(prefix)] == prefix
    )


def _record_ordered(state: SemanticState, event: StageEvent) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "event_is_legal_next": event in legal_next_events(state.event_history),
        "no_drawer_active": state.active_drawer is None,
    }
    if all(checks.values()):
        state.event_history.append(event)
        return TransitionReport(event.value, True, None, checks)
    duplicate = event in state.event_history
    reason = f"duplicate_{event.value}" if duplicate else f"invalid_{event.value}_order"
    state.fail(reason)
    return TransitionReport(event.value, False, reason, checks)


def record_a1(state: SemanticState) -> TransitionReport:
    report = _record_ordered(state, StageEvent.A1)
    if report.passed:
        state.a1_complete = True
        state.knob_on_event_count += 1
    return report


def record_a2(state: SemanticState) -> TransitionReport:
    report = _record_ordered(state, StageEvent.A2)
    if report.passed:
        state.a2_complete = True
        state.knob_off_event_count += 1
    return report


def record_c1(state: SemanticState) -> TransitionReport:
    state.toaster_oven_attempt_history.append("start")
    report = _record_ordered(state, StageEvent.C1)
    if report.passed:
        state.c1_complete = True
        state.toaster_oven_start_event_count += 1
    return report


def record_c2(state: SemanticState) -> TransitionReport:
    state.toaster_oven_attempt_history.append("stop")
    report = _record_ordered(state, StageEvent.C2)
    if report.passed:
        state.c2_complete = True
        state.toaster_oven_stop_event_count += 1
    return report


def start_drawer_cycle(state: SemanticState, drawer: str) -> TransitionReport:
    name = str(drawer)
    expected_event = StageEvent.B1 if not state.b1_complete else StageEvent.B2
    checks = {
        "known_drawer": name in DRAWER_NAMES,
        "no_absorbing_failure": not state.absorbing_failure,
        "no_active_drawer": state.active_drawer is None,
        "not_previously_checked": name not in state.checked_drawers,
        "drawer_event_is_legal_next": expected_event
        in legal_next_events(state.event_history),
    }
    if all(checks.values()):
        state.active_drawer = name
        state.drawer_dwell_steps = 0
        state.drawer_inspected = False
        return TransitionReport("drawer_cycle_started", True, None, checks)
    if name in state.checked_drawers:
        reason = "duplicate_drawer_check"
    elif state.active_drawer is not None:
        reason = "overlapping_drawer_cycles"
    else:
        reason = f"invalid_{expected_event.value}_order"
    state.fail(reason)
    return TransitionReport("drawer_cycle_started", False, reason, checks)


def mark_active_drawer_inspected(state: SemanticState) -> None:
    if state.active_drawer is not None:
        state.drawer_inspected = True


def cancel_incomplete_drawer_cycle(state: SemanticState) -> None:
    state.active_drawer = None
    state.drawer_dwell_steps = 0
    state.drawer_inspected = False


def record_drawer_completion(
    state: SemanticState, drawer: str
) -> TransitionReport:
    name = str(drawer)
    event = StageEvent.B1 if not state.b1_complete else StageEvent.B2
    checks = {
        "known_drawer": name in DRAWER_NAMES,
        "matches_active_drawer": state.active_drawer == name,
        "drawer_inspected": state.drawer_inspected,
        "not_previously_checked": name not in state.checked_drawers,
        "event_is_legal_next": event in legal_next_events(state.event_history),
        "no_absorbing_failure": not state.absorbing_failure,
    }
    if all(checks.values()):
        state.checked_drawers.add(name)
        state.completed_drawer_history.append(name)
        state.active_drawer = None
        state.drawer_dwell_steps = 0
        state.drawer_inspected = False
        state.event_history.append(event)
        if event is StageEvent.B1:
            state.b1_complete = True
        else:
            state.b2_complete = True
        return TransitionReport(event.value, True, None, checks)
    reason = (
        "duplicate_drawer_check"
        if name in state.checked_drawers
        else f"invalid_{event.value}_completion"
    )
    state.fail(reason)
    return TransitionReport(event.value, False, reason, checks)


def record_completion_request(
    state: SemanticState, button_id: str
) -> TransitionReport:
    name = str(button_id)
    state.completion_button_attempted = True
    state.green_press_count += 1
    state.completion_button_attempt_history.append(name)
    checks = {
        "known_completion_button": name in COMPLETION_BUTTON_NAMES,
        "no_absorbing_failure": not state.absorbing_failure,
        "legal_core_route_complete": tuple(state.event_history) in LEGAL_CORE_ROUTES,
        "a2_complete": state.a2_complete,
        "b2_complete": state.b2_complete,
        "c2_complete": state.c2_complete,
        "no_active_drawer": state.active_drawer is None,
        "not_previously_pressed": not state.completion_button_pressed,
    }
    if all(checks.values()):
        state.completion_button_pressed = True
        state.completion_button_id = name
        state.event_history.append(StageEvent.COMPLETION_BUTTON)
        return TransitionReport(StageEvent.COMPLETION_BUTTON.value, True, None, checks)
    pending = sorted(state.pending_mask)
    reason = (
        "duplicate_completion_button_press"
        if state.completion_button_pressed
        else "premature_termination_" + "_and_".join(pending)
        if pending
        else "invalid_completion_button_press"
    )
    state.fail(reason)
    return TransitionReport(StageEvent.COMPLETION_BUTTON.value, False, reason, checks)


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
        toaster_oven_started_once=state.toaster_oven_start_event_count == 1,
        toaster_oven_stopped_once=state.toaster_oven_stop_event_count == 1,
        knob_off=float(physical.knob_qpos_rad) <= float(off_commit_max_rad),
        burner_off=not physical.target_burner_on,
        toaster_oven_off=not physical.toaster_oven_on,
        toaster_oven_door_closed=physical.toaster_oven_door_closed,
        toaster_oven_food_inside=physical.toaster_oven_food_inside,
        all_other_burners_off=physical.all_other_burners_off,
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
        pot_stable=physical.pot_stable_on_burner,
        either_completion_button_pressed=state.completion_button_pressed,
        no_absorbing_failure=not state.absorbing_failure,
    )


def diagnostics(state: SemanticState) -> dict[str, Any]:
    return {
        "a1_complete": state.a1_complete,
        "a2_complete": state.a2_complete,
        "b1_complete": state.b1_complete,
        "b2_complete": state.b2_complete,
        "c1_complete": state.c1_complete,
        "c2_complete": state.c2_complete,
        "pending_A2": state.pending_a2,
        "pending_B2": state.pending_b2,
        "pending_C2": state.pending_c2,
        "pending_mask": sorted(state.pending_mask),
        "legal_next_events": sorted(
            event.value for event in legal_next_events(state.event_history)
        ),
        "checked_drawers": sorted(state.checked_drawers),
        "completed_drawer_history": list(state.completed_drawer_history),
        "active_drawer": state.active_drawer,
        "event_history": [event.value for event in state.event_history],
        "knob_on_event_count": state.knob_on_event_count,
        "knob_off_event_count": state.knob_off_event_count,
        "toaster_oven_start_event_count": state.toaster_oven_start_event_count,
        "toaster_oven_stop_event_count": state.toaster_oven_stop_event_count,
        "toaster_oven_attempt_history": list(state.toaster_oven_attempt_history),
        "green_press_count": state.green_press_count,
        "completion_button_attempted": state.completion_button_attempted,
        "completion_button_pressed": state.completion_button_pressed,
        "completion_button_id": state.completion_button_id,
        "completion_button_attempt_history": list(
            state.completion_button_attempt_history
        ),
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
    "diagnostics",
    "evaluate_success",
    "legal_next_events",
    "mark_active_drawer_inspected",
    "record_a1",
    "record_a2",
    "record_c1",
    "record_c2",
    "record_completion_request",
    "record_drawer_completion",
    "start_drawer_cycle",
]
