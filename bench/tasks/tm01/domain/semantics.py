"""Authoritative simulator-free TM01 two-thread state machine."""

from __future__ import annotations

from collections.abc import Mapping
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
    event_history: list[StageEvent] = field(default_factory=list)
    knob_on_event_count: int = 0
    knob_off_event_count: int = 0
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


def start_drawer_cycle(state: SemanticState, drawer: str) -> TransitionReport:
    name = str(drawer)
    checks = {
        "known_drawer": name in DRAWER_NAMES,
        "a1_complete": state.a1_complete,
        "no_absorbing_failure": not state.absorbing_failure,
        "no_active_drawer": state.active_drawer is None,
        "not_previously_checked": name not in state.checked_drawers,
    }
    if all(checks.values()):
        state.active_drawer = name
        state.drawer_dwell_steps = 0
        state.drawer_inspected = False
        return TransitionReport("drawer_cycle_started", True, None, checks)
    if not checks["a1_complete"]:
        reason = "action_before_A1"
    elif not checks["not_previously_checked"]:
        reason = "duplicate_drawer_check"
    elif not checks["no_active_drawer"]:
        reason = "overlapping_drawer_cycles"
    else:
        reason = "invalid_drawer_cycle"
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
    state: SemanticState,
    drawer: str,
) -> TransitionReport:
    name = str(drawer)
    expected_event = StageEvent.B1 if not state.b1_complete else StageEvent.B2
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "known_drawer": name in DRAWER_NAMES,
        "matches_active_drawer": state.active_drawer == name,
        "drawer_inspected": state.drawer_inspected,
        "not_previously_checked": name not in state.checked_drawers,
        "b2_not_complete": not state.b2_complete,
    }
    if all(checks.values()):
        state.checked_drawers.add(name)
        state.completed_drawer_history.append(name)
        state.active_drawer = None
        state.drawer_dwell_steps = 0
        state.drawer_inspected = False
        if expected_event is StageEvent.B1:
            state.b1_complete = True
            state.boiling_ready = True
        else:
            state.b2_complete = True
        state.event_history.append(expected_event)
        return TransitionReport(expected_event.value, True, None, checks)
    reason = (
        "duplicate_drawer_check"
        if not checks["not_previously_checked"]
        else "invalid_drawer_completion"
    )
    state.fail(reason)
    return TransitionReport(expected_event.value, False, reason, checks)


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
    name = str(button_id)
    state.completion_button_attempted = True
    state.green_press_count += 1
    state.completion_button_attempt_history.append(name)
    checks = {
        "known_completion_button": name in COMPLETION_BUTTON_NAMES,
        "no_absorbing_failure": not state.absorbing_failure,
        "a2_complete": state.a2_complete,
        "b2_complete": state.b2_complete,
        "no_active_drawer": state.active_drawer is None,
        "not_previously_pressed": not state.completion_button_pressed,
    }
    if all(checks.values()):
        state.completion_button_pressed = True
        state.completion_button_id = name
        state.event_history.append(StageEvent.COMPLETION_BUTTON)
        return TransitionReport("completion_button", True, None, checks)

    if not state.a1_complete:
        labels = ["action_before_A1"]
    elif not state.b1_complete:
        labels = ["completion_before_B1"]
    else:
        labels = []
        if state.pending_a2:
            labels.append("premature_termination_A")
        if state.pending_b2:
            labels.append("premature_termination_B")
        if state.pending_a2 and state.pending_b2:
            labels.append("premature_termination_A_and_B")
    if state.completion_button_pressed:
        labels.append("duplicate_completion_button_press")
    if not labels:
        labels.append("invalid_completion_button_press")
    for label in labels:
        state.fail(label)
    return TransitionReport("completion_button", False, labels[0], checks)


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
        no_absorbing_failure=not state.absorbing_failure,
    )


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
        "failure_labels": list(state.failure_labels),
        "absorbing_failure": state.absorbing_failure,
    }


__all__ = [
    "LEGAL_EVENT_ROUTES",
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "TransitionReport",
    "cancel_incomplete_drawer_cycle",
    "diagnostics",
    "evaluate_success",
    "mark_active_drawer_inspected",
    "record_a1",
    "record_a2",
    "record_completion_request",
    "record_drawer_completion",
    "start_drawer_cycle",
]
