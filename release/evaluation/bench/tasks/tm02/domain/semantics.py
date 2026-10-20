"""Authoritative simulator-free TM02 two-thread state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.tasks.tm02.domain.contracts import StageEvent


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
    pot_fill_ready: bool = False
    cabinet_opened: bool = False
    can_removed_from_cabinet: bool = False
    event_history: list[StageEvent] = field(default_factory=list)
    faucet_on_event_count: int = 0
    faucet_off_event_count: int = 0
    completion_button_attempted: bool = False
    completion_button_pressed: bool = False
    completion_button_press_count: int = 0
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
    faucet_on: bool
    pot_in_sink: bool
    pot_under_water: bool
    pot_stable: bool
    cabinet_open_fraction: float
    can_inside_cabinet: bool
    can_in_place_region: bool
    can_released: bool
    pot_fill_progress: float = 0.0
    pot_filled: bool = False


@dataclass(frozen=True)
class TransitionReport:
    event: str
    passed: bool
    reason: str | None
    checks: dict[str, bool]

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
    faucet_on_once: bool
    faucet_off_once: bool
    faucet_off: bool
    pot_filled: bool
    pot_stable_in_sink: bool
    can_retrieved: bool
    can_released_in_place_region: bool
    cabinet_closed: bool
    completion_button_pressed: bool
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
    }
    if all(checks.values()):
        state.a1_complete = True
        state.faucet_on_event_count += 1
        state.event_history.append(StageEvent.A1)
        return TransitionReport("A1", True, None, checks)
    reason = "duplicate_A1" if state.a1_complete else "invalid_A1_order"
    state.fail(reason)
    return TransitionReport("A1", False, reason, checks)


def record_cabinet_opened(state: SemanticState) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "a1_complete": state.a1_complete,
        "b1_not_complete": not state.b1_complete,
        "b2_not_complete": not state.b2_complete,
    }
    if all(checks.values()):
        state.cabinet_opened = True
        return TransitionReport("cabinet_opened", True, None, checks)
    if not state.a1_complete:
        reason = "action_before_A1"
    elif state.b2_complete:
        reason = "cabinet_reopened_after_B2"
    else:
        reason = "invalid_cabinet_open"
    state.fail(reason)
    return TransitionReport("cabinet_opened", False, reason, checks)


def record_can_removed(state: SemanticState) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "cabinet_opened": state.cabinet_opened,
        "b1_not_complete": not state.b1_complete,
        "not_previously_removed": not state.can_removed_from_cabinet,
    }
    if all(checks.values()):
        state.can_removed_from_cabinet = True
        return TransitionReport("can_removed", True, None, checks)
    reason = (
        "can_removed_before_cabinet_open"
        if not state.cabinet_opened
        else "duplicate_can_removal"
    )
    state.fail(reason)
    return TransitionReport("can_removed", False, reason, checks)


def record_pot_filled(state: SemanticState) -> TransitionReport:
    """Latch physical fill completion without adding a workflow route event."""

    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "a1_complete": state.a1_complete,
        "pot_not_previously_filled": not state.pot_fill_ready,
    }
    if all(checks.values()):
        state.pot_fill_ready = True
        return TransitionReport("pot_filled", True, None, checks)
    reason = (
        "pot_filled_before_A1"
        if not state.a1_complete
        else "duplicate_pot_filled"
    )
    return TransitionReport("pot_filled", False, reason, checks)


def record_b1(
    state: SemanticState,
    *,
    faucet_on: bool,
    pot_under_water: bool,
    pot_filled: bool,
    cabinet_open: bool,
    can_in_place_region: bool,
    can_released: bool,
) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "a1_complete": state.a1_complete,
        "b1_not_complete": not state.b1_complete,
        "cabinet_opened": state.cabinet_opened,
        "can_removed_from_cabinet": state.can_removed_from_cabinet,
        "cabinet_still_open": bool(cabinet_open),
        "can_in_place_region": bool(can_in_place_region),
        "can_released": bool(can_released),
        "faucet_still_on": bool(faucet_on),
        "pot_under_water": bool(pot_under_water),
        "pot_fill_ready": state.pot_fill_ready,
        "pot_physically_filled": bool(pot_filled),
    }
    if all(checks.values()):
        state.b1_complete = True
        state.event_history.append(StageEvent.B1)
        return TransitionReport("B1", True, None, checks)
    reason = "duplicate_B1" if state.b1_complete else "B1_incomplete"
    state.fail(reason)
    return TransitionReport("B1", False, reason, checks)


def record_a2(state: SemanticState) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "a1_complete": state.a1_complete,
        "b1_complete": state.b1_complete,
        "pot_fill_ready": state.pot_fill_ready,
        "a2_not_complete": not state.a2_complete,
    }
    if all(checks.values()):
        state.a2_complete = True
        state.faucet_off_event_count += 1
        state.event_history.append(StageEvent.A2)
        return TransitionReport("A2", True, None, checks)
    if state.a2_complete:
        reason = "duplicate_A2"
    elif not state.b1_complete:
        reason = "early_A2"
    else:
        reason = "invalid_A2_order"
    state.fail(reason)
    return TransitionReport("A2", False, reason, checks)


def record_b2(state: SemanticState) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "b1_complete": state.b1_complete,
        "cabinet_opened": state.cabinet_opened,
        "b2_not_complete": not state.b2_complete,
    }
    if all(checks.values()):
        state.b2_complete = True
        state.event_history.append(StageEvent.B2)
        return TransitionReport("B2", True, None, checks)
    reason = "duplicate_B2" if state.b2_complete else "B2_before_B1"
    state.fail(reason)
    return TransitionReport("B2", False, reason, checks)


def record_completion_request(state: SemanticState) -> TransitionReport:
    """Commit the final event only after both task threads have terminated."""

    state.completion_button_attempted = True
    state.completion_button_press_count += 1
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "a1_complete": state.a1_complete,
        "a2_complete": state.a2_complete,
        "b1_complete": state.b1_complete,
        "b2_complete": state.b2_complete,
        "not_previously_pressed": not state.completion_button_pressed,
    }
    if all(checks.values()):
        state.completion_button_pressed = True
        state.event_history.append(StageEvent.COMPLETION_BUTTON)
        return TransitionReport("completion_button", True, None, checks)
    if state.completion_button_pressed:
        reason = "duplicate_completion_button_press"
    else:
        reason = "premature_completion_button_press"
    state.fail(reason)
    return TransitionReport("completion_button", False, reason, checks)


def evaluate_success(
    state: SemanticState,
    physical: PhysicalState,
    *,
    cabinet_closed_threshold: float = 0.08,
) -> SuccessReport:
    return SuccessReport(
        legal_event_route=tuple(state.event_history) in LEGAL_EVENT_ROUTES,
        faucet_on_once=state.faucet_on_event_count == 1,
        faucet_off_once=state.faucet_off_event_count == 1,
        faucet_off=not bool(physical.faucet_on),
        pot_filled=bool(state.pot_fill_ready and physical.pot_filled),
        pot_stable_in_sink=(
            bool(physical.pot_in_sink)
            and bool(physical.pot_under_water)
            and bool(physical.pot_stable)
        ),
        can_retrieved=(
            state.can_removed_from_cabinet
            and not bool(physical.can_inside_cabinet)
        ),
        can_released_in_place_region=(
            bool(physical.can_in_place_region) and bool(physical.can_released)
        ),
        cabinet_closed=(
            float(physical.cabinet_open_fraction)
            <= float(cabinet_closed_threshold)
        ),
        completion_button_pressed=state.completion_button_pressed,
        no_absorbing_failure=not state.absorbing_failure,
    )


def diagnostics(state: SemanticState) -> dict[str, Any]:
    return {
        "a1_complete": state.a1_complete,
        "a2_complete": state.a2_complete,
        "b1_complete": state.b1_complete,
        "b2_complete": state.b2_complete,
        "pot_fill_ready": state.pot_fill_ready,
        "pending_A2": state.pending_a2,
        "pending_B2": state.pending_b2,
        "pending_mask": sorted(state.pending_mask),
        "cabinet_opened": state.cabinet_opened,
        "can_removed_from_cabinet": state.can_removed_from_cabinet,
        "event_history": [event.value for event in state.event_history],
        "faucet_on_event_count": state.faucet_on_event_count,
        "faucet_off_event_count": state.faucet_off_event_count,
        "completion_button_attempted": state.completion_button_attempted,
        "completion_button_pressed": state.completion_button_pressed,
        "completion_button_press_count": state.completion_button_press_count,
        "failure_labels": list(state.failure_labels),
        "absorbing_failure": state.absorbing_failure,
    }


__all__ = [
    "LEGAL_EVENT_ROUTES",
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "TransitionReport",
    "diagnostics",
    "evaluate_success",
    "record_a1",
    "record_a2",
    "record_b1",
    "record_b2",
    "record_cabinet_opened",
    "record_can_removed",
    "record_completion_request",
    "record_pot_filled",
]
