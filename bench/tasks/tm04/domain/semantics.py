"""Authoritative simulator-free TM04 three-thread state machine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .contracts import ROUTE_SPECS, SUGAR_NAMES, StageEvent


LEGAL_CORE_ROUTES = frozenset(spec.events for spec in ROUTE_SPECS)
LEGAL_EVENT_ROUTES = frozenset((*route, StageEvent.COMPLETION_BUTTON) for route in LEGAL_CORE_ROUTES)
LEGAL_PREFIXES = frozenset(
    route[:length]
    for route in LEGAL_CORE_ROUTES
    for length in range(len(route) + 1)
)


@dataclass
class SemanticState:
    event_history: list[StageEvent] = field(default_factory=list)
    completed_sugar_names: list[str] = field(default_factory=list)
    pot_fill_ready: bool = False
    cabinet_opened: bool = False
    can_removed_from_cabinet: bool = False
    faucet_on_event_count: int = 0
    faucet_off_event_count: int = 0
    completion_button_attempted: bool = False
    completion_button_pressed: bool = False
    completion_button_press_count: int = 0
    failure_labels: list[str] = field(default_factory=list)
    absorbing_failure: bool = False

    def completed(self, event: StageEvent) -> bool:
        return event in self.event_history

    @property
    def pending_mask(self) -> frozenset[str]:
        return frozenset(
            event.value
            for event in (StageEvent.A2, StageEvent.B2, StageEvent.C2)
            if not self.completed(event)
            and self.completed(StageEvent(f"{event.value[0]}1"))
        )

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
    pot_filled: bool
    cabinet_open_fraction: float
    can_inside_cabinet: bool
    can_in_place_region: bool
    can_released: bool
    sugars_in_saucepan: Mapping[str, bool]
    sugars_released: bool


@dataclass(frozen=True)
class TransitionReport:
    event: str
    passed: bool
    reason: str | None
    checks: Mapping[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {"event": self.event, "pass": self.passed, "reason": self.reason, "checks": dict(self.checks)}


@dataclass(frozen=True)
class SuccessReport:
    legal_event_route: bool
    faucet_on_once: bool
    faucet_off_once: bool
    faucet_off: bool
    pot_filled_and_stable_in_sink: bool
    can_retrieved_and_released_in_place_region: bool
    cabinet_closed: bool
    exact_target_sugar_count: bool
    third_sugar_outside_saucepan: bool
    sugars_released: bool
    completion_button_pressed: bool
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
        if len(prefix) < len(route) and route[:len(prefix)] == prefix
    )


def _record_ordered(state: SemanticState, event: StageEvent, checks: Mapping[str, bool] | None = None) -> TransitionReport:
    resolved = {
        "no_absorbing_failure": not state.absorbing_failure,
        "event_is_legal_next": event in legal_next_events(state.event_history),
        **dict(checks or {}),
    }
    if all(resolved.values()):
        state.event_history.append(event)
        return TransitionReport(event.value, True, None, resolved)
    reason = f"duplicate_{event.value}" if event in state.event_history else f"invalid_{event.value}_order"
    state.fail(reason)
    return TransitionReport(event.value, False, reason, resolved)


def record_a1(state: SemanticState) -> TransitionReport:
    report = _record_ordered(state, StageEvent.A1)
    if report.passed:
        state.faucet_on_event_count += 1
    return report


def record_pot_filled(state: SemanticState) -> TransitionReport:
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "a1_complete": state.completed(StageEvent.A1),
        "not_previously_filled": not state.pot_fill_ready,
    }
    if all(checks.values()):
        state.pot_fill_ready = True
        return TransitionReport("pot_filled", True, None, checks)
    return TransitionReport("pot_filled", False, "invalid_pot_fill", checks)


def record_b1(
    state: SemanticState,
    *,
    cabinet_opened: bool,
    can_removed: bool,
    can_in_place_region: bool,
    can_released: bool,
    faucet_on: bool,
    pot_filled: bool,
) -> TransitionReport:
    state.cabinet_opened = state.cabinet_opened or bool(cabinet_opened)
    state.can_removed_from_cabinet = state.can_removed_from_cabinet or bool(can_removed)
    return _record_ordered(
        state,
        StageEvent.B1,
        {
            "cabinet_opened": state.cabinet_opened,
            "can_removed": state.can_removed_from_cabinet,
            "can_in_place_region": bool(can_in_place_region),
            "can_released": bool(can_released),
            "faucet_still_on": bool(faucet_on),
            "pot_fill_ready": state.pot_fill_ready,
            "pot_physically_filled": bool(pot_filled),
        },
    )


def record_c1(state: SemanticState, sugar_name: str, *, in_saucepan: bool, released: bool) -> TransitionReport:
    return _record_sugar(state, StageEvent.C1, sugar_name, in_saucepan=in_saucepan, released=released)


def record_a2(state: SemanticState, *, faucet_off: bool) -> TransitionReport:
    report = _record_ordered(
        state,
        StageEvent.A2,
        {"pot_fill_ready": state.pot_fill_ready, "faucet_physically_off": bool(faucet_off)},
    )
    if report.passed:
        state.faucet_off_event_count += 1
    return report


def record_b2(state: SemanticState, *, cabinet_closed: bool) -> TransitionReport:
    return _record_ordered(state, StageEvent.B2, {"cabinet_physically_closed": bool(cabinet_closed)})


def record_c2(state: SemanticState, sugar_name: str, *, in_saucepan: bool, released: bool) -> TransitionReport:
    return _record_sugar(state, StageEvent.C2, sugar_name, in_saucepan=in_saucepan, released=released)


def _record_sugar(
    state: SemanticState,
    event: StageEvent,
    sugar_name: str,
    *,
    in_saucepan: bool,
    released: bool,
) -> TransitionReport:
    name = str(sugar_name)
    report = _record_ordered(
        state,
        event,
        {
            "known_sugar": name in SUGAR_NAMES,
            "new_sugar": name not in state.completed_sugar_names,
            "sugar_in_saucepan": bool(in_saucepan),
            "sugar_released": bool(released),
            "count_below_target": len(state.completed_sugar_names) < 2,
        },
    )
    if report.passed:
        state.completed_sugar_names.append(name)
    return report


def record_completion_request(state: SemanticState) -> TransitionReport:
    state.completion_button_attempted = True
    state.completion_button_press_count += 1
    checks = {
        "no_absorbing_failure": not state.absorbing_failure,
        "legal_core_route_complete": tuple(state.event_history) in LEGAL_CORE_ROUTES,
        "exactly_two_sugar_events": len(state.completed_sugar_names) == 2,
        "not_previously_pressed": not state.completion_button_pressed,
    }
    if all(checks.values()):
        state.completion_button_pressed = True
        state.event_history.append(StageEvent.COMPLETION_BUTTON)
        return TransitionReport(StageEvent.COMPLETION_BUTTON.value, True, None, checks)
    reason = "duplicate_completion_button_press" if state.completion_button_pressed else "premature_completion_button_press"
    state.fail(reason)
    return TransitionReport(StageEvent.COMPLETION_BUTTON.value, False, reason, checks)


def evaluate_success(
    state: SemanticState,
    physical: PhysicalState,
    *,
    cabinet_closed_threshold: float = 0.08,
) -> SuccessReport:
    sugar_states = {name: bool(physical.sugars_in_saucepan.get(name, False)) for name in SUGAR_NAMES}
    added = sum(sugar_states.values())
    return SuccessReport(
        legal_event_route=tuple(state.event_history) in LEGAL_EVENT_ROUTES,
        faucet_on_once=state.faucet_on_event_count == 1,
        faucet_off_once=state.faucet_off_event_count == 1,
        faucet_off=not physical.faucet_on,
        pot_filled_and_stable_in_sink=bool(physical.pot_filled and physical.pot_in_sink and physical.pot_stable),
        can_retrieved_and_released_in_place_region=bool(not physical.can_inside_cabinet and physical.can_in_place_region and physical.can_released),
        cabinet_closed=float(physical.cabinet_open_fraction) <= float(cabinet_closed_threshold),
        exact_target_sugar_count=added == 2,
        third_sugar_outside_saucepan=added == 2,
        sugars_released=bool(physical.sugars_released),
        completion_button_pressed=state.completion_button_pressed,
        no_absorbing_failure=not state.absorbing_failure,
    )


def diagnostics(state: SemanticState) -> dict[str, Any]:
    return {
        "event_history": [event.value for event in state.event_history],
        "legal_next_events": sorted(event.value for event in legal_next_events(state.event_history)),
        "pending_mask": sorted(state.pending_mask),
        "completed_sugar_names": list(state.completed_sugar_names),
        "pot_fill_ready": state.pot_fill_ready,
        "completion_button_attempted": state.completion_button_attempted,
        "completion_button_pressed": state.completion_button_pressed,
        "failure_labels": list(state.failure_labels),
        "absorbing_failure": state.absorbing_failure,
    }


__all__ = [
    "LEGAL_CORE_ROUTES", "LEGAL_EVENT_ROUTES", "PhysicalState", "SemanticState",
    "SuccessReport", "TransitionReport", "diagnostics", "evaluate_success",
    "legal_next_events", "record_a1", "record_a2", "record_b1", "record_b2",
    "record_c1", "record_c2", "record_completion_request", "record_pot_filled",
]
