"""Reconstruct TM04 semantic events from one physical sample per control step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .domain.contracts import ObserverConfig, SUGAR_NAMES, StageEvent
from .domain.semantics import (
    PhysicalState,
    SemanticState,
    diagnostics,
    legal_next_events,
    record_a1,
    record_a2,
    record_b1,
    record_b2,
    record_c1,
    record_c2,
    record_completion_request,
    record_pot_filled,
)


@dataclass(frozen=True)
class PhysicalSample:
    step_id: int
    faucet_on: bool
    water_button_press_edge: bool
    water_button_press_depth_m: float
    completion_button_press_edge: bool
    completion_button_press_depth_m: float
    pot_in_sink: bool
    pot_under_water: bool
    pot_stable: bool
    pot_fill_progress: float
    pot_filled: bool
    cabinet_open_fraction: float
    can_inside_cabinet: bool
    can_in_place_region: bool
    can_released: bool
    sugars_in_saucepan: Mapping[str, bool]
    sugar_released: Mapping[str, bool]
    saucepan_stable: bool


@dataclass(frozen=True)
class ObserverUpdate:
    step_id: int
    semantic_events: tuple[str, ...]
    diagnostics: dict[str, Any]


class TM04PhysicalObserver:
    """One authoritative edge observer for all three interleaved threads."""

    def __init__(self, config: ObserverConfig) -> None:
        self.config = config
        self.state = SemanticState()
        self.invalid_environment_labels: list[str] = []
        self._last: PhysicalSample | None = None
        self._pending_sugars: set[str] = set()
        self._saucepan_unstable_steps = 0
        self._max_saucepan_unstable_steps = 0

    def reset(self, initial: PhysicalSample) -> None:
        self.state = SemanticState()
        self.invalid_environment_labels = []
        self._last = initial
        self._pending_sugars = set()
        self._saucepan_unstable_steps = 0
        self._max_saucepan_unstable_steps = 0
        sugar_states = _sugar_states(initial)
        checks = {
            "faucet_initially_off": not initial.faucet_on,
            "cabinet_initially_closed": initial.cabinet_open_fraction <= self.config.cabinet_closed_threshold,
            "can_initially_in_cabinet": initial.can_inside_cabinet,
            "can_not_initially_in_place_region": not initial.can_in_place_region,
            "pot_initially_in_sink": initial.pot_in_sink,
            "pot_initially_under_water": initial.pot_under_water,
            "pot_initially_stable": initial.pot_stable,
            "pot_initially_empty": not initial.pot_filled and float(initial.pot_fill_progress) <= 0.0,
            "no_sugar_initially_in_saucepan": not any(sugar_states.values()),
            "saucepan_initially_stable": initial.saucepan_stable,
            "water_button_initially_released": initial.water_button_press_depth_m <= self.config.button_released_threshold_m,
            "completion_button_initially_released": initial.completion_button_press_depth_m <= self.config.button_released_threshold_m,
            "no_initial_press_edges": not initial.water_button_press_edge and not initial.completion_button_press_edge,
        }
        self.invalid_environment_labels.extend(name for name, passed in checks.items() if not passed)

    def observe(self, sample: PhysicalSample) -> ObserverUpdate:
        if self._last is None:
            self.reset(sample)
            return self._update(sample.step_id, ())
        if int(sample.step_id) <= int(self._last.step_id):
            return self._update(sample.step_id, ())

        previous = self._last
        semantic_events: list[str] = []

        if sample.water_button_press_edge:
            report = record_a1(self.state) if sample.faucet_on else record_a2(
                self.state, faucet_off=not sample.faucet_on
            )
            if report.passed:
                semantic_events.append(report.event)

        if sample.pot_filled and not self.state.pot_fill_ready:
            record_pot_filled(self.state)

        crossed_open = (
            previous.cabinet_open_fraction < self.config.cabinet_open_threshold
            and sample.cabinet_open_fraction >= self.config.cabinet_open_threshold
        )
        if crossed_open:
            self.state.cabinet_opened = True
        if previous.can_inside_cabinet and not sample.can_inside_cabinet:
            self.state.can_removed_from_cabinet = True

        ready_for_b1 = (
            StageEvent.B1 in legal_next_events(self.state.event_history)
            and self.state.cabinet_opened
            and self.state.can_removed_from_cabinet
            and sample.cabinet_open_fraction >= self.config.cabinet_open_threshold
            and sample.can_in_place_region
            and sample.can_released
            and sample.faucet_on
            and sample.pot_under_water
            and sample.pot_filled
            and self.state.pot_fill_ready
        )
        if ready_for_b1:
            report = record_b1(
                self.state,
                cabinet_opened=True,
                can_removed=True,
                can_in_place_region=True,
                can_released=True,
                faucet_on=True,
                pot_filled=True,
            )
            if report.passed:
                semantic_events.append(report.event)

        previous_sugars = _sugar_states(previous)
        current_sugars = _sugar_states(sample)
        entered = [
            name for name in SUGAR_NAMES
            if not previous_sugars[name] and current_sugars[name]
        ]
        if len(entered) > 1 and not self.state.absorbing_failure:
            self.state.fail("simultaneous_sugar_additions")
        else:
            self._pending_sugars.update(entered)
        if sum(current_sugars.values()) > 2:
            self.state.fail("over_add")
        ready_sugars = sorted(
            name
            for name in self._pending_sugars
            if current_sugars[name]
            and bool(sample.sugar_released.get(name, False))
            and name not in self.state.completed_sugar_names
        )
        if len(ready_sugars) > 1 and not self.state.absorbing_failure:
            self.state.fail("simultaneous_sugar_release_events")
        elif len(ready_sugars) == 1 and not self.state.absorbing_failure:
            name = ready_sugars[0]
            event = (
                StageEvent.C1
                if StageEvent.C1 in legal_next_events(self.state.event_history)
                else StageEvent.C2
            )
            report = (
                record_c1(
                    self.state,
                    name,
                    in_saucepan=True,
                    released=bool(sample.sugar_released.get(name, False)),
                )
                if event is StageEvent.C1
                else record_c2(
                    self.state,
                    name,
                    in_saucepan=True,
                    released=bool(sample.sugar_released.get(name, False)),
                )
            )
            self._pending_sugars.discard(name)
            if report.passed:
                semantic_events.append(report.event)

        crossed_closed = (
            previous.cabinet_open_fraction > self.config.cabinet_closed_threshold
            and sample.cabinet_open_fraction <= self.config.cabinet_closed_threshold
        )
        if crossed_closed and StageEvent.B2 in legal_next_events(self.state.event_history):
            report = record_b2(self.state, cabinet_closed=True)
            if report.passed:
                semantic_events.append(report.event)

        if (
            self.state.completed(StageEvent.A1)
            and not self.state.completed(StageEvent.A2)
            and not sample.faucet_on
            and not sample.water_button_press_edge
            and not self.state.absorbing_failure
        ):
            self.state.fail("unverified_faucet_interruption")
        if sum(current_sugars.values()) > 2 and not self.state.absorbing_failure:
            self.state.fail("over_add")
        if sample.saucepan_stable:
            self._saucepan_unstable_steps = 0
        else:
            self._saucepan_unstable_steps += 1
            self._max_saucepan_unstable_steps = max(
                self._max_saucepan_unstable_steps,
                self._saucepan_unstable_steps,
            )
        if (
            self._saucepan_unstable_steps
            >= self.config.saucepan_unstable_dwell_steps
            and not self.state.absorbing_failure
        ):
            self.state.fail("hot_chocolate_saucepan_moved")

        if sample.completion_button_press_edge:
            if semantic_events:
                self.state.fail("simultaneous_semantic_events")
            else:
                report = record_completion_request(self.state)
                if report.passed:
                    semantic_events.append(report.event)
        if len(semantic_events) > 1:
            self.state.fail("simultaneous_semantic_events")

        self._last = sample
        return self._update(sample.step_id, tuple(semantic_events))

    def physical_state(self) -> PhysicalState:
        if self._last is None:
            raise RuntimeError("TM04 observer has no physical sample")
        sample = self._last
        return PhysicalState(
            faucet_on=sample.faucet_on,
            pot_in_sink=sample.pot_in_sink,
            pot_under_water=sample.pot_under_water,
            pot_stable=sample.pot_stable,
            pot_filled=sample.pot_filled,
            cabinet_open_fraction=sample.cabinet_open_fraction,
            can_inside_cabinet=sample.can_inside_cabinet,
            can_in_place_region=sample.can_in_place_region,
            can_released=sample.can_released,
            sugars_in_saucepan=_sugar_states(sample),
            sugars_released=all(bool(sample.sugar_released.get(name, False)) for name in SUGAR_NAMES),
        )

    def saucepan_stability_diagnostics(self) -> dict[str, int]:
        return {
            "consecutive_unstable_steps": int(self._saucepan_unstable_steps),
            "max_consecutive_unstable_steps": int(self._max_saucepan_unstable_steps),
            "failure_dwell_steps": int(self.config.saucepan_unstable_dwell_steps),
        }

    def _update(self, step_id: int, events: tuple[str, ...]) -> ObserverUpdate:
        values = diagnostics(self.state)
        values["invalid_environment_labels"] = list(self.invalid_environment_labels)
        return ObserverUpdate(int(step_id), events, values)


def _sugar_states(sample: PhysicalSample) -> dict[str, bool]:
    return {name: bool(sample.sugars_in_saucepan.get(name, False)) for name in SUGAR_NAMES}


__all__ = ["ObserverUpdate", "PhysicalSample", "TM04PhysicalObserver"]
