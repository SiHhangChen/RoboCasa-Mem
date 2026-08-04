"""Reconstruct TM02 semantic events from one physical sample per control step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bench.tasks.tm02.domain.contracts import ObserverConfig
from bench.tasks.tm02.domain.semantics import (
    PhysicalState,
    SemanticState,
    diagnostics,
    record_a1,
    record_a2,
    record_b1,
    record_b2,
    record_cabinet_opened,
    record_can_removed,
    record_completion_request,
    record_pot_filled,
)


@dataclass(frozen=True)
class PhysicalSample:
    step_id: int
    faucet_on: bool
    water_button_press_edge: bool
    water_button_press_depth_m: float
    cabinet_open_fraction: float
    can_inside_cabinet: bool
    can_in_place_region: bool
    can_released: bool
    pot_in_sink: bool
    pot_under_water: bool
    pot_stable: bool = True
    pot_fill_progress: float = 0.0
    pot_filled: bool = False
    completion_button_press_edge: bool = False
    completion_button_press_depth_m: float = 0.0


@dataclass(frozen=True)
class ObserverUpdate:
    step_id: int
    semantic_events: tuple[str, ...]
    diagnostics: dict[str, Any]


class TM02PhysicalObserver:
    """Stateful edge observer; it never commands simulator state."""

    def __init__(self, config: ObserverConfig) -> None:
        self.config = config
        self.state = SemanticState()
        self.invalid_environment_labels: list[str] = []
        self._last: PhysicalSample | None = None

    def reset(self, initial: PhysicalSample) -> None:
        self.state = SemanticState()
        self.invalid_environment_labels = []
        self._last = initial
        checks = {
            "faucet_initially_off": not initial.faucet_on,
            "cabinet_initially_closed": (
                initial.cabinet_open_fraction
                <= self.config.cabinet_closed_threshold
            ),
            "can_initially_in_cabinet": initial.can_inside_cabinet,
            "can_not_initially_in_place_region": not initial.can_in_place_region,
            "pot_initially_in_sink": initial.pot_in_sink,
            "pot_initially_under_water": initial.pot_under_water,
            "pot_initially_stable": initial.pot_stable,
            "pot_initially_empty": (
                not initial.pot_filled
                and float(initial.pot_fill_progress) <= 0.0
            ),
            "water_button_initially_released": (
                initial.water_button_press_depth_m
                <= self.config.button_released_threshold_m
            ),
            "completion_button_initially_released": (
                initial.completion_button_press_depth_m
                <= self.config.button_released_threshold_m
            ),
            "no_initial_press_edges": (
                not initial.water_button_press_edge
                and not initial.completion_button_press_edge
            ),
        }
        self.invalid_environment_labels.extend(
            name for name, passed in checks.items() if not passed
        )

    def observe(self, sample: PhysicalSample) -> ObserverUpdate:
        if self._last is None:
            self.reset(sample)
            return self._update(sample.step_id, ())
        if int(sample.step_id) <= int(self._last.step_id):
            return self._update(sample.step_id, ())

        previous = self._last
        semantic_events: list[str] = []

        if sample.water_button_press_edge:
            report = record_a1(self.state) if sample.faucet_on else record_a2(self.state)
            if report.passed:
                semantic_events.append(report.event)

        if sample.pot_filled and not self.state.pot_fill_ready:
            record_pot_filled(self.state)

        # A1's waiting interval must remain physically active until B1 commits.
        if (
            self.state.a1_complete
            and not self.state.b1_complete
            and not sample.faucet_on
            and not self.state.absorbing_failure
        ):
            self.state.fail("faucet_interrupted_before_B1")

        crossed_open = (
            previous.cabinet_open_fraction < self.config.cabinet_open_threshold
            and sample.cabinet_open_fraction >= self.config.cabinet_open_threshold
        )
        if crossed_open:
            record_cabinet_opened(self.state)

        removed_now = previous.can_inside_cabinet and not sample.can_inside_cabinet
        if removed_now and not self.state.can_removed_from_cabinet:
            record_can_removed(self.state)

        ready_for_b1 = (
            not self.state.b1_complete
            and self.state.a1_complete
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
                faucet_on=sample.faucet_on,
                pot_under_water=sample.pot_under_water,
                pot_filled=sample.pot_filled,
                cabinet_open=True,
                can_in_place_region=sample.can_in_place_region,
                can_released=sample.can_released,
            )
            if report.passed:
                semantic_events.append(report.event)

        crossed_closed = (
            previous.cabinet_open_fraction > self.config.cabinet_closed_threshold
            and sample.cabinet_open_fraction <= self.config.cabinet_closed_threshold
        )
        if crossed_closed and self.state.cabinet_opened:
            if self.state.b1_complete and not self.state.b2_complete:
                report = record_b2(self.state)
                if report.passed:
                    semantic_events.append(report.event)
            elif not self.state.b1_complete and not self.state.absorbing_failure:
                self.state.fail("B1_incomplete")

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
            raise RuntimeError("TM02 observer has no physical sample")
        sample = self._last
        return PhysicalState(
            faucet_on=sample.faucet_on,
            pot_in_sink=sample.pot_in_sink,
            pot_under_water=sample.pot_under_water,
            pot_stable=sample.pot_stable,
            cabinet_open_fraction=sample.cabinet_open_fraction,
            can_inside_cabinet=sample.can_inside_cabinet,
            can_in_place_region=sample.can_in_place_region,
            can_released=sample.can_released,
            pot_fill_progress=float(sample.pot_fill_progress),
            pot_filled=bool(sample.pot_filled),
        )

    def _update(
        self,
        step_id: int,
        semantic_events: tuple[str, ...],
    ) -> ObserverUpdate:
        values = diagnostics(self.state)
        values["invalid_environment_labels"] = list(
            self.invalid_environment_labels
        )
        return ObserverUpdate(
            step_id=int(step_id),
            semantic_events=semantic_events,
            diagnostics=values,
        )


__all__ = ["ObserverUpdate", "PhysicalSample", "TM02PhysicalObserver"]
