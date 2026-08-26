"""Per-control-step physical event observer for TS04."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bench.tasks.ts04.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts04.domain.semantics import (
    SemanticState,
    record_close_attempt,
    record_completion_button_attempt,
    record_inspection_attempt,
    record_open_attempt,
)


@dataclass(frozen=True)
class PhysicalSample:
    step_id: int
    drawer_open_fraction: Mapping[str, float]
    completion_button_press_depth_m: float


@dataclass(frozen=True)
class ObserverUpdate:
    step_id: int
    semantic_events: tuple[str, ...]
    duplicate_step: bool
    diagnostics: Mapping[str, Any]


class TS04PhysicalObserver:
    """Infer open-inspect-close cycles and debounced button presses."""

    def __init__(
        self,
        *,
        fixture_names: tuple[str, ...] = FIXTURE_NAMES,
        target_fixture: str,
        drawer_open_threshold: float,
        drawer_closed_threshold: float,
        button_pressed_threshold_m: float,
        button_released_threshold_m: float = 0.001,
        button_release_dwell_steps: int = 2,
    ) -> None:
        if not (
            0.0 <= float(drawer_closed_threshold) < float(drawer_open_threshold) <= 1.0
        ):
            raise ValueError("TS04 observer requires closed < open thresholds")
        if not (
            0.0
            <= float(button_released_threshold_m)
            < float(button_pressed_threshold_m)
        ):
            raise ValueError("TS04 observer requires released < pressed thresholds")
        if int(button_release_dwell_steps) < 1:
            raise ValueError("TS04 button release dwell must be positive")
        self.fixture_names = tuple(str(name) for name in fixture_names)
        self.target_fixture = str(target_fixture)
        self.drawer_open_threshold = float(drawer_open_threshold)
        self.drawer_closed_threshold = float(drawer_closed_threshold)
        self.button_pressed_threshold_m = float(button_pressed_threshold_m)
        self.button_released_threshold_m = float(button_released_threshold_m)
        self.button_release_dwell_steps = int(button_release_dwell_steps)
        self.state = SemanticState(self.fixture_names, self.target_fixture)
        self.completed_fixture_steps: dict[str, int] = {}
        self.valid_button_press_steps: list[int] = []
        self._previous: PhysicalSample | None = None
        self._button_armed = True
        self._button_release_steps = 0

    def reset_state(self) -> None:
        self.state = SemanticState(self.fixture_names, self.target_fixture)
        self.completed_fixture_steps = {}
        self.valid_button_press_steps = []
        self._previous = None
        self._button_armed = True
        self._button_release_steps = 0

    def reset(self, sample: PhysicalSample) -> ObserverUpdate:
        fractions = self._fractions(sample)
        self.reset_state()
        self._previous = PhysicalSample(
            step_id=int(sample.step_id),
            drawer_open_fraction=fractions,
            completion_button_press_depth_m=float(
                sample.completion_button_press_depth_m
            ),
        )
        self._button_armed = bool(
            sample.completion_button_press_depth_m <= self.button_released_threshold_m
        )
        return self._report(sample.step_id, (), duplicate_step=False)

    def observe(self, sample: PhysicalSample) -> ObserverUpdate:
        fractions = self._fractions(sample)
        if self._previous is None:
            return self.reset(sample)
        if int(sample.step_id) <= int(self._previous.step_id):
            return self._report(sample.step_id, (), duplicate_step=True)

        previous_fractions = dict(self._previous.drawer_open_fraction)
        events: list[str] = []
        for fixture in self.fixture_names:
            crossed_open = bool(
                previous_fractions[fixture] < self.drawer_open_threshold
                and fractions[fixture] >= self.drawer_open_threshold
            )
            if crossed_open:
                report = record_open_attempt(
                    self.state,
                    fixture,
                    drawer_open=True,
                )
                if report.passed:
                    events.append(f"open:{fixture}")

        active = self.state.active_fixture
        if (
            active is not None
            and not self.state.inspected_active_fixture
            and fractions[active] >= self.drawer_open_threshold
        ):
            report = record_inspection_attempt(
                self.state,
                active,
                drawer_open=True,
            )
            if report.passed:
                events.append(f"inspect:{active}")

        active = self.state.active_fixture
        if active is not None:
            crossed_closed = bool(
                previous_fractions[active] > self.drawer_closed_threshold
                and fractions[active] <= self.drawer_closed_threshold
            )
            if crossed_closed:
                report = record_close_attempt(
                    self.state,
                    active,
                    drawer_closed=True,
                )
                if report.passed:
                    events.append(f"close:{active}")
                    self.completed_fixture_steps.setdefault(
                        str(active),
                        int(sample.step_id),
                    )

        button_depth = float(sample.completion_button_press_depth_m)
        if self._button_armed and button_depth >= self.button_pressed_threshold_m:
            self._button_armed = False
            self._button_release_steps = 0
            report = record_completion_button_attempt(
                self.state,
                all_drawers_closed=all(
                    value <= self.drawer_closed_threshold
                    for value in fractions.values()
                ),
                button_physically_pressed=True,
            )
            if report.passed:
                events.append("press:completion_button")
                self.valid_button_press_steps.append(int(sample.step_id))
        elif not self._button_armed:
            if button_depth <= self.button_released_threshold_m:
                self._button_release_steps += 1
                if self._button_release_steps >= self.button_release_dwell_steps:
                    self._button_armed = True
                    self._button_release_steps = 0
            else:
                self._button_release_steps = 0

        self._previous = PhysicalSample(
            step_id=int(sample.step_id),
            drawer_open_fraction=fractions,
            completion_button_press_depth_m=button_depth,
        )
        return self._report(sample.step_id, tuple(events), duplicate_step=False)

    def _fractions(self, sample: PhysicalSample) -> dict[str, float]:
        values = {
            str(name): float(value)
            for name, value in sample.drawer_open_fraction.items()
        }
        missing = sorted(set(self.fixture_names) - set(values))
        if missing:
            raise ValueError(
                "TS04 physical sample is missing drawers: " + ", ".join(missing)
            )
        return {name: values[name] for name in self.fixture_names}

    def _report(
        self,
        step_id: int,
        events: tuple[str, ...],
        *,
        duplicate_step: bool,
    ) -> ObserverUpdate:
        return ObserverUpdate(
            step_id=int(step_id),
            semantic_events=events,
            duplicate_step=bool(duplicate_step),
            diagnostics={
                "button_armed": self._button_armed,
                "button_release_steps": self._button_release_steps,
                "failure_reason": self.state.failure_reason,
                "completed_fixture_steps": dict(self.completed_fixture_steps),
                "valid_button_press_steps": list(self.valid_button_press_steps),
            },
        )


__all__ = ["ObserverUpdate", "PhysicalSample", "TS04PhysicalObserver"]
