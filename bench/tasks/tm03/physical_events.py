"""Reconstruct TM03 A/B/C events from physical samples."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.tm03.domain.contracts import (
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    ObserverConfig,
    StageEvent,
)
from bench.tasks.tm03.domain.semantics import (
    SemanticState,
    cancel_incomplete_drawer_cycle,
    mark_active_drawer_inspected,
    record_a1,
    record_a2,
    record_c1,
    record_c2,
    record_completion_request,
    record_drawer_completion,
    start_drawer_cycle,
)


@dataclass(frozen=True)
class PhysicalSample:
    step_id: int
    knob_qpos_rad: float
    knob_qvel_rad_s: float
    knob_gripper_contact: bool
    knob_any_robot_contact: bool
    target_burner_on: bool
    other_burners_on: bool
    drawer_open_fraction: Mapping[str, float]
    completion_button_press_depth_m: Mapping[str, float]
    pot_stable_on_burner: bool
    toaster_oven_on: bool = False
    toaster_oven_door_closed: bool = True
    toaster_oven_food_inside: bool = True
    toaster_oven_time_qpos_rad: float = 0.0
    toaster_oven_time_qvel_rad_s: float = 0.0
    toaster_oven_time_gripper_contact: bool = False
    toaster_oven_time_any_robot_contact: bool = False
    toaster_oven_time_eef_distance_m: float = float("inf")
    direct_knob_state_write_detected: bool = False
    knob_actuator_commanded: bool = False
    button_to_burner_bridge_detected: bool = False
    direct_toaster_oven_state_write_detected: bool = False
    toaster_oven_knob_actuator_commanded: bool = False


@dataclass(frozen=True)
class ObserverUpdate:
    step_id: int
    semantic_events: tuple[str, ...]
    failure_labels: tuple[str, ...]
    invalid_environment_labels: tuple[str, ...]
    duplicate_step: bool
    diagnostics: Mapping[str, Any]


@dataclass
class _KnobEvidence:
    anchor_qpos: float
    min_qpos: float
    max_qpos: float
    target_dwell_steps: int = 0
    window: deque[tuple[bool, bool]] = field(default_factory=deque)

    def observe(
        self,
        *,
        qpos: float,
        gripper_contact: bool,
        any_robot_contact: bool,
        maxlen: int,
    ) -> None:
        self.min_qpos = min(self.min_qpos, qpos)
        self.max_qpos = max(self.max_qpos, qpos)
        self.window.append((bool(gripper_contact), bool(any_robot_contact)))
        while len(self.window) > maxlen:
            self.window.popleft()

    @property
    def gripper_contact_steps(self) -> int:
        return sum(int(gripper) for gripper, _ in self.window)

    @property
    def any_robot_contact_steps(self) -> int:
        return sum(int(contact) for _, contact in self.window)


class TM03PhysicalObserver:
    """Task-owned observer; handlers only create physical contact."""

    def __init__(
        self,
        config: ObserverConfig,
        *,
        semantic_state: SemanticState | None = None,
    ) -> None:
        self.config = config
        self.state = semantic_state or SemanticState()
        self.invalid_environment_labels: list[str] = []
        self._last_step_id: int | None = None
        self._previous: PhysicalSample | None = None
        self._knob: _KnobEvidence | None = None
        self._toaster_oven_knob: _KnobEvidence | None = None
        self._button_armed = {name: True for name in COMPLETION_BUTTON_NAMES}
        self._button_release_steps = {name: 0 for name in COMPLETION_BUTTON_NAMES}

    def reset(self, initial: PhysicalSample) -> None:
        self.state = SemanticState()
        self.invalid_environment_labels = []
        self._last_step_id = int(initial.step_id)
        self._previous = initial
        qpos = float(initial.knob_qpos_rad)
        self._knob = _KnobEvidence(qpos, qpos, qpos)
        toaster_qpos = float(initial.toaster_oven_time_qpos_rad)
        self._toaster_oven_knob = _KnobEvidence(
            toaster_qpos, toaster_qpos, toaster_qpos
        )
        self._button_armed = {
            name: float(initial.completion_button_press_depth_m.get(name, float("inf")))
            <= self.config.button_released_threshold_m
            for name in COMPLETION_BUTTON_NAMES
        }
        self._button_release_steps = {name: 0 for name in COMPLETION_BUTTON_NAMES}
        if qpos > self.config.off_commit_max_rad:
            self._invalid("reset_knob_not_off")
        if initial.target_burner_on or initial.other_burners_on:
            self._invalid("reset_burner_not_off")
        for name in DRAWER_NAMES:
            if (
                float(initial.drawer_open_fraction.get(name, 1.0))
                > self.config.drawer_closed_threshold
            ):
                self._invalid("reset_drawer_not_closed")
        if (
            initial.toaster_oven_on
            or toaster_qpos > self.config.toaster_oven_off_commit_max_rad
        ):
            self._invalid("reset_toaster_oven_not_off")
        if not initial.toaster_oven_door_closed:
            self._invalid("reset_toaster_oven_door_not_closed")
        if not initial.toaster_oven_food_inside:
            self._invalid("reset_toaster_oven_food_not_inside")
        if not all(self._button_armed.values()):
            self._invalid("reset_completion_button_not_released")
        if not initial.pot_stable_on_burner:
            self._invalid("reset_pot_not_stable")

    def observe(self, sample: PhysicalSample) -> ObserverUpdate:
        step_id = int(sample.step_id)
        if self._last_step_id is not None and step_id <= self._last_step_id:
            return self._report(step_id, (), duplicate_step=True)
        if (
            self._previous is None
            or self._knob is None
            or self._toaster_oven_knob is None
        ):
            self.reset(sample)
            return self._report(step_id, (), duplicate_step=False)

        failures_before = tuple(self.state.failure_labels)
        self._observe_environment_integrity(sample)
        self._knob.observe(
            qpos=float(sample.knob_qpos_rad),
            gripper_contact=sample.knob_gripper_contact,
            any_robot_contact=sample.knob_any_robot_contact,
            maxlen=self.config.contact_evidence_window_steps,
        )
        self._toaster_oven_knob.observe(
            qpos=float(sample.toaster_oven_time_qpos_rad),
            gripper_contact=sample.toaster_oven_time_gripper_contact,
            any_robot_contact=sample.toaster_oven_time_any_robot_contact,
            maxlen=self.config.toaster_oven_contact_evidence_window_steps,
        )
        knob_event = self._observe_knob(sample)
        drawer_event = self._observe_drawers(sample)
        toaster_oven_event = self._observe_toaster_oven(sample)
        button_events = self._observe_completion_buttons(sample)
        intents = [
            value
            for value in (knob_event, drawer_event, toaster_oven_event, *button_events)
            if value is not None
        ]
        committed: list[str] = []
        if len(intents) > 1:
            self.state.fail("simultaneous_semantic_events")
        elif intents and not self.state.absorbing_failure and not self.invalid_environment_labels:
            event, payload = intents[0]
            report = self._commit(event, payload)
            if report.passed:
                committed.append(event.value)

        self._last_step_id = step_id
        self._previous = sample
        new_failures = tuple(
            label for label in self.state.failure_labels if label not in failures_before
        )
        return self._report(
            step_id,
            tuple(committed),
            duplicate_step=False,
            new_failures=new_failures,
        )

    def _observe_environment_integrity(self, sample: PhysicalSample) -> None:
        if sample.direct_knob_state_write_detected:
            self._invalid("nonphysical_knob_actuation")
        if sample.knob_actuator_commanded:
            self._invalid("knob_actuator_shortcut")
        if sample.button_to_burner_bridge_detected:
            self._invalid("button_to_burner_bridge")
        if sample.direct_toaster_oven_state_write_detected:
            self._invalid("nonphysical_toaster_oven_actuation")
        if sample.toaster_oven_knob_actuator_commanded:
            self._invalid("toaster_oven_knob_actuator_shortcut")
        if not sample.pot_stable_on_burner:
            self.state.fail("pot_displaced")
        if not sample.toaster_oven_door_closed:
            self.state.fail("toaster_oven_door_opened")
        if not sample.toaster_oven_food_inside:
            self.state.fail("toaster_oven_food_displaced")
        if sample.other_burners_on:
            self.state.fail("other_burner_activated")
        previous = self._previous
        assert previous is not None
        delta = abs(float(sample.knob_qpos_rad) - float(previous.knob_qpos_rad))
        explained = (
            sample.knob_any_robot_contact
            or previous.knob_any_robot_contact
            or abs(float(sample.knob_qvel_rad_s)) > 1e-4
            or abs(float(previous.knob_qvel_rad_s)) > 1e-4
        )
        if delta > self.config.max_unexplained_delta_rad_per_step and not explained:
            self._invalid("unexplained_knob_state_change")
        toaster_delta = abs(
            float(sample.toaster_oven_time_qpos_rad)
            - float(previous.toaster_oven_time_qpos_rad)
        )
        toaster_explained = (
            sample.toaster_oven_time_any_robot_contact
            or previous.toaster_oven_time_any_robot_contact
            or abs(float(sample.toaster_oven_time_qvel_rad_s)) > 1e-4
            or abs(float(previous.toaster_oven_time_qvel_rad_s)) > 1e-4
        )
        if (
            toaster_delta
            > self.config.toaster_oven_max_unexplained_delta_rad_per_step
            and not toaster_explained
        ):
            self._invalid("unexplained_toaster_oven_time_knob_state_change")

    def _observe_knob(
        self, sample: PhysicalSample
    ) -> tuple[StageEvent, str | None] | None:
        state = self.state
        evidence = self._knob
        assert evidence is not None
        if state.a2_complete:
            if sample.target_burner_on:
                state.fail("burner_reactivated_after_A2")
            return None
        target_on = not state.a1_complete
        qpos = float(sample.knob_qpos_rad)
        in_target = (
            qpos >= self.config.on_commit_min_rad
            if target_on
            else qpos <= self.config.off_commit_max_rad
        )
        evidence.target_dwell_steps = evidence.target_dwell_steps + 1 if in_target else 0
        if evidence.target_dwell_steps < self.config.commit_dwell_steps:
            return None
        motion = (
            evidence.max_qpos - evidence.anchor_qpos
            if target_on
            else evidence.anchor_qpos - evidence.min_qpos
        )
        contact_ok = evidence.gripper_contact_steps >= self.config.min_gripper_contact_steps
        motion_ok = motion >= self.config.min_interaction_motion_rad
        state_ok = sample.target_burner_on if target_on else not sample.target_burner_on
        if not (contact_ok and motion_ok and state_ok):
            if (
                evidence.any_robot_contact_steps
                and not evidence.gripper_contact_steps
                and motion_ok
            ):
                state.fail("invalid_knob_contact")
            return None
        return (StageEvent.A1 if target_on else StageEvent.A2, None)

    def _observe_drawers(
        self, sample: PhysicalSample
    ) -> tuple[StageEvent, str] | None:
        fractions = {
            name: float(sample.drawer_open_fraction.get(name, 0.0))
            for name in DRAWER_NAMES
        }
        active = self.state.active_drawer
        if active is None:
            candidates = [
                name
                for name, fraction in fractions.items()
                if fraction > self.config.drawer_closed_threshold
            ]
            if len(candidates) > 1:
                self.state.fail("overlapping_drawer_cycles")
                return None
            if candidates:
                report = start_drawer_cycle(self.state, candidates[0])
                if not report.passed:
                    return None
                active = candidates[0]
        else:
            if any(
                name != active and fraction > self.config.drawer_closed_threshold
                for name, fraction in fractions.items()
            ):
                self.state.fail("overlapping_drawer_cycles")
                return None
        if active is None:
            return None
        fraction = fractions[active]
        if not self.state.drawer_inspected:
            if fraction >= self.config.drawer_open_threshold:
                self.state.drawer_dwell_steps += 1
                if self.state.drawer_dwell_steps >= self.config.inspect_dwell_steps:
                    mark_active_drawer_inspected(self.state)
            elif fraction > self.config.drawer_closed_threshold:
                self.state.drawer_dwell_steps = 0
            else:
                cancel_incomplete_drawer_cycle(self.state)
                return None
        if self.state.drawer_inspected and fraction <= self.config.drawer_closed_threshold:
            event = StageEvent.B1 if not self.state.b1_complete else StageEvent.B2
            return (event, active)
        return None

    def _observe_toaster_oven(
        self, sample: PhysicalSample
    ) -> tuple[StageEvent, str | None] | None:
        state = self.state
        evidence = self._toaster_oven_knob
        assert evidence is not None
        if state.c2_complete:
            if sample.toaster_oven_on:
                state.fail("toaster_oven_reactivated_after_C2")
            return None
        target_on = not state.c1_complete
        qpos = float(sample.toaster_oven_time_qpos_rad)
        in_target = (
            qpos >= self.config.toaster_oven_on_commit_min_rad
            if target_on
            else qpos <= self.config.toaster_oven_off_commit_max_rad
        )
        evidence.target_dwell_steps = (
            evidence.target_dwell_steps + 1 if in_target else 0
        )
        if (
            evidence.target_dwell_steps
            < self.config.toaster_oven_commit_dwell_steps
        ):
            return None
        motion = (
            evidence.max_qpos - evidence.anchor_qpos
            if target_on
            else evidence.anchor_qpos - evidence.min_qpos
        )
        contact_ok = (
            evidence.gripper_contact_steps
            >= self.config.toaster_oven_min_gripper_contact_steps
        )
        motion_ok = (
            motion >= self.config.toaster_oven_min_interaction_motion_rad
        )
        state_ok = sample.toaster_oven_on if target_on else not sample.toaster_oven_on
        if not (contact_ok and motion_ok and state_ok):
            if (
                evidence.any_robot_contact_steps
                and not evidence.gripper_contact_steps
                and motion_ok
            ):
                state.fail("invalid_toaster_oven_time_knob_contact")
            return None
        return (StageEvent.C1 if target_on else StageEvent.C2, None)

    def _observe_completion_buttons(
        self, sample: PhysicalSample
    ) -> tuple[tuple[StageEvent, str], ...]:
        events: list[tuple[StageEvent, str]] = []
        for name in COMPLETION_BUTTON_NAMES:
            depth = float(sample.completion_button_press_depth_m.get(name, 0.0))
            if self._button_armed[name] and depth >= self.config.button_pressed_threshold_m:
                self._button_armed[name] = False
                self._button_release_steps[name] = 0
                events.append((StageEvent.COMPLETION_BUTTON, name))
            elif not self._button_armed[name]:
                if depth <= self.config.button_released_threshold_m:
                    self._button_release_steps[name] += 1
                    if self._button_release_steps[name] >= self.config.button_release_dwell_steps:
                        self._button_armed[name] = True
                        self._button_release_steps[name] = 0
                else:
                    self._button_release_steps[name] = 0
        return tuple(events)

    def _commit(self, event: StageEvent, payload: str | None):
        if event is StageEvent.A1:
            report = record_a1(self.state)
        elif event is StageEvent.A2:
            report = record_a2(self.state)
        elif event in (StageEvent.B1, StageEvent.B2):
            assert payload is not None
            report = record_drawer_completion(self.state, payload)
        elif event is StageEvent.C1:
            report = record_c1(self.state)
        elif event is StageEvent.C2:
            report = record_c2(self.state)
        else:
            assert payload is not None
            report = record_completion_request(self.state, payload)
        if event in (StageEvent.A1, StageEvent.A2) and report.passed:
            qpos = float(self._previous.knob_qpos_rad) if self._previous else 0.0
            if self._knob is not None:
                qpos = self._knob.max_qpos if event is StageEvent.A1 else self._knob.min_qpos
            self._knob = _KnobEvidence(qpos, qpos, qpos)
        if event in (StageEvent.C1, StageEvent.C2) and report.passed:
            qpos = (
                float(self._previous.toaster_oven_time_qpos_rad)
                if self._previous
                else 0.0
            )
            if self._toaster_oven_knob is not None:
                qpos = (
                    self._toaster_oven_knob.max_qpos
                    if event is StageEvent.C1
                    else self._toaster_oven_knob.min_qpos
                )
            self._toaster_oven_knob = _KnobEvidence(qpos, qpos, qpos)
        return report

    def _invalid(self, label: str) -> None:
        if label not in self.invalid_environment_labels:
            self.invalid_environment_labels.append(label)

    def _report(
        self,
        step_id: int,
        events: tuple[str, ...],
        *,
        duplicate_step: bool,
        new_failures: tuple[str, ...] = (),
    ) -> ObserverUpdate:
        return ObserverUpdate(
            step_id=int(step_id),
            semantic_events=events,
            failure_labels=new_failures,
            invalid_environment_labels=tuple(self.invalid_environment_labels),
            duplicate_step=bool(duplicate_step),
            diagnostics={
                "toaster_oven_time_knob_qpos_rad": (
                    None
                    if self._previous is None
                    else float(self._previous.toaster_oven_time_qpos_rad)
                ),
                "invalid_environment": bool(self.invalid_environment_labels),
            },
        )


__all__ = ["ObserverUpdate", "PhysicalSample", "TM03PhysicalObserver"]
