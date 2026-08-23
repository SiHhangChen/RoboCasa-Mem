"""Reconstruct TM01 semantic events from one physical sample per control step."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.tm01.domain.contracts import (
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    ObserverConfig,
    StageEvent,
)
from bench.tasks.tm01.domain.semantics import (
    SemanticState,
    record_a1,
    record_a2,
    record_close_attempt,
    record_completion_button_attempt,
    record_inspection_attempt,
    record_open_attempt,
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
    pot_stable_on_burner: bool = True
    direct_knob_state_write_detected: bool = False
    knob_actuator_commanded: bool = False
    button_to_burner_bridge_detected: bool = False


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


class TM01PhysicalObserver:
    """Task-owned observer; expert handlers never submit semantic stages directly."""

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
        self._pending_drawer: str | None = None
        self._button_armed = {name: True for name in COMPLETION_BUTTON_NAMES}
        self._button_release_steps = {name: 0 for name in COMPLETION_BUTTON_NAMES}

    def reset(self, initial: PhysicalSample) -> None:
        """Arm only from the frozen all-off, all-closed initial state."""

        self.state = SemanticState()
        self.invalid_environment_labels = []
        self._last_step_id = int(initial.step_id)
        self._previous = initial
        self._knob = _KnobEvidence(
            anchor_qpos=float(initial.knob_qpos_rad),
            min_qpos=float(initial.knob_qpos_rad),
            max_qpos=float(initial.knob_qpos_rad),
        )
        self._pending_drawer = None
        self._button_armed = {
            name: float(initial.completion_button_press_depth_m.get(name, float("inf")))
            <= self.config.button_released_threshold_m
            for name in COMPLETION_BUTTON_NAMES
        }
        self._button_release_steps = {name: 0 for name in COMPLETION_BUTTON_NAMES}
        if float(initial.knob_qpos_rad) > self.config.off_commit_max_rad:
            self._invalid("reset_knob_not_off")
        if bool(initial.target_burner_on) or bool(initial.other_burners_on):
            self._invalid("reset_burner_not_off")
        for name in DRAWER_NAMES:
            if (
                float(initial.drawer_open_fraction.get(name, 1.0))
                > self.config.drawer_closed_threshold
            ):
                self._invalid("reset_drawer_not_closed")
        if not all(self._button_armed.values()):
            self._invalid("reset_completion_button_not_released")
        if not initial.pot_stable_on_burner:
            self._invalid("reset_pot_not_stable")

    def observe(self, sample: PhysicalSample) -> ObserverUpdate:
        step_id = int(sample.step_id)
        if self._last_step_id is not None and step_id <= self._last_step_id:
            return self._report(step_id, (), duplicate_step=True)
        if self._previous is None or self._knob is None:
            self.reset(sample)
            return self._report(step_id, (), duplicate_step=False)

        failures_before = tuple(self.state.failure_labels)
        self._observe_environment_integrity(sample)
        if not sample.pot_stable_on_burner:
            self.state.fail("pot_displaced")
        self._knob.observe(
            qpos=float(sample.knob_qpos_rad),
            gripper_contact=sample.knob_gripper_contact,
            any_robot_contact=sample.knob_any_robot_contact,
            maxlen=self.config.contact_evidence_window_steps,
        )

        knob_event = self._observe_knob(sample)
        drawer_event = self._observe_drawers(sample)
        button_events = self._observe_buttons(sample)
        intents = [
            value
            for value in (knob_event, drawer_event, *button_events)
            if value is not None
        ]
        committed: list[str] = []
        if len(intents) > 1:
            for _, button_id in button_events:
                assert button_id is not None
                self.state.completion_button_attempted = True
                self.state.green_press_count += 1
                self.state.completion_button_attempt_count += 1
                self.state.completion_button_press_count += 1
                self.state.completion_button_attempt_history.append(button_id)
                self.state.invalid_completion_button_press = True
            self.state.fail("simultaneous_semantic_events")
        elif (
            intents
            and not self.state.absorbing_failure
            and not self.invalid_environment_labels
        ):
            event, payload = intents[0]
            report = self._commit(event, payload, sample=sample)
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
        previous = self._previous
        assert previous is not None
        delta = abs(float(sample.knob_qpos_rad) - float(previous.knob_qpos_rad))
        physically_explained = (
            sample.knob_any_robot_contact
            or previous.knob_any_robot_contact
            or abs(float(sample.knob_qvel_rad_s)) > 1e-4
            or abs(float(previous.knob_qvel_rad_s)) > 1e-4
        )
        if (
            delta > self.config.max_unexplained_delta_rad_per_step
            and not physically_explained
        ):
            self._invalid("unexplained_knob_state_change")

    def _observe_knob(
        self,
        sample: PhysicalSample,
    ) -> tuple[StageEvent, str | None] | None:
        state = self.state
        qpos = float(sample.knob_qpos_rad)
        evidence = self._knob
        assert evidence is not None

        if sample.other_burners_on:
            state.fail("other_burner_activated")
            return None
        if state.a2_complete and sample.target_burner_on:
            state.fail("burner_reactivated_after_A2")
            return None
        if state.a1_complete and not state.b1_complete and not sample.target_burner_on:
            state.fail("early_A2")
            state.fail("burner_interrupted_before_B1")
            return None
        if (
            (state.active_drawer is not None or self._pending_drawer is not None)
            and sample.knob_gripper_contact
            and evidence.max_qpos - qpos >= 0.05
        ):
            state.fail("A2_during_active_drawer")
            return None

        target_on = not state.a1_complete
        in_target = (
            qpos >= self.config.on_commit_min_rad
            if target_on
            else qpos <= self.config.off_commit_max_rad
        )
        evidence.target_dwell_steps = (
            evidence.target_dwell_steps + 1 if in_target else 0
        )
        if evidence.target_dwell_steps < self.config.commit_dwell_steps:
            return None

        motion = (
            evidence.max_qpos - evidence.anchor_qpos
            if target_on
            else evidence.anchor_qpos - evidence.min_qpos
        )
        contact_ok = (
            evidence.gripper_contact_steps >= self.config.min_gripper_contact_steps
        )
        motion_ok = motion >= self.config.min_interaction_motion_rad
        burner_ok = (
            bool(sample.target_burner_on)
            if target_on
            else not bool(sample.target_burner_on)
        )
        if not contact_ok or not motion_ok or not burner_ok:
            if (
                evidence.any_robot_contact_steps > 0
                and evidence.gripper_contact_steps == 0
                and motion_ok
            ):
                state.fail("invalid_knob_contact")
            return None
        return (StageEvent.A1 if target_on else StageEvent.A2, None)

    def _observe_drawers(
        self,
        sample: PhysicalSample,
    ) -> tuple[StageEvent, str] | None:
        fractions = {
            name: float(sample.drawer_open_fraction.get(name, 0.0))
            for name in DRAWER_NAMES
        }
        active = self.state.active_drawer
        pending = self._pending_drawer
        if active is None and pending is None:
            candidates = [
                name
                for name, fraction in fractions.items()
                if fraction > self.config.drawer_closed_threshold
            ]
            if len(candidates) > 1:
                self.state.fail("overlapping_drawer_cycles")
                return None
            if candidates:
                if (
                    self.state.b1_complete
                    and not self.state.a2_complete
                    and not sample.target_burner_on
                ):
                    self.state.fail("burner_interrupted_before_A2")
                    return None
                pending = candidates[0]
                self._pending_drawer = pending
        elif pending is not None:
            for name, fraction in fractions.items():
                if name != pending and fraction > self.config.drawer_closed_threshold:
                    self.state.fail("overlapping_drawer_cycles")
                    return None
        else:
            assert active is not None
            for name, fraction in fractions.items():
                if name != active and fraction > self.config.drawer_closed_threshold:
                    self.state.fail("overlapping_drawer_cycles")
                    return None

        if pending is not None:
            fraction = fractions[pending]
            if fraction >= self.config.drawer_open_threshold:
                report = record_open_attempt(
                    self.state,
                    pending,
                    drawer_open=True,
                )
                self._pending_drawer = None
                if not report.passed:
                    return None
                active = pending
            elif fraction <= self.config.drawer_closed_threshold:
                record_open_attempt(
                    self.state,
                    pending,
                    drawer_open=False,
                    failure_reason="drawer_not_open",
                )
                self._pending_drawer = None
                return None

        if active is None:
            return None
        fraction = fractions[active]
        if not self.state.drawer_inspected:
            if fraction >= self.config.drawer_open_threshold:
                self.state.drawer_dwell_steps += 1
                if self.state.drawer_dwell_steps >= self.config.inspect_dwell_steps:
                    report = record_inspection_attempt(
                        self.state,
                        active,
                        drawer_open=True,
                    )
                    if not report.passed:
                        return None
            elif fraction > self.config.drawer_closed_threshold:
                self.state.drawer_dwell_steps = 0
            else:
                record_close_attempt(
                    self.state,
                    active,
                    drawer_closed=True,
                )
                return None
        if (
            self.state.active_drawer == active
            and self.state.drawer_inspected
            and fraction <= self.config.drawer_closed_threshold
        ):
            return (
                StageEvent.B1 if not self.state.b1_complete else StageEvent.B2,
                active,
            )
        return None

    def _observe_buttons(
        self,
        sample: PhysicalSample,
    ) -> tuple[tuple[StageEvent, str], ...]:
        events: list[tuple[StageEvent, str]] = []
        for name in COMPLETION_BUTTON_NAMES:
            depth = float(sample.completion_button_press_depth_m.get(name, 0.0))
            if (
                self._button_armed[name]
                and depth >= self.config.button_pressed_threshold_m
            ):
                self._button_armed[name] = False
                self._button_release_steps[name] = 0
                events.append((StageEvent.COMPLETION_BUTTON, name))
                continue
            if not self._button_armed[name]:
                if depth <= self.config.button_released_threshold_m:
                    self._button_release_steps[name] += 1
                    if (
                        self._button_release_steps[name]
                        >= self.config.button_release_dwell_steps
                    ):
                        self._button_armed[name] = True
                        self._button_release_steps[name] = 0
                else:
                    self._button_release_steps[name] = 0
        return tuple(events)

    def _commit(
        self,
        event: StageEvent,
        payload: str | None,
        *,
        sample: PhysicalSample,
    ):
        if event is StageEvent.A1:
            report = record_a1(self.state)
        elif event is StageEvent.A2:
            report = record_a2(self.state)
        elif event in (StageEvent.B1, StageEvent.B2):
            assert payload is not None
            report = record_close_attempt(
                self.state,
                payload,
                drawer_closed=True,
            )
        else:
            assert payload is not None
            report = record_completion_button_attempt(
                self.state,
                button_id=payload,
                all_drawers_closed=all(
                    float(sample.drawer_open_fraction.get(name, 1.0))
                    <= self.config.drawer_closed_threshold
                    for name in DRAWER_NAMES
                ),
                button_physically_pressed=True,
            )
        if event in (StageEvent.A1, StageEvent.A2) and report.passed:
            qpos = float(self._previous.knob_qpos_rad) if self._previous else 0.0
            if event is StageEvent.A1 and self._knob is not None:
                qpos = self._knob.max_qpos
            elif self._knob is not None:
                qpos = self._knob.min_qpos
            self._knob = _KnobEvidence(qpos, qpos, qpos)
        return report

    def _invalid(self, label: str) -> None:
        value = str(label)
        if value not in self.invalid_environment_labels:
            self.invalid_environment_labels.append(value)

    def _report(
        self,
        step_id: int,
        events: tuple[str, ...],
        *,
        duplicate_step: bool,
        new_failures: tuple[str, ...] = (),
    ) -> ObserverUpdate:
        evidence = self._knob
        diagnostics = {
            "button_armed": self._button_armed,
            "button_release_steps": self._button_release_steps,
            "knob_contact_steps": (
                0 if evidence is None else evidence.gripper_contact_steps
            ),
            "knob_any_robot_contact_steps": (
                0 if evidence is None else evidence.any_robot_contact_steps
            ),
            "invalid_environment": bool(self.invalid_environment_labels),
            "pending_drawer": self._pending_drawer,
        }
        return ObserverUpdate(
            step_id=int(step_id),
            semantic_events=tuple(events),
            failure_labels=tuple(new_failures),
            invalid_environment_labels=tuple(self.invalid_environment_labels),
            duplicate_step=bool(duplicate_step),
            diagnostics=diagnostics,
        )


__all__ = [
    "ObserverUpdate",
    "PhysicalSample",
    "TM01PhysicalObserver",
]
