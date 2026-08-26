"""Per-control-step physical event observer for WR-06.

WR-06 follows WR-04's strict single-pass, dwell-based state machine, with the
staging fixture replaced by the fixed serving plate beside the island counter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bench.tasks.wr06.domain.contracts import ITEM_ID
from bench.tasks.wr06.domain.episode import validate_level
from bench.tasks.wr06.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_memory_barrier,
    record_item_removed,
    record_return_attempt,
    record_return_completion,
    record_stage_regression,
)


@dataclass(frozen=True)
class PhysicalSample:
    step_id: int
    item_at_fixed_plate: bool
    item_in_lower: bool
    item_in_upper: bool
    item_supported_by_lower: bool
    item_supported_by_upper: bool
    # Position alone cannot distinguish a successful pick from an open-
    # gripper collision that bumps the bottle off the plate.
    item_grasped: bool = False

    @property
    def item_outside_shelves(self) -> bool:
        return not self.item_in_lower and not self.item_in_upper

    def item_in_level(self, level: str) -> bool:
        return self.item_in_lower if validate_level(level) == "shelf_lower" else self.item_in_upper

    def item_supported_in_level(self, level: str) -> bool:
        return (
            self.item_supported_by_lower
            if validate_level(level) == "shelf_lower"
            else self.item_supported_by_upper
        )

    def final_physical_state(self) -> PhysicalState:
        return PhysicalState(
            item_at_fixed_plate=self.item_at_fixed_plate,
            item_in_lower=self.item_in_lower,
            item_in_upper=self.item_in_upper,
            item_supported_by_lower=self.item_supported_by_lower,
            item_supported_by_upper=self.item_supported_by_upper,
        )


@dataclass(frozen=True)
class ObserverUpdate:
    step_id: int
    semantic_events: tuple[str, ...]
    duplicate_step: bool
    diagnostics: Mapping[str, Any]


class WR06PhysicalObserver:
    """Strict single-pass WR-06 state machine driven only by object state."""

    HOME = "home"
    TO_PLATE = "to_plate"
    AT_PLATE = "at_plate"
    TO_RETURN = "to_return"
    VERIFY_RETURN = "verify_return"
    COMPLETE = "complete"
    INVALID = "invalid"
    _STAGES = ("pickup", "staged", "barrier", "return_correct", "returned", "success")

    def __init__(
        self,
        state: SemanticState,
        *,
        dwell_steps: int = 3,
        terminal_verification_steps: int = 100,
    ) -> None:
        if int(dwell_steps) < 1:
            raise ValueError("WR-06 physical-event dwell must be positive")
        if state.initial_level is None or not state.initial_level_physically_verified:
            raise ValueError("WR-06 observer requires a verified initial level")
        if int(terminal_verification_steps) < 1:
            raise ValueError("WR-06 terminal verification must be positive")
        self.state = state
        self.dwell_steps = int(dwell_steps)
        self.terminal_verification_steps = int(terminal_verification_steps)
        self._previous: PhysicalSample | None = None
        self._counts: dict[str, int] = {}
        self._stage_first_steps: dict[str, int] = {}
        self._phase = self.HOME
        self._phase_history: list[dict[str, int | str]] = []
        self._violation_reason: str | None = None
        self._verification_progress = 0
        self._return_verification_armed = False
        self._plate_pick_confirmed = False
        self._plate_pick_release_seen = False

    @property
    def stage_first_steps(self) -> dict[str, int]:
        return dict(self._stage_first_steps)

    def begin_return_placement(self) -> bool:
        """Hold return verification until the final release and retreat finish."""

        if self._phase != self.TO_RETURN:
            return False
        self._return_verification_armed = False
        self._counts.pop("return:shelf_lower", None)
        self._counts.pop("return:shelf_upper", None)
        return True

    def arm_return_verification(self) -> bool:
        """Allow the next physical shelf contact to start terminal validation."""

        if self._phase != self.TO_RETURN:
            return False
        self._return_verification_armed = True
        return True

    def reset(self, sample: PhysicalSample) -> ObserverUpdate:
        self._previous = sample
        self._counts.clear()
        self._stage_first_steps.clear()
        self._phase = self.HOME
        self._phase_history = [{"phase": self.HOME, "step": int(sample.step_id)}]
        self._violation_reason = None
        self._verification_progress = 0
        self._return_verification_armed = False
        self._plate_pick_confirmed = False
        self._plate_pick_release_seen = False
        self._capture_latched_stages(sample)
        return self._report(sample.step_id, (), duplicate_step=False)

    def observe(self, sample: PhysicalSample) -> ObserverUpdate:
        if self._previous is None:
            return self.reset(sample)
        if int(sample.step_id) <= int(self._previous.step_id):
            return self._report(sample.step_id, (), duplicate_step=True)

        events: list[str] = []
        original = validate_level(self.state.initial_level)

        if self._phase == self.HOME:
            if self._stable("leave_home", sample.item_outside_shelves):
                if record_item_removed(self.state, physically_removed=True):
                    events.append("pickup")
                    self._transition(self.TO_PLATE, sample.step_id)

        elif self._phase == self.TO_PLATE:
            supported = self._supported_level(sample)
            # Passing through the plate target while still grasped is transport,
            # not completed staging. Qualify the stage only after a stable
            # physical release in the configured plate region.
            if self._stable(
                "plate_first_release",
                sample.item_at_fixed_plate and not sample.item_grasped,
            ):
                latch_memory_barrier(
                    self.state,
                    item_at_fixed_plate=True,
                    item_outside_shelves=True,
                )
                self._plate_pick_release_seen = True
                events.extend(("staged", "barrier"))
                self._transition(self.AT_PLATE, sample.step_id)
            elif supported is not None and self._stable(
                f"premature_shelf:{supported}", True
            ):
                self._invalidate("shelf_reentered_before_plate", sample.step_id)
                events.append("violation:shelf_reentered_before_plate")

        elif self._phase == self.AT_PLATE:
            # The bottle is still held while it is first transported onto the
            # plate. Require a real release after staging before accepting a
            # later grasp as the plate-side re-pick.
            if not sample.item_grasped:
                self._plate_pick_release_seen = True
            if self._plate_pick_release_seen and self._stable(
                "plate_pick_grasp", sample.item_grasped
            ):
                self._plate_pick_confirmed = True
            if self._stable(
                "leave_plate",
                sample.item_outside_shelves
                and not sample.item_at_fixed_plate
                and self._plate_pick_release_seen
                and (self._plate_pick_confirmed or sample.item_grasped),
            ):
                self._transition(self.TO_RETURN, sample.step_id)
                events.append("leave_plate")

        elif self._phase == self.TO_RETURN:
            supported = self._supported_level(sample)
            if self._stable("plate_reentry", sample.item_at_fixed_plate):
                self._invalidate("plate_reentered_after_departure", sample.step_id)
                events.append("violation:plate_reentered_after_departure")
            else:
                # Recorded-action replay does not execute expert handlers, so
                # return verification cannot depend on an out-of-band handler
                # call. A released bottle physically supported by a shelf is
                # the action-derived boundary that arms the return check.
                if (
                    not self._return_verification_armed
                    and supported is not None
                    and not sample.item_grasped
                ):
                    self._return_verification_armed = True
                    events.append("return_verification_armed")
                return_stable = bool(
                    self._return_verification_armed
                    and supported is not None
                    and self._stable(f"return:{supported}", True)
                )
            if self._phase == self.TO_RETURN and return_stable:
                if record_return_attempt(self.state, supported):
                    events.append("return_correct")
                    self._transition(self.VERIFY_RETURN, sample.step_id)
                else:
                    self._invalidate("wrong_return_level", sample.step_id)
                    events.append(f"violation:wrong_return_level:{supported}")

        elif self._phase == self.VERIFY_RETURN:
            remains_returned = bool(
                sample.item_in_level(original)
                and sample.item_supported_in_level(original)
            )
            if remains_returned:
                self._verification_progress += 1
                self._counts.pop("terminal_disturbance", None)
                if self._verification_progress >= self.terminal_verification_steps:
                    record_return_completion(
                        self.state,
                        ITEM_ID,
                        item_in_original_level=True,
                    )
                    events.extend(("returned", "terminal_verified"))
                    self._transition(self.COMPLETE, sample.step_id)
            else:
                self._verification_progress = 0
                if self._stable("terminal_disturbance", True):
                    self._invalidate(
                        "item_disturbed_during_terminal_verification",
                        sample.step_id,
                    )
                    events.append(
                        "violation:item_disturbed_during_terminal_verification"
                    )

        elif self._phase == self.COMPLETE:
            remains_returned = bool(
                sample.item_in_level(original)
                and sample.item_supported_in_level(original)
            )
            if self._stable("leave_complete", not remains_returned):
                self._invalidate("item_removed_after_success", sample.step_id)
                events.append("violation:item_removed_after_success")

        self._previous = sample
        self._capture_latched_stages(sample)
        return self._report(sample.step_id, tuple(events), duplicate_step=False)

    def _stable(self, name: str, condition: bool) -> bool:
        count = self._counts.get(name, 0) + 1 if condition else 0
        self._counts[name] = count
        return count >= self.dwell_steps

    def _transition(self, phase: str, step_id: int) -> None:
        if phase == self._phase:
            return
        self._phase = str(phase)
        self._counts.clear()
        self._phase_history.append({"phase": self._phase, "step": int(step_id)})

    def _invalidate(self, reason: str, step_id: int) -> None:
        record_stage_regression(self.state, kind=reason)
        self._violation_reason = str(reason)
        self._transition(self.INVALID, step_id)

    @staticmethod
    def _supported_level(sample: PhysicalSample) -> str | None:
        supported = [
            level
            for level in ("shelf_lower", "shelf_upper")
            if sample.item_in_level(level) and sample.item_supported_in_level(level)
        ]
        if len(supported) != 1:
            return None
        return supported[0]

    def _capture_latched_stages(self, sample: PhysicalSample) -> None:
        original = validate_level(self.state.initial_level)
        values = {
            "pickup": self.state.item_removed_from_home,
            "staged": self.state.item_staged_at_plate,
            "barrier": self.state.memory_barrier_latched,
            "return_correct": self.state.first_return_level == original,
            "returned": self.state.returned_item_id == ITEM_ID,
            "success": evaluate_success(
                self.state,
                sample.final_physical_state(),
            ).success,
        }
        for name in self._STAGES:
            if values[name] and name not in self._stage_first_steps:
                self._stage_first_steps[name] = int(sample.step_id)

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
                "dwell_steps": self.dwell_steps,
                "dwell_counts": dict(self._counts),
                "stage_first_steps": self.stage_first_steps,
                "stage_regressions": list(self.state.stage_regressions),
                "phase": self._phase,
                "phase_history": list(self._phase_history),
                "violation_reason": self._violation_reason,
                "terminal_verification_steps": self.terminal_verification_steps,
                "terminal_verification_progress": self._verification_progress,
                "return_verification_armed": self._return_verification_armed,
                "plate_pick_confirmed": self._plate_pick_confirmed,
                "plate_pick_release_seen": self._plate_pick_release_seen,
            },
        )


__all__ = ["ObserverUpdate", "PhysicalSample", "WR06PhysicalObserver"]
