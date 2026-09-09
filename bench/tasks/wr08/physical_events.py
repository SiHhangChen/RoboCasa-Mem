"""Per-step physical event reconstruction for WR-08."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bench.tasks.wr08.domain.contracts import ITEM_IDS, SOURCE_SLOTS, TARGET_SLOTS
from bench.tasks.wr08.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_source_color_memory,
    record_item_picked,
    record_target_attempt,
    record_target_placement,
)


@dataclass(frozen=True)
class PhysicalSample:
    step_id: int
    item_in_source: Mapping[str, bool] = field(default_factory=dict)
    item_in_target: Mapping[str, bool] = field(default_factory=dict)
    released: Mapping[str, bool] = field(default_factory=dict)
    observed_target_slot_by_item: Mapping[str, str | None] = field(default_factory=dict)
    target_slot_by_item: Mapping[str, str] = field(default_factory=dict)
    # Legacy spellings remain accepted for serialized observers.
    item_at_plate: Mapping[str, bool] = field(default_factory=dict)
    item_in_zone: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)
    item_grasped: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.step_id) < 0:
            raise ValueError("WR-08 physical step_id must be non-negative")
        source = dict(self.item_in_source)
        target = dict(self.item_in_target)
        released = dict(self.released)
        observed = {
            str(item): (None if slot is None else str(slot))
            for item, slot in dict(self.observed_target_slot_by_item).items()
        }
        if not target and self.item_at_plate:
            target = {
                item: bool(self.item_at_plate.get(item, False)) for item in ITEM_IDS
            }
        if not source and self.item_in_zone:
            source = {
                item: any(
                    bool(self.item_in_zone.get(item, {}).get(slot, False))
                    for slot in SOURCE_SLOTS
                )
                for item in ITEM_IDS
            }
        if not released:
            released = {
                item: not bool(self.item_grasped.get(item, False)) for item in ITEM_IDS
            }
        object.__setattr__(self, "step_id", int(self.step_id))
        object.__setattr__(
            self,
            "item_in_source",
            {item: bool(source.get(item, False)) for item in ITEM_IDS},
        )
        object.__setattr__(
            self,
            "item_in_target",
            {item: bool(target.get(item, False)) for item in ITEM_IDS},
        )
        object.__setattr__(
            self,
            "released",
            {item: bool(released.get(item, False)) for item in ITEM_IDS},
        )
        object.__setattr__(
            self,
            "observed_target_slot_by_item",
            {item: observed.get(item) for item in ITEM_IDS},
        )
        target_slots = {
            str(item): str(slot)
            for item, slot in dict(self.target_slot_by_item).items()
        }
        invalid_target_slots = set(target_slots.values()) - set(TARGET_SLOTS)
        invalid_observed_slots = {
            slot for slot in observed.values() if slot is not None
        } - set(TARGET_SLOTS)
        if invalid_target_slots or invalid_observed_slots:
            raise ValueError("WR-08 physical sample contains an invalid target slot")
        object.__setattr__(
            self,
            "target_slot_by_item",
            target_slots,
        )

    def final_physical_state(self) -> PhysicalState:
        return PhysicalState(self.item_in_source, self.item_in_target, self.released)

    def outside_source(self, item: str) -> bool:
        return not bool(self.item_in_source.get(str(item), False))

    def outside_zones(self, item: str) -> bool:
        return self.outside_source(item)

    def zone_for_item(self, item: str) -> str | None:
        active = [
            slot
            for slot in SOURCE_SLOTS
            if bool(self.item_in_zone.get(str(item), {}).get(slot, False))
        ]
        return active[0] if len(active) == 1 else None


@dataclass(frozen=True)
class PhysicalUpdate:
    step_id: int
    semantic_events: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    duplicate_step: bool = False


class WR08PhysicalObserver:
    """Infer pick, color-memory, and target-placement events from physics only."""

    def __init__(
        self,
        state: SemanticState,
        *,
        transport_order: Sequence[str] | None = None,
        target_slot_by_item: Mapping[str, str] | None = None,
        dwell_steps: int = 3,
        terminal_verification_steps: int = 1,
    ) -> None:
        if int(dwell_steps) < 1 or int(terminal_verification_steps) < 1:
            raise ValueError("WR-08 physical observer dwell values must be positive")
        self.state = state
        self.transport_order = tuple(
            str(item) for item in (transport_order or ITEM_IDS)
        )
        if len(self.transport_order) != len(ITEM_IDS) or set(
            self.transport_order
        ) != set(ITEM_IDS):
            raise ValueError(
                "WR-08 transport_order must contain each item exactly once"
            )
        self.target_slot_by_item = {
            str(item): str(slot)
            for item, slot in dict(target_slot_by_item or {}).items()
        }
        self.dwell_steps = int(dwell_steps)
        self.terminal_verification_steps = int(terminal_verification_steps)
        self._previous: PhysicalSample | None = None
        self._counts: dict[str, int] = {}
        self._phase = "transport"
        self._phase_history: list[dict[str, int | str]] = []
        self._stage_first_steps: dict[str, int] = {}

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def stage_first_steps(self) -> dict[str, int]:
        return dict(self._stage_first_steps)

    def reset(self, sample: PhysicalSample) -> PhysicalUpdate:
        self._previous = sample
        self._counts.clear()
        self._phase = "transport"
        self._phase_history = [{"phase": self._phase, "step": int(sample.step_id)}]
        self._stage_first_steps.clear()
        return self._report(sample, (), False)

    def observe(self, sample: PhysicalSample) -> PhysicalUpdate:
        if self._previous is None:
            return self.reset(sample)
        previous = self._previous
        if int(sample.step_id) <= int(previous.step_id):
            return self._report(sample, (), True)

        events: list[str] = []
        for item in self.transport_order:
            pick_key = f"picked:{item}"
            grasped = bool(sample.item_grasped.get(item, False))
            # A valid grasp can lift the carton while its footprint remains
            # inside the source-plate detection radius. Stable physical grasp
            # evidence is sufficient for the event.
            pick_candidate = bool(grasped)
            if self._stable(
                pick_key, pick_candidate
            ) and not self.state.item_picked_by_item.get(item, False):
                if record_item_picked(self.state, item, physically_picked=True):
                    latch_source_color_memory(self.state, item)
                    self._mark(f"{item}_picked", sample.step_id)
                    self._mark(f"{item}_color_latched", sample.step_id)
                    events.extend((f"{item}_picked", f"{item}_color_latched"))

            expected_slot = sample.target_slot_by_item.get(
                item, self.target_slot_by_item.get(item)
            )
            current_slot = sample.observed_target_slot_by_item.get(item)
            if current_slot is None and sample.item_in_target.get(item, False):
                # Older serialized samples only carry an ``item_at_plate`` bit;
                # use the configured target as the compatibility slot.
                current_slot = (
                    expected_slot
                    or self.state.target_slot_attempted_by_item.get(item)
                    or TARGET_SLOTS[0]
                )
            target_released = bool(
                sample.item_in_target.get(item, False)
                and sample.released.get(item, False)
                and current_slot is not None
            )
            previous_slot = previous.observed_target_slot_by_item.get(item)
            if previous_slot is None and previous.item_in_target.get(item, False):
                previous_slot = (
                    expected_slot
                    or self.state.target_slot_attempted_by_item.get(item)
                    or TARGET_SLOTS[0]
                )
            previous_released = bool(
                previous.item_in_target.get(item, False)
                and previous.released.get(item, False)
                and previous_slot is not None
            )
            target_key = f"target:{item}:{current_slot}"
            target_candidate = bool(
                target_released
                and ((not previous_released) or self._counts.get(target_key, 0) > 0)
            )
            if self._stable(target_key, target_candidate):
                matches = expected_slot is not None and str(current_slot) == str(
                    expected_slot
                )
                attempted = record_target_attempt(
                    self.state, item, str(current_slot), matches=matches
                )
                if attempted and record_target_placement(
                    self.state, item, matches=True
                ):
                    self._mark(f"{item}_matched", sample.step_id)
                    events.append(f"{item}_matched")
                elif not matches:
                    events.append(f"{item}_wrong_target:{current_slot}")

        self._synchronize_phase(sample.step_id)
        self._previous = sample
        return self._report(sample, tuple(events), False)

    def _synchronize_phase(self, step_id: int) -> None:
        if all(self.state.item_placed_by_item.get(item, False) for item in ITEM_IDS):
            phase = "complete"
        elif self.state.item_picked_by_item.get(self.transport_order[1], False):
            phase = "second_item"
        elif self.state.item_picked_by_item.get(self.transport_order[0], False):
            phase = "first_item"
        else:
            phase = "transport"
        if phase != self._phase:
            self._phase = phase
            self._counts.clear()
            self._phase_history.append({"phase": phase, "step": int(step_id)})

    def _stable(self, name: str, condition: bool) -> bool:
        count = self._counts.get(name, 0) + 1 if condition else 0
        self._counts[name] = count
        return count >= self.dwell_steps

    def _mark(self, name: str, step_id: int) -> None:
        self._stage_first_steps.setdefault(str(name), int(step_id))

    def _report(
        self, sample: PhysicalSample, events: tuple[str, ...], duplicate: bool
    ) -> PhysicalUpdate:
        success = evaluate_success(
            self.state,
            sample.final_physical_state(),
            target_slot_by_item={
                item: sample.target_slot_by_item.get(
                    item, self.target_slot_by_item.get(item, "")
                )
                for item in ITEM_IDS
            },
        ).success
        return PhysicalUpdate(
            int(sample.step_id),
            events,
            {
                "phase": self._phase,
                "phase_history": list(self._phase_history),
                "dwell_steps": self.dwell_steps,
                "dwell_counts": dict(self._counts),
                "stage_first_steps": self.stage_first_steps,
                "protocol_violation": bool(self.state.protocol_violation),
                "success": bool(success),
                "item_in_source": dict(sample.item_in_source),
                "item_in_target": dict(sample.item_in_target),
                "observed_target_slot_by_item": dict(
                    sample.observed_target_slot_by_item
                ),
                "released": dict(sample.released),
            },
            bool(duplicate),
        )


__all__ = ["PhysicalSample", "PhysicalUpdate", "WR08PhysicalObserver"]
