"""WR-07 semantic events for color memory and matching placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from bench.tasks.wr07.domain.contracts import SOURCE_SLOTS, TARGET_SLOTS

@dataclass
class SemanticState:
    source_slot: str | None = None
    source_color: str | None = None
    initial_source_verified: bool = False
    item_picked: bool = False
    source_color_memory_latched: bool = False
    target_slot_attempted: str | None = None
    item_placed: bool = False
    target_color_matches: bool = False
    protocol_violation: bool = False
    stage_regressions: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class PhysicalState:
    item_in_source: Mapping[str, bool]
    item_in_target: Mapping[str, bool]
    item_supported: bool
    item_released: bool

@dataclass(frozen=True)
class SuccessReport:
    initial_source_verified: bool
    item_picked: bool
    source_color_memory_latched: bool
    item_placed: bool
    target_color_matches: bool
    no_wrong_target_placement: bool
    no_protocol_violation: bool
    @property
    def success(self) -> bool:
        return all(self.checks().values())
    def checks(self) -> dict[str, bool]:
        return {
            "initial_source_verified": bool(self.initial_source_verified),
            "item_picked": bool(self.item_picked),
            "source_color_memory_latched": bool(self.source_color_memory_latched),
            "item_placed": bool(self.item_placed),
            "target_color_matches": bool(self.target_color_matches),
            "no_wrong_target_placement": bool(self.no_wrong_target_placement),
            "no_protocol_violation": bool(self.no_protocol_violation),
        }

def record_initial_source(state: SemanticState, slot: str, color: str) -> bool:
    slot, color = str(slot), str(color)
    if state.source_slot is not None or slot not in SOURCE_SLOTS or not color:
        state.protocol_violation = True
        return False
    state.source_slot, state.source_color = slot, color
    state.initial_source_verified = True
    return True

def record_item_picked(state: SemanticState, *, physically_picked: bool) -> bool:
    if state.item_picked:
        return True
    if state.initial_source_verified and physically_picked and not state.protocol_violation:
        state.item_picked = True
    return state.item_picked

def latch_source_color_memory(state: SemanticState) -> bool:
    if state.initial_source_verified and state.item_picked and not state.protocol_violation:
        state.source_color_memory_latched = True
    return state.source_color_memory_latched

def record_target_attempt(state: SemanticState, slot: str, *, matches: bool) -> bool:
    slot = str(slot)
    if (
        not state.initial_source_verified
        or not state.item_picked
        or not state.source_color_memory_latched
        or slot not in TARGET_SLOTS
    ):
        state.protocol_violation = True
        return False
    if state.target_slot_attempted is None:
        state.target_slot_attempted = slot
    if not matches:
        state.protocol_violation = True
        return False
    return True

def record_target_placement(state: SemanticState, *, matches: bool) -> bool:
    if (
        not state.initial_source_verified
        or not state.item_picked
        or not state.source_color_memory_latched
        or not matches
        or state.protocol_violation
    ):
        if not matches:
            state.protocol_violation = True
        return False
    state.item_placed = True
    state.target_color_matches = True
    return True

def evaluate_success(state: SemanticState, physical: PhysicalState, *, target_slot: str | None) -> SuccessReport:
    wrong_target = any(bool(physical.item_in_target.get(slot, False)) for slot in TARGET_SLOTS if slot != target_slot)
    in_target = bool(
        target_slot
        and state.item_placed
        and state.target_color_matches
        and physical.item_in_target.get(target_slot, False)
        and physical.item_supported
    )
    return SuccessReport(
        initial_source_verified=state.initial_source_verified,
        item_picked=state.item_picked,
        source_color_memory_latched=state.source_color_memory_latched,
        target_color_matches=bool(state.target_color_matches),
        # A successful WR-07 ends with the carton on the matching target
        # plate; there is no source-plate return requirement.
        item_placed=in_target,
        no_wrong_target_placement=not wrong_target,
        no_protocol_violation=not state.protocol_violation,
    )

__all__ = [
    "PhysicalState", "SemanticState", "SuccessReport", "evaluate_success",
    "latch_source_color_memory", "record_initial_source", "record_item_picked",
    "record_target_attempt", "record_target_placement",
]
