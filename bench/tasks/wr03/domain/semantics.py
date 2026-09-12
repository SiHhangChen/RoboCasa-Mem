"""Authoritative WR-03 history state and final success predicate."""

from __future__ import annotations

from dataclasses import dataclass, field

from bench.tasks.wr03.domain.contracts import CABINET_LEVELS, CAN_ID


@dataclass
class SemanticState:
    """History facts that cannot be inferred from the final simulator frame."""

    stored_level: str | None = None
    closed_levels_after_storage: set[str] = field(default_factory=set)
    storage_hidden: bool = False
    # Recording metadata only; never a policy evaluation prerequisite.
    canonical_reset_complete: bool = False
    retrieval_level_opened: str | None = None
    wrong_retrieval_level_opened: bool = False
    retrieved_item_id: str | None = None
    protocol_violation: bool = False


@dataclass(frozen=True)
class PhysicalState:
    can_in_lower: bool
    can_in_upper: bool
    lower_door_closed: bool
    upper_door_closed: bool

    def can_in_level(self, level: str) -> bool:
        return self.can_in_lower if _level(level) == "lower" else self.can_in_upper

    @property
    def can_outside_both(self) -> bool:
        return not self.can_in_lower and not self.can_in_upper

    @property
    def both_doors_closed(self) -> bool:
        return self.lower_door_closed and self.upper_door_closed


@dataclass(frozen=True)
class SuccessReport:
    can_storage_verified: bool
    both_doors_closed_after_storage: bool
    memory_barrier_latched: bool
    correct_retrieval_door: bool
    same_can_retrieved: bool
    can_outside_both_cabinets: bool
    no_protocol_violation: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {
            "can_storage_verified": self.can_storage_verified,
            "both_doors_closed_after_storage": self.both_doors_closed_after_storage,
            "memory_barrier_latched": self.memory_barrier_latched,
            "correct_retrieval_door": self.correct_retrieval_door,
            "same_can_retrieved": self.same_can_retrieved,
            "can_outside_both_cabinets": self.can_outside_both_cabinets,
            "no_protocol_violation": self.no_protocol_violation,
        }


def record_storage(state: SemanticState, item_id: str, level: str) -> bool:
    """Latch the first physically verified can-to-cabinet relation."""

    _item(item_id)
    resolved = _level(level)
    if state.storage_hidden or state.retrieved_item_id is not None:
        state.protocol_violation = True
        return False
    if state.stored_level is None:
        state.stored_level = resolved
        return True
    if state.stored_level != resolved:
        state.protocol_violation = True
        return False
    return True


def record_storage_door_closed(state: SemanticState, level: str) -> bool:
    """Record a successful physical close only after storage was verified."""

    resolved = _level(level)
    if resolved in state.closed_levels_after_storage:
        return True
    if state.stored_level is None or state.storage_hidden:
        state.protocol_violation = True
        return False
    state.closed_levels_after_storage.add(resolved)
    return True


def latch_storage_hidden(
    state: SemanticState,
    *,
    can_still_stored: bool,
    cabinets_closed: bool,
) -> bool:
    """Latch when the stored can is hidden by both currently closed doors."""

    ready = bool(
        state.stored_level is not None
        and can_still_stored
        and cabinets_closed
        and not state.protocol_violation
    )
    if ready:
        # Preserve recording acknowledgements without requiring a close dwell
        # or a neutral base/torso pose to establish the memory barrier.
        state.closed_levels_after_storage.update(CABINET_LEVELS)
        state.storage_hidden = True
    return state.storage_hidden


def mark_canonical_reset(state: SemanticState, *, valid: bool) -> bool:
    if valid:
        state.canonical_reset_complete = True
    return state.canonical_reset_complete


def record_retrieval_door_open(state: SemanticState, level: str) -> bool:
    """Latch the target opening; opening the empty cabinet is diagnostic only."""

    resolved = _level(level)
    if not state.storage_hidden or state.stored_level is None:
        state.protocol_violation = True
        return False
    if resolved != state.stored_level:
        state.wrong_retrieval_level_opened = True
        return False
    if state.protocol_violation:
        return False
    state.retrieval_level_opened = resolved
    return True


def record_retrieval_pick(
    state: SemanticState,
    item_id: str,
    *,
    physically_extracted: bool,
) -> bool:
    resolved_item = _item(item_id)
    valid = bool(
        physically_extracted
        and state.storage_hidden
        and state.stored_level is not None
        and state.retrieval_level_opened == state.stored_level
        and not state.protocol_violation
    )
    if not valid:
        state.protocol_violation = True
        return False
    state.retrieved_item_id = resolved_item
    return True


def evaluate_success(state: SemanticState, physical: PhysicalState) -> SuccessReport:
    stored = state.stored_level
    storage_verified = stored is not None
    return SuccessReport(
        can_storage_verified=storage_verified,
        both_doors_closed_after_storage=(
            state.closed_levels_after_storage == set(CABINET_LEVELS)
        ),
        memory_barrier_latched=bool(state.storage_hidden),
        correct_retrieval_door=(
            stored is not None and state.retrieval_level_opened == stored
        ),
        same_can_retrieved=state.retrieved_item_id == CAN_ID,
        can_outside_both_cabinets=physical.can_outside_both,
        no_protocol_violation=not state.protocol_violation,
    )


def _item(value: str) -> str:
    item_id = str(value)
    if item_id != CAN_ID:
        raise ValueError(f"invalid WR-03 item id: {item_id!r}")
    return item_id


def _level(value: str) -> str:
    level = str(value)
    if level not in CABINET_LEVELS:
        raise ValueError(f"invalid WR-03 cabinet level: {level!r}")
    return level


__all__ = [
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "evaluate_success",
    "latch_storage_hidden",
    "mark_canonical_reset",
    "record_retrieval_door_open",
    "record_retrieval_pick",
    "record_storage",
    "record_storage_door_closed",
]
