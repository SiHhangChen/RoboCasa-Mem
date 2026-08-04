"""Authoritative WR-04 history state and final success predicate."""

from __future__ import annotations

from dataclasses import dataclass

from bench.tasks.wr04.domain.contracts import HOME_LEVELS, ITEM_ID
from bench.tasks.wr04.domain.episode import validate_level


@dataclass
class SemanticState:
    initial_level: str | None = None
    initial_level_physically_verified: bool = False
    item_removed_from_home: bool = False
    item_staged_at_sink: bool = False
    transport_posture_normalized: bool = False
    memory_barrier_latched: bool = False
    first_return_level: str | None = None
    wrong_return_level_attempted: bool = False
    returned_item_id: str | None = None
    protocol_violation: bool = False


@dataclass(frozen=True)
class PhysicalState:
    item_at_fixed_sink: bool
    item_in_lower: bool
    item_in_upper: bool
    item_released: bool
    item_supported_by_shelf: bool

    def item_in_level(self, level: str) -> bool:
        return self.item_in_lower if validate_level(level) == "shelf_lower" else self.item_in_upper

    @property
    def item_outside_shelves(self) -> bool:
        return not self.item_in_lower and not self.item_in_upper


@dataclass(frozen=True)
class SuccessReport:
    initial_level_physically_verified: bool
    item_removed_from_home: bool
    item_staged_at_fixed_sink: bool
    memory_barrier_latched: bool
    first_return_level_correct: bool
    no_wrong_return_level_attempt: bool
    same_item_returned: bool
    item_in_original_level: bool
    item_not_in_other_level: bool
    item_released: bool
    no_protocol_violation: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {
            "initial_level_physically_verified": self.initial_level_physically_verified,
            "item_removed_from_home": self.item_removed_from_home,
            "item_staged_at_fixed_sink": self.item_staged_at_fixed_sink,
            "memory_barrier_latched": self.memory_barrier_latched,
            "first_return_level_correct": self.first_return_level_correct,
            "no_wrong_return_level_attempt": self.no_wrong_return_level_attempt,
            "same_item_returned": self.same_item_returned,
            "item_in_original_level": self.item_in_original_level,
            "item_not_in_other_level": self.item_not_in_other_level,
            "item_released": self.item_released,
            "no_protocol_violation": self.no_protocol_violation,
        }


def record_initial_level(
    state: SemanticState,
    level: str,
    *,
    physically_verified: bool,
) -> bool:
    resolved = validate_level(level)
    if state.initial_level is not None or not physically_verified:
        state.protocol_violation = True
        return False
    state.initial_level = resolved
    state.initial_level_physically_verified = True
    return True


def record_transport_posture_normalized(
    state: SemanticState,
    *,
    physically_normalized: bool,
) -> bool:
    if state.initial_level is None or not physically_normalized:
        state.protocol_violation = True
        return False
    state.transport_posture_normalized = True
    return True


def latch_memory_barrier(
    state: SemanticState,
    *,
    item_at_fixed_sink: bool,
    item_released: bool,
    item_outside_shelves: bool,
    posture_normalized: bool,
) -> bool:
    ready = bool(
        state.initial_level_physically_verified
        and item_at_fixed_sink
        and item_released
        and item_outside_shelves
        and posture_normalized
        and state.transport_posture_normalized
        and not state.protocol_violation
    )
    if ready:
        state.item_removed_from_home = True
        state.item_staged_at_sink = True
        state.memory_barrier_latched = True
    return state.memory_barrier_latched


def record_return_attempt(state: SemanticState, level: str) -> bool:
    resolved = validate_level(level)
    if not state.memory_barrier_latched or state.initial_level is None:
        state.protocol_violation = True
        return False
    if state.first_return_level is None:
        state.first_return_level = resolved
    if resolved != state.initial_level:
        state.wrong_return_level_attempted = True
    return bool(
        resolved == state.initial_level
        and not state.wrong_return_level_attempted
        and not state.protocol_violation
    )


def record_return_completion(
    state: SemanticState,
    item_id: str,
    *,
    item_in_original_level: bool,
    item_released: bool,
) -> bool:
    item = str(item_id)
    if item != ITEM_ID:
        raise ValueError(f"invalid WR-04 item id: {item!r}")
    valid = bool(
        state.memory_barrier_latched
        and state.initial_level is not None
        and state.first_return_level == state.initial_level
        and not state.wrong_return_level_attempted
        and item_in_original_level
        and item_released
        and not state.protocol_violation
    )
    if not valid:
        state.protocol_violation = True
        return False
    state.returned_item_id = item
    return True


def evaluate_success(state: SemanticState, physical: PhysicalState) -> SuccessReport:
    original = state.initial_level
    in_original = bool(original is not None and physical.item_in_level(original))
    in_other = bool(
        original is not None
        and physical.item_in_level(_other_level(original))
    )
    return SuccessReport(
        initial_level_physically_verified=bool(
            state.initial_level_physically_verified
        ),
        item_removed_from_home=bool(state.item_removed_from_home),
        item_staged_at_fixed_sink=bool(state.item_staged_at_sink),
        memory_barrier_latched=bool(state.memory_barrier_latched),
        first_return_level_correct=bool(
            original is not None and state.first_return_level == original
        ),
        no_wrong_return_level_attempt=not state.wrong_return_level_attempted,
        same_item_returned=state.returned_item_id == ITEM_ID,
        item_in_original_level=bool(in_original and physical.item_supported_by_shelf),
        item_not_in_other_level=not in_other,
        item_released=bool(physical.item_released),
        no_protocol_violation=not state.protocol_violation,
    )


def _other_level(level: str) -> str:
    resolved = validate_level(level)
    return "shelf_upper" if resolved == "shelf_lower" else "shelf_lower"


__all__ = [
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "evaluate_success",
    "latch_memory_barrier",
    "record_initial_level",
    "record_return_attempt",
    "record_return_completion",
    "record_transport_posture_normalized",
]
