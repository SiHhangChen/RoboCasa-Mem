"""Authoritative WR-01 history state and final success predicate."""

from __future__ import annotations

from dataclasses import dataclass, field

from bench.tasks.wr01.domain.contracts import CABINET_LEVELS, ITEM_IDS


@dataclass
class SemanticState:
    """Facts that cannot be reconstructed from the final simulator frame."""

    target_item_id: str
    target_level: str
    stored_levels_by_item: dict[str, str] = field(default_factory=dict)
    storage_hidden: bool = False
    # Expert-recording metadata, never a policy-evaluation prerequisite.
    canonical_reset_complete: bool = False
    # Latches opening the target cabinet after the memory barrier.
    retrieval_level_opened: str | None = None
    # Diagnostic history only: opening the other cabinet never invalidates a run.
    wrong_retrieval_level_opened: bool = False
    retrieved_item_id: str | None = None

    def __post_init__(self) -> None:
        _item(self.target_item_id)
        _level(self.target_level)


@dataclass(frozen=True)
class PhysicalState:
    distractor_still_stored: bool


@dataclass(frozen=True)
class SuccessReport:
    storage_assignment_complete: bool
    storage_was_hidden: bool
    correct_retrieval_door: bool
    correct_target_retrieved: bool
    distractor_still_stored: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {
            "storage_assignment_complete": self.storage_assignment_complete,
            "storage_was_hidden": self.storage_was_hidden,
            "correct_retrieval_door": self.correct_retrieval_door,
            "correct_target_retrieved": self.correct_target_retrieved,
            "distractor_still_stored": self.distractor_still_stored,
        }


def record_storage(state: SemanticState, item_id: str, level: str) -> bool:
    """Record a physically verified storage event; conflicting rewrites fail."""

    item_id, level = _item(item_id), _level(level)
    existing = state.stored_levels_by_item.get(item_id)
    if existing is not None and existing != level:
        return False
    state.stored_levels_by_item[item_id] = level
    return True


def latch_storage_hidden(
    state: SemanticState,
    *,
    all_stored: bool,
    cabinets_closed: bool,
) -> bool:
    """Latch the memory barrier only after both relation facts became hidden."""

    if all_stored and cabinets_closed:
        state.storage_hidden = True
    return state.storage_hidden


def mark_canonical_reset(state: SemanticState, *, valid: bool) -> bool:
    """Record the expert's neutral handoff without changing evaluation gates."""

    if valid:
        state.canonical_reset_complete = True
    return state.canonical_reset_complete


def record_retrieval_door_open(state: SemanticState, level: str) -> bool:
    """Latch target-cabinet opening independently of any other door operation."""

    level = _level(level)
    if not state.storage_hidden:
        return False
    if level == state.target_level:
        state.retrieval_level_opened = level
    else:
        state.wrong_retrieval_level_opened = True
    return level == state.target_level


def record_retrieval_pick(state: SemanticState, item_id: str) -> bool:
    item_id = _item(item_id)
    state.retrieved_item_id = item_id
    return item_id == state.target_item_id


def evaluate_success(state: SemanticState, physical: PhysicalState) -> SuccessReport:
    assignment_complete = (
        set(state.stored_levels_by_item) == set(ITEM_IDS)
        and set(state.stored_levels_by_item.values()) == set(CABINET_LEVELS)
    )
    return SuccessReport(
        storage_assignment_complete=assignment_complete,
        storage_was_hidden=bool(state.storage_hidden),
        correct_retrieval_door=state.retrieval_level_opened == state.target_level,
        correct_target_retrieved=state.retrieved_item_id == state.target_item_id,
        distractor_still_stored=bool(physical.distractor_still_stored),
    )


def _item(value: str) -> str:
    item_id = str(value)
    if item_id not in ITEM_IDS:
        raise ValueError(f"invalid WR-01 item id: {item_id!r}")
    return item_id


def _level(value: str) -> str:
    level = str(value)
    if level not in CABINET_LEVELS:
        raise ValueError(f"invalid WR-01 cabinet level: {level!r}")
    return level


def _other_level(level: str) -> str:
    return "upper" if level == "lower" else "lower"


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
]
