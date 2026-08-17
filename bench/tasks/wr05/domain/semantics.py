"""Authoritative WR-05 history and final success predicate.

The use station is a fixed serving plate on the dining island: a condiment is
staged when it sits beside the plate (within ``plate_stage_threshold`` of its
plate-left or plate-right target or the plate itself).  Shelf return semantics
follow WR-02.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bench.tasks.wr05.domain.contracts import CONDIMENT_IDS


@dataclass
class SemanticState:
    staged_at_plate: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in CONDIMENT_IDS}
    )


@dataclass(frozen=True)
class PhysicalState:
    condiments_in_return_shelf: bool
    condiment_shelf_level_relation_correct: bool
    condiments_released: bool


@dataclass(frozen=True)
class SuccessReport:
    ketchup_staged_at_plate: bool
    mustard_staged_at_plate: bool
    condiments_in_return_shelf: bool
    condiment_shelf_level_relation_correct: bool
    condiments_released: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {
            "ketchup_staged_at_plate": self.ketchup_staged_at_plate,
            "mustard_staged_at_plate": self.mustard_staged_at_plate,
            "condiments_in_return_shelf": self.condiments_in_return_shelf,
            "condiment_shelf_level_relation_correct": self.condiment_shelf_level_relation_correct,
            "condiments_released": self.condiments_released,
        }


def record_plate_stage(state: SemanticState, condiment_id: str) -> bool:
    condiment = str(condiment_id)
    if condiment not in state.staged_at_plate:
        raise KeyError(f"unknown WR-05 condiment {condiment!r}")
    changed = not state.staged_at_plate[condiment]
    state.staged_at_plate[condiment] = True
    return changed


def evaluate_success(state: SemanticState, physical: PhysicalState) -> SuccessReport:
    return SuccessReport(
        ketchup_staged_at_plate=bool(state.staged_at_plate["ketchup"]),
        mustard_staged_at_plate=bool(state.staged_at_plate["mustard"]),
        condiments_in_return_shelf=bool(physical.condiments_in_return_shelf),
        condiment_shelf_level_relation_correct=bool(
            physical.condiment_shelf_level_relation_correct
        ),
        condiments_released=bool(physical.condiments_released),
    )


__all__ = [
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "evaluate_success",
    "record_plate_stage",
]
