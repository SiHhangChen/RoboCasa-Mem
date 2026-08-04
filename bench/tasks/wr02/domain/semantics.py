"""Authoritative WR-02 history and final success predicate."""

from __future__ import annotations

from dataclasses import dataclass, field

from bench.tasks.wr02.domain.contracts import CONDIMENT_IDS


@dataclass
class SemanticState:
    staged_at_sink: dict[str, bool] = field(
        default_factory=lambda: {name: False for name in CONDIMENT_IDS}
    )


@dataclass(frozen=True)
class PhysicalState:
    condiments_in_return_shelf: bool
    condiment_shelf_level_relation_correct: bool
    condiments_released: bool


@dataclass(frozen=True)
class SuccessReport:
    ketchup_staged_at_sink: bool
    mustard_staged_at_sink: bool
    condiments_in_return_shelf: bool
    condiment_shelf_level_relation_correct: bool
    condiments_released: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {
            "ketchup_staged_at_sink": self.ketchup_staged_at_sink,
            "mustard_staged_at_sink": self.mustard_staged_at_sink,
            "condiments_in_return_shelf": self.condiments_in_return_shelf,
            "condiment_shelf_level_relation_correct": self.condiment_shelf_level_relation_correct,
            "condiments_released": self.condiments_released,
        }


def record_sink_stage(state: SemanticState, condiment_id: str) -> bool:
    condiment = str(condiment_id)
    if condiment not in state.staged_at_sink:
        raise KeyError(f"unknown WR-02 condiment {condiment!r}")
    changed = not state.staged_at_sink[condiment]
    state.staged_at_sink[condiment] = True
    return changed


def evaluate_success(state: SemanticState, physical: PhysicalState) -> SuccessReport:
    return SuccessReport(
        ketchup_staged_at_sink=bool(state.staged_at_sink["ketchup"]),
        mustard_staged_at_sink=bool(state.staged_at_sink["mustard"]),
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
    "record_sink_stage",
]
