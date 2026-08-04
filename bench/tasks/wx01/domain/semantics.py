"""Authoritative semantic state and final success predicate for WX-01."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SemanticState:
    """Historical facts that cannot be reconstructed from the final frame."""

    remembered_area_wiped: bool = False


@dataclass(frozen=True)
class PhysicalState:
    """Final physical facts read from the simulator."""

    blocking_object_at_origin: bool
    blocking_object_on_dining_table: bool
    cleaning_tool_released: bool
    blocking_object_released: bool


@dataclass(frozen=True)
class SuccessReport:
    """The one authoritative WX-01 success report."""

    remembered_area_wiped: bool
    blocking_object_at_origin: bool
    blocking_object_on_dining_table: bool
    cleaning_tool_released: bool
    blocking_object_released: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {
            "remembered_area_wiped": self.remembered_area_wiped,
            "blocking_object_at_origin": self.blocking_object_at_origin,
            "blocking_object_on_dining_table": self.blocking_object_on_dining_table,
            "cleaning_tool_released": self.cleaning_tool_released,
            "blocking_object_released": self.blocking_object_released,
        }


def record_wipe_completion(state: SemanticState) -> bool:
    """Latch the wipe event exactly once."""

    changed = not state.remembered_area_wiped
    state.remembered_area_wiped = True
    return changed


def evaluate_success(state: SemanticState, physical: PhysicalState) -> SuccessReport:
    """Combine historical and final physical facts in one predicate."""

    return SuccessReport(
        remembered_area_wiped=bool(state.remembered_area_wiped),
        blocking_object_at_origin=bool(physical.blocking_object_at_origin),
        blocking_object_on_dining_table=bool(physical.blocking_object_on_dining_table),
        cleaning_tool_released=bool(physical.cleaning_tool_released),
        blocking_object_released=bool(physical.blocking_object_released),
    )


__all__ = [
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "evaluate_success",
    "record_wipe_completion",
]
