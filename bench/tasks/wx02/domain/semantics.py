"""Authoritative history and final success predicate for WX-02."""

from __future__ import annotations

from dataclasses import dataclass

from bench.tasks.wx02.domain.contracts import PRIMARY_CUP, SECONDARY_CUP


@dataclass
class SemanticState:
    """Historical facts that cannot be reconstructed from the final frame."""

    cleaning_tool_acquired: bool = False
    primary_cup_moved: bool = False
    primary_stain_cleaned: bool = False
    primary_cup_returned: bool = False
    secondary_cup_moved: bool = False
    secondary_stain_cleaned: bool = False
    secondary_cup_returned: bool = False
    cleaning_tool_left_return_region: bool = False
    cleaning_tool_returned: bool = False


@dataclass(frozen=True)
class PhysicalState:
    """Current physical facts read from the simulator."""

    primary_cup_at_origin: bool
    primary_cup_on_dining_table: bool
    secondary_cup_at_origin: bool
    secondary_cup_on_dining_table: bool
    cleaning_tool_in_return_region: bool
    cleaning_tool_on_dining_table: bool
    cleaning_tool_released: bool
    primary_cup_released: bool
    secondary_cup_released: bool


@dataclass(frozen=True)
class SuccessReport:
    """The one authoritative WX-02 success report."""

    cleaning_tool_acquired: bool
    primary_cup_moved: bool
    primary_stain_cleaned: bool
    primary_cup_at_origin: bool
    primary_cup_on_dining_table: bool
    secondary_cup_moved: bool
    secondary_stain_cleaned: bool
    secondary_cup_at_origin: bool
    secondary_cup_on_dining_table: bool
    cleaning_tool_released: bool
    cleaning_tool_in_return_region: bool
    cleaning_tool_on_dining_table: bool

    @property
    def success(self) -> bool:
        return all(self.checks().values())

    def checks(self) -> dict[str, bool]:
        return {
            "cleaning_tool_acquired": self.cleaning_tool_acquired,
            "primary_cup_moved": self.primary_cup_moved,
            "primary_stain_cleaned": self.primary_stain_cleaned,
            "primary_cup_at_origin": self.primary_cup_at_origin,
            "primary_cup_on_dining_table": self.primary_cup_on_dining_table,
            "secondary_cup_moved": self.secondary_cup_moved,
            "secondary_stain_cleaned": self.secondary_stain_cleaned,
            "secondary_cup_at_origin": self.secondary_cup_at_origin,
            "secondary_cup_on_dining_table": self.secondary_cup_on_dining_table,
            "cleaning_tool_released": self.cleaning_tool_released,
            "cleaning_tool_in_return_region": self.cleaning_tool_in_return_region,
            "cleaning_tool_on_dining_table": self.cleaning_tool_on_dining_table,
        }


def record_tool_acquired(state: SemanticState) -> bool:
    changed = not state.cleaning_tool_acquired
    state.cleaning_tool_acquired = True
    return changed


def record_tool_left_return_region(state: SemanticState) -> bool:
    if not state.cleaning_tool_acquired:
        return False
    changed = not state.cleaning_tool_left_return_region
    state.cleaning_tool_left_return_region = True
    return changed


def record_object_moved(state: SemanticState, object_name: str) -> bool:
    if object_name == PRIMARY_CUP:
        if not state.cleaning_tool_acquired:
            return False
        changed = not state.primary_cup_moved
        state.primary_cup_moved = True
        return changed
    if object_name == SECONDARY_CUP:
        if not state.primary_cup_returned:
            return False
        changed = not state.secondary_cup_moved
        state.secondary_cup_moved = True
        return changed
    raise KeyError(f"unknown WX-02 cup: {object_name!r}")


def record_stain_cleaned(state: SemanticState, object_name: str) -> bool:
    if object_name == PRIMARY_CUP:
        if not state.primary_cup_moved:
            return False
        changed = not state.primary_stain_cleaned
        state.primary_stain_cleaned = True
        return changed
    if object_name == SECONDARY_CUP:
        if not state.primary_cup_returned or not state.secondary_cup_moved:
            return False
        changed = not state.secondary_stain_cleaned
        state.secondary_stain_cleaned = True
        return changed
    raise KeyError(f"unknown WX-02 cup: {object_name!r}")


def record_object_returned(state: SemanticState, object_name: str) -> bool:
    if object_name == PRIMARY_CUP:
        if not state.primary_stain_cleaned:
            return False
        changed = not state.primary_cup_returned
        state.primary_cup_returned = True
        return changed
    if object_name == SECONDARY_CUP:
        if not state.secondary_stain_cleaned:
            return False
        changed = not state.secondary_cup_returned
        state.secondary_cup_returned = True
        return changed
    raise KeyError(f"unknown WX-02 cup: {object_name!r}")


def record_tool_returned(state: SemanticState) -> bool:
    if (
        not state.cleaning_tool_left_return_region
        or not state.primary_cup_returned
        or not state.secondary_cup_returned
    ):
        return False
    changed = not state.cleaning_tool_returned
    state.cleaning_tool_returned = True
    return changed


def evaluate_success(state: SemanticState, physical: PhysicalState) -> SuccessReport:
    return SuccessReport(
        cleaning_tool_acquired=bool(state.cleaning_tool_acquired),
        primary_cup_moved=bool(state.primary_cup_moved),
        primary_stain_cleaned=bool(state.primary_stain_cleaned),
        primary_cup_at_origin=bool(
            state.primary_cup_returned and physical.primary_cup_at_origin
        ),
        primary_cup_on_dining_table=bool(
            state.primary_cup_returned and physical.primary_cup_on_dining_table
        ),
        secondary_cup_moved=bool(state.secondary_cup_moved),
        secondary_stain_cleaned=bool(state.secondary_stain_cleaned),
        secondary_cup_at_origin=bool(
            state.secondary_cup_returned and physical.secondary_cup_at_origin
        ),
        secondary_cup_on_dining_table=bool(
            state.secondary_cup_returned and physical.secondary_cup_on_dining_table
        ),
        cleaning_tool_released=bool(
            state.cleaning_tool_returned and physical.cleaning_tool_released
        ),
        cleaning_tool_in_return_region=bool(
            state.cleaning_tool_returned and physical.cleaning_tool_in_return_region
        ),
        cleaning_tool_on_dining_table=bool(
            state.cleaning_tool_returned and physical.cleaning_tool_on_dining_table
        ),
    )


__all__ = [
    "PhysicalState",
    "SemanticState",
    "SuccessReport",
    "evaluate_success",
    "record_object_moved",
    "record_object_returned",
    "record_stain_cleaned",
    "record_tool_acquired",
    "record_tool_left_return_region",
    "record_tool_returned",
]
