"""Authoritative WA-04 history and final success predicate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.tasks.wa04.domain.contracts import CUP_IDS, SERVICE_SLOTS, TEMPERATURES
from bench.tasks.wa04.domain.geometry import validate_service_slot_assignment


@dataclass
class SemanticState:
    """Task history that cannot be recovered reliably from the final image."""

    temperature_by_cup: dict[str, str]
    requested_temperature: str
    requested_cup_id: str
    service_slot_by_cup: dict[str, str]
    filled_temperature_by_cup: dict[str, str] = field(default_factory=dict)
    staged_slot_by_cup: dict[str, str] = field(default_factory=dict)
    cups_staged_before_final_pick: bool = False
    final_taken_cup_id: str | None = None

    def __post_init__(self) -> None:
        mapping = {str(cup): _temperature(value) for cup, value in self.temperature_by_cup.items()}
        if set(mapping) != set(CUP_IDS):
            raise ValueError("WA-04 temperature mapping must contain all three cups")
        self.temperature_by_cup = mapping
        self.requested_temperature = _temperature(self.requested_temperature)
        self.requested_cup_id = _cup(self.requested_cup_id)
        matches = [
            cup for cup, temperature in mapping.items()
            if temperature == self.requested_temperature
        ]
        if matches != [self.requested_cup_id]:
            raise ValueError("WA-04 requested temperature must identify exactly the requested cup")
        self.service_slot_by_cup = validate_service_slot_assignment(self.service_slot_by_cup)


@dataclass(frozen=True)
class SuccessReport:
    """Named WA-04 checks and the single final success result."""

    all_three_cups_filled_as_planned: bool
    all_three_cups_staged_before_final_pick: bool
    requested_cup_picked_up: bool
    unique_requested_temperature_selected: bool

    @property
    def success(self) -> bool:
        return bool(all(self.checks().values()))

    def checks(self) -> dict[str, bool]:
        return {
            "all_three_cups_filled_as_planned": bool(self.all_three_cups_filled_as_planned),
            "all_three_cups_staged_before_final_pick": bool(
                self.all_three_cups_staged_before_final_pick
            ),
            "requested_cup_picked_up": bool(self.requested_cup_picked_up),
            "unique_requested_temperature_selected": bool(
                self.unique_requested_temperature_selected
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.checks(), "success": self.success}


def record_verified_fill(state: SemanticState, cup_id: str, temperature: str) -> bool:
    cup = _cup(cup_id)
    temp = _temperature(temperature)
    if cup in state.filled_temperature_by_cup or state.temperature_by_cup[cup] != temp:
        return False
    state.filled_temperature_by_cup[cup] = temp
    return True


def record_verified_staging(state: SemanticState, cup_id: str, slot: str) -> bool:
    cup = _cup(cup_id)
    resolved_slot = str(slot)
    if resolved_slot not in SERVICE_SLOTS or cup in state.staged_slot_by_cup:
        return False
    state.staged_slot_by_cup[cup] = resolved_slot
    return True


def latch_cups_staged(state: SemanticState, *, currently_ready: bool) -> bool:
    if currently_ready:
        state.cups_staged_before_final_pick = True
    return state.cups_staged_before_final_pick


def record_final_take(state: SemanticState, cup_id: str) -> None:
    state.final_taken_cup_id = _cup(cup_id)


def evaluate_success(state: SemanticState) -> SuccessReport:
    filled = state.filled_temperature_by_cup == state.temperature_by_cup
    staged = bool(
        state.cups_staged_before_final_pick
        and state.staged_slot_by_cup == state.service_slot_by_cup
    )
    requested_taken = state.final_taken_cup_id == state.requested_cup_id
    selected_temp = (
        state.filled_temperature_by_cup.get(state.final_taken_cup_id or "")
        == state.requested_temperature
    )
    return SuccessReport(
        all_three_cups_filled_as_planned=filled,
        all_three_cups_staged_before_final_pick=staged,
        requested_cup_picked_up=requested_taken,
        unique_requested_temperature_selected=bool(requested_taken and selected_temp),
    )


def _cup(value: str) -> str:
    cup = str(value)
    if cup not in CUP_IDS:
        raise ValueError(f"invalid WA-04 cup id: {value!r}")
    return cup


def _temperature(value: str) -> str:
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURES:
        raise ValueError(f"invalid WA-04 temperature: {value!r}")
    return temperature


__all__ = [
    "SemanticState",
    "SuccessReport",
    "evaluate_success",
    "latch_cups_staged",
    "record_final_take",
    "record_verified_fill",
    "record_verified_staging",
]
