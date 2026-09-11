"""Authoritative WA-02 history and final success predicate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bench.tasks.wa02.domain.contracts import CUP_IDS, SERVICE_SLOTS, TEMPERATURES
from bench.tasks.wa02.domain.geometry import validate_service_slot_assignment


@dataclass
class SemanticState:
    """Task history that cannot be recovered reliably from the final image."""

    hot_cup_id: str
    cold_cup_id: str
    requested_temperature: str
    requested_cup_id: str
    service_slot_by_cup: dict[str, str]
    filled_temperature_by_cup: dict[str, str] = field(default_factory=dict)
    staged_slot_by_cup: dict[str, str] = field(default_factory=dict)
    cups_staged_before_final_pick: bool = False
    final_taken_cup_id: str | None = None

    def __post_init__(self) -> None:
        self.hot_cup_id = _cup(self.hot_cup_id)
        self.cold_cup_id = _cup(self.cold_cup_id)
        if self.cold_cup_id == self.hot_cup_id:
            raise ValueError("WA-02 hot and cold cups must be distinct")
        self.requested_temperature = _temperature(self.requested_temperature)
        self.requested_cup_id = _cup(self.requested_cup_id)
        expected = self.hot_cup_id if self.requested_temperature == "hot" else self.cold_cup_id
        if self.requested_cup_id != expected:
            raise ValueError("WA-02 requested cup must match requested temperature")
        self.service_slot_by_cup = validate_service_slot_assignment(self.service_slot_by_cup)


@dataclass(frozen=True)
class SuccessReport:
    """Named WA-02 checks and the single final success result."""

    hot_water_filled: bool
    cold_water_filled: bool
    both_cups_staged_before_final_pick: bool
    requested_cup_picked_up: bool
    correct_temperature_selected: bool

    @property
    def success(self) -> bool:
        return bool(all(self.checks().values()))

    def checks(self) -> dict[str, bool]:
        return {
            "hot_water_filled": bool(self.hot_water_filled),
            "cold_water_filled": bool(self.cold_water_filled),
            "both_cups_staged_before_final_pick": bool(
                self.both_cups_staged_before_final_pick
            ),
            "requested_cup_picked_up": bool(self.requested_cup_picked_up),
            "correct_temperature_selected": bool(self.correct_temperature_selected),
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.checks(), "success": self.success}


def record_verified_fill(state: SemanticState, cup_id: str, temperature: str) -> bool:
    """Record one physically verified fill event; duplicates are rejected."""

    cup = _cup(cup_id)
    temp = _temperature(temperature)
    if cup in state.filled_temperature_by_cup:
        return False
    state.filled_temperature_by_cup[cup] = temp
    return True


def record_verified_staging(state: SemanticState, cup_id: str, slot: str) -> bool:
    """Record one physically verified service-slot placement.

    This is the recording/expert event.  Learned-policy evaluation does not
    need this slot assignment; it consumes the environment's physical
    counter-contact + upright readiness snapshot instead.
    """

    cup = _cup(cup_id)
    resolved_slot = str(slot)
    if resolved_slot not in SERVICE_SLOTS or cup in state.staged_slot_by_cup:
        return False
    state.staged_slot_by_cup[cup] = resolved_slot
    return True


def latch_cups_staged(state: SemanticState, *, currently_ready: bool) -> bool:
    """Latch the historical pre-final-pick condition after physical validation."""

    if currently_ready:
        state.cups_staged_before_final_pick = True
    return state.cups_staged_before_final_pick


def record_final_take(state: SemanticState, cup_id: str) -> None:
    """Record the cup that was actually picked in the terminal operation."""

    state.final_taken_cup_id = _cup(cup_id)


def cup_for_temperature(state: SemanticState, temperature: str) -> str | None:
    """Return the uniquely observed cup for one physically recorded temperature."""

    temp = _temperature(temperature)
    matches = [
        cup
        for cup, recorded in state.filled_temperature_by_cup.items()
        if recorded == temp
    ]
    return matches[0] if len(matches) == 1 else None


def evaluate_success(state: SemanticState) -> SuccessReport:
    """Compute the strict semantic/recording success report.

    Canonical learned-policy evaluation is owned by ``WA02Evaluator`` and
    consumes the environment snapshot's counter-contact + upright readiness.
    This domain report remains the strict fallback for recording/expert paths
    and legacy diagnostics, where both service-slot assignments are required.
    """

    hot_cup = cup_for_temperature(state, "hot")
    cold_cup = cup_for_temperature(state, "cold")
    hot_filled = hot_cup is not None
    cold_filled = cold_cup is not None and cold_cup != hot_cup
    staged = bool(
        state.cups_staged_before_final_pick
        and set(state.staged_slot_by_cup) == set(CUP_IDS)
        and set(state.staged_slot_by_cup.values()) == set(SERVICE_SLOTS)
    )
    requested_cup = cup_for_temperature(state, state.requested_temperature)
    requested_taken = (
        requested_cup is not None and state.final_taken_cup_id == requested_cup
    )
    selected_temp = (
        state.filled_temperature_by_cup.get(state.final_taken_cup_id or "")
        == state.requested_temperature
    )
    return SuccessReport(
        hot_water_filled=hot_filled,
        cold_water_filled=cold_filled,
        both_cups_staged_before_final_pick=staged,
        requested_cup_picked_up=requested_taken,
        correct_temperature_selected=bool(requested_taken and selected_temp),
    )


def _cup(value: str) -> str:
    cup = str(value)
    if cup not in CUP_IDS:
        raise ValueError(f"invalid WA-02 cup id: {value!r}")
    return cup


def _temperature(value: str) -> str:
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURES:
        raise ValueError(f"invalid WA-02 temperature: {value!r}")
    return temperature


__all__ = [
    "SemanticState",
    "SuccessReport",
    "cup_for_temperature",
    "evaluate_success",
    "latch_cups_staged",
    "record_final_take",
    "record_verified_fill",
    "record_verified_staging",
]
