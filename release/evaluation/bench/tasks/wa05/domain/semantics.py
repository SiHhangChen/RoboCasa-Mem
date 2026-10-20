"""Authoritative WA-05 event history and success predicate."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from bench.tasks.wa05.domain.contracts import TEMPERATURES

@dataclass
class SemanticState:
    filled_temperature: str | None = None
    staged_before_answer: bool = False
    pressed_temperature: str | None = None
    absorbing_failure: bool = False
    failure_code: str | None = None
    def fail(self, code: str) -> None:
        if not self.absorbing_failure:
            self.absorbing_failure = True
            self.failure_code = str(code)

@dataclass(frozen=True)
class SuccessReport:
    single_cup_filled: bool
    single_temperature_fill: bool
    cup_staged_before_answer: bool
    physical_answer_button_pressed: bool
    answer_matches_filled_temperature: bool
    no_absorbing_failure: bool
    @property
    def success(self) -> bool:
        return bool(all(self.checks().values()))
    def checks(self) -> dict[str, bool]:
        return {"single_cup_filled": self.single_cup_filled, "single_temperature_fill": self.single_temperature_fill, "cup_staged_before_answer": self.cup_staged_before_answer, "physical_answer_button_pressed": self.physical_answer_button_pressed, "answer_matches_filled_temperature": self.answer_matches_filled_temperature, "no_absorbing_failure": self.no_absorbing_failure}
    def as_dict(self) -> dict[str, Any]:
        return {**self.checks(), "success": self.success}

def record_verified_fill(state: SemanticState, temperature: str) -> bool:
    temp = _temperature(temperature)
    if state.absorbing_failure: return False
    if state.filled_temperature is None:
        state.filled_temperature = temp
        return True
    if state.filled_temperature != temp: state.fail("mixed_temperature_fill")
    return state.filled_temperature == temp and not state.absorbing_failure

def record_verified_staging(state: SemanticState, *, currently_ready: bool) -> bool:
    if state.absorbing_failure or state.filled_temperature is None or not currently_ready: return False
    state.staged_before_answer = True
    return True

def record_button_press(state: SemanticState, temperature: str) -> bool:
    temp = _temperature(temperature)
    if state.absorbing_failure: return False
    if state.filled_temperature is None:
        state.fail("button_pressed_before_fill"); return False
    if not state.staged_before_answer:
        state.fail("button_pressed_before_staging"); return False
    if state.pressed_temperature is not None:
        state.fail("multiple_answer_buttons"); return False
    state.pressed_temperature = temp
    if temp != state.filled_temperature:
        state.fail("wrong_temperature_button"); return False
    return True

def evaluate_success(state: SemanticState) -> SuccessReport:
    filled = state.filled_temperature in TEMPERATURES
    pressed = state.pressed_temperature in TEMPERATURES
    matched = bool(filled and pressed and state.pressed_temperature == state.filled_temperature)
    return SuccessReport(filled, filled, bool(state.staged_before_answer), pressed, matched, not state.absorbing_failure)

def _temperature(value: str) -> str:
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURES: raise ValueError(f"invalid WA-05 temperature: {value!r}")
    return temperature

__all__ = ["SemanticState", "SuccessReport", "evaluate_success", "record_button_press", "record_verified_fill", "record_verified_staging"]
