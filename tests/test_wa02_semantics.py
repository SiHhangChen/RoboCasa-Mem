from __future__ import annotations

from bench.tasks.wa02.domain.semantics import (
    SemanticState,
    cup_for_temperature,
    evaluate_success,
    record_final_take,
    record_verified_fill,
    record_verified_staging,
)


def _state(*, requested_temperature: str = "hot") -> SemanticState:
    return SemanticState(
        hot_cup_id="cup_A",
        cold_cup_id="cup_B",
        requested_temperature=requested_temperature,
        requested_cup_id=("cup_A" if requested_temperature == "hot" else "cup_B"),
        service_slot_by_cup={"cup_A": "slot_1", "cup_B": "slot_2"},
    )


def test_wa02_success_uses_actual_fill_binding_not_seed_plan() -> None:
    state = _state(requested_temperature="hot")
    record_verified_fill(state, "cup_B", "hot")
    record_verified_fill(state, "cup_A", "cold")
    record_verified_staging(state, "cup_B", "slot_1")
    record_verified_staging(state, "cup_A", "slot_2")
    state.cups_staged_before_final_pick = True
    record_final_take(state, "cup_B")

    assert cup_for_temperature(state, "hot") == "cup_B"
    assert evaluate_success(state).success


def test_wa02_success_rejects_seed_planned_cup_when_actual_fill_differs() -> None:
    state = _state(requested_temperature="hot")
    record_verified_fill(state, "cup_B", "hot")
    record_verified_fill(state, "cup_A", "cold")
    record_verified_staging(state, "cup_A", "slot_1")
    record_verified_staging(state, "cup_B", "slot_2")
    state.cups_staged_before_final_pick = True
    record_final_take(state, "cup_A")

    report = evaluate_success(state)
    assert not report.requested_cup_picked_up
    assert not report.correct_temperature_selected
    assert not report.success
