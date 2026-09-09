from __future__ import annotations

from types import SimpleNamespace

from bench.evaluation.stage_integration import get_stage_evaluation_integration
from bench.tasks.wa02.stage_evaluation import build_expert_stage_boundaries


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "fill_temperature_order": ["cold", "hot"],
            "requested_cup_id": "cup_B",
        },
    )


def test_wa02_stage_integration_uses_seven_independent_handoffs() -> None:
    integration = get_stage_evaluation_integration("WA-02")

    assert integration.target_stage_ids == (
        "first_cup_pick",
        "first_cup_fill",
        "first_cup_ready",
        "second_cup_pick",
        "second_cup_fill",
        "second_cup_ready",
        "selection_success",
    )
    assert integration.prefix_stage_by_target == {
        "first_cup_pick": None,
        "first_cup_fill": "first_cup_pick",
        "first_cup_ready": "first_cup_fill",
        "second_cup_pick": "first_cup_ready",
        "second_cup_fill": "second_cup_pick",
        "second_cup_ready": "second_cup_fill",
        "selection_success": "second_cup_ready",
    }


def test_wa02_expert_boundaries_include_return_arm_and_sink_exit() -> None:
    boundaries = build_expert_stage_boundaries(
        None,
        _plan(),
        {
            "return_to_regular_after_pick_first_cup_A": 100,
            "lift_after_sink_exit_first_cold_cup_A": 240,
            "return_after_service_first_cup_A": 380,
            "return_to_regular_after_pick_second_cup_B": 520,
            "lift_after_sink_exit_second_hot_cup_B": 660,
            "return_after_service_second_cup_B": 800,
            "record_final_take_cup_B": 940,
        },
    )

    assert boundaries == {
        "first_cup_pick": 100,
        "first_cup_fill": 240,
        "first_cup_ready": 380,
        "second_cup_pick": 520,
        "second_cup_fill": 660,
        "second_cup_ready": 800,
        "selection_success": 940,
    }
