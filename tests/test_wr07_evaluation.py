"""WR-07 policy evaluation follows physical transport, not expert events."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr07.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WR07EvaluationSnapshot,
    WR07Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


def _snapshot(step: int, **kwargs) -> WR07EvaluationSnapshot:
    return WR07EvaluationSnapshot(step_id=step, **kwargs)


def _evaluator(**kwargs) -> WR07Evaluator:
    evaluator = WR07Evaluator(**kwargs)
    evaluator.reset(_snapshot(0, item_placed=True))
    return evaluator


def _validate(result) -> None:
    validate_report_stages(
        result.to_report_stages(),
        expected_task_id="WR-07",
        expected_profile_id="WR-07/report-v4",
        expected_success=result.success,
    )


@pytest.mark.parametrize("protocol_violation", (False, True))
def test_policy_success_requires_no_expert_memory_or_domain_success(protocol_violation):
    evaluator = _evaluator()
    picked = evaluator.observe(_snapshot(1, item_picked=True))
    assert picked.stages["item_picked"].qualified
    assert not picked.stages["item_placed"].qualified
    for step in (2, 3):
        result = evaluator.observe(_snapshot(
            step,
            item_placed=True,
            target_color_matches=True,
            source_color_memory_latched=False,
            protocol_violation=protocol_violation,
            domain_success=False,
        ))
        _validate(result)
        assert result.success is (step == 3)
        assert result.stop_execution is (step == 3)
    assert result.terminal
    assert result.failure_code is None
    assert result.details["evaluation"]["mode"] == "policy_physical"
    assert {name: result.stages[name].qualified_step for name in STAGE_ORDER} == {
        "item_picked": 1,
        "item_placed": 3,
        "target_plate_matched": 3,
        "success": 3,
    }
    assert REPORT_STAGE_PROFILE_ID == "WR-07/report-v4"
    assert "memory_color_latched" not in result.stages


def test_wrong_plate_is_a_terminal_failure_and_cannot_be_corrected():
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, item_picked=True))
    for step in (2, 3):
        result = evaluator.observe(_snapshot(
            step, item_placed=True, target_slot_attempted="target_left"
        ))
        assert result.terminal is (step == 3)
        assert result.stop_execution is (step == 3)
    assert result.stages["item_placed"].qualified
    assert not result.stages["target_plate_matched"].qualified
    assert not result.success and result.terminal
    assert result.failure_code == "wrong_target_placement"
    assert result.stages["target_plate_matched"].invalidated
    assert result.stages["success"].reason == "wrong_target_placement"
    assert result.details["evaluation"]["wrong_target_slot"] == "target_left"
    assert result.details["evaluation"]["terminal_step"] == 3
    _validate(result)

    terminal = result
    assert evaluator.observe(_snapshot(4, item_picked=True)) is terminal
    for step in (5, 6):
        result = evaluator.observe(_snapshot(
            step, item_placed=True, target_color_matches=True
        ))
        assert result is terminal
    assert not result.success
    assert result.stages["item_placed"].qualified_step == 3
    assert result.stages["target_plate_matched"].qualified_step is None
    _validate(result)


def test_wrong_target_needs_consecutive_released_contact_on_the_same_tray():
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, item_picked=True))
    for step, slot, placed in (
        (2, "target_left", True),
        (3, "target_center", True),
        (4, "target_center", False),
        (5, "target_center", True),
        (7, "target_center", True),
    ):
        sample = _snapshot(step, item_placed=placed, target_slot_attempted=slot)
        result = evaluator.observe(sample)
        assert not result.terminal
        assert evaluator.observe(sample) is result
    result = evaluator.observe(_snapshot(
        8, item_placed=True, target_slot_attempted="target_center",
    ))
    assert result.terminal
    assert result.failure_code == "wrong_target_placement"
    assert result.details["evaluation"]["terminal_step"] == 8


def test_non_destination_putdown_does_not_trigger_wrong_target_failure():
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, item_picked=True))
    for step in (2, 3):
        result = evaluator.observe(_snapshot(step, item_placed=True))
    assert result.stages["item_placed"].qualified
    assert not result.terminal
    assert not result.stop_execution
    assert result.failure_code is None


def test_wrong_target_without_pickup_history_does_not_create_a_placement():
    evaluator = _evaluator()
    for step in (1, 2, 3):
        result = evaluator.observe(_snapshot(
            step, item_placed=True, target_slot_attempted="target_left",
        ))
        assert not result.terminal
        assert not result.stages["item_placed"].qualified


@pytest.mark.parametrize("placed", (False, True))
def test_placement_and_semantic_success_cannot_skip_pickup(placed):
    evaluator = _evaluator()
    for step in range(1, 5):
        result = evaluator.observe(_snapshot(
            step, item_placed=placed, target_color_matches=True,
            source_color_memory_latched=True, domain_success=True,
        ))
        assert not any(stage.qualified for stage in result.stages.values())
        _validate(result)


def test_pick_and_putdown_cannot_be_credited_from_the_same_policy_sample():
    evaluator = _evaluator()
    for step in (1, 2, 3):
        result = evaluator.observe(_snapshot(
            step, item_picked=True, item_placed=True, target_color_matches=True,
        ))
        assert not any(stage.qualified for stage in result.stages.values())


def test_pickup_present_at_reset_needs_a_new_pickup_event():
    evaluator = WR07Evaluator()
    evaluator.reset(_snapshot(0, item_picked=True))
    result = evaluator.observe(_snapshot(1, item_picked=True))
    assert not result.stages["item_picked"].qualified
    evaluator.observe(_snapshot(2, item_placed=True))
    evaluator.observe(_snapshot(3, item_picked=True))
    evaluator.observe(_snapshot(4, item_placed=True, target_color_matches=True))
    result = evaluator.observe(_snapshot(5, item_placed=True, target_color_matches=True))
    assert result.success


def test_airborne_release_or_held_carton_does_not_count_as_putdown():
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, item_picked=True))
    for step in (2, 3):
        result = evaluator.observe(_snapshot(step, target_color_matches=True))
        assert not result.stages["item_placed"].qualified
        assert not result.stages["target_plate_matched"].qualified
        assert not result.success
        _validate(result)


def test_duplicate_or_missing_steps_do_not_supply_placement_dwell():
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, item_picked=True))
    placement = _snapshot(2, item_placed=True, target_color_matches=True)
    result = evaluator.observe(placement)
    for _ in range(5):
        assert evaluator.observe(placement) is result
    assert not result.success
    result = evaluator.observe(replace(placement, step_id=4))
    assert not result.success
    result = evaluator.observe(replace(placement, step_id=5))
    assert result.success
    with pytest.raises(ValueError, match="backwards"):
        evaluator.observe(placement)


def test_transient_contact_resets_dwell_and_confirmed_success_is_final():
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, item_picked=True))
    evaluator.observe(_snapshot(2, item_placed=True, target_color_matches=True))
    evaluator.observe(_snapshot(3))
    result = evaluator.observe(_snapshot(4, item_placed=True, target_color_matches=True))
    assert not result.success
    result = evaluator.observe(_snapshot(5, item_placed=True, target_color_matches=True))
    assert result.success
    assert result.terminal
    assert result.details["evaluation"]["terminal_step"] == 5
    terminal = result
    assert evaluator.observe(_snapshot(6, item_picked=True)) is terminal
    assert evaluator.observe(_snapshot(
        7, item_placed=True, target_slot_attempted="target_left",
    )) is terminal
    assert all(result.stages[name].qualified for name in STAGE_ORDER)
    _validate(result)
    result = evaluator.observe(_snapshot(8, item_placed=True, target_color_matches=True))
    assert result is terminal


@pytest.mark.parametrize("matched", (False, True))
def test_reset_clears_a_terminal_policy_result(matched):
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, item_picked=True))
    for step in (2, 3):
        result = evaluator.observe(_snapshot(
            step, item_placed=True, target_slot_attempted="target_left",
            target_color_matches=matched,
        ))
    assert result.terminal
    assert result.success is matched

    result = evaluator.reset(_snapshot(0, item_placed=True))
    assert not result.terminal and not result.success
    assert not result.stop_execution
    assert result.failure_code is None
    assert all(not stage.qualified for stage in result.stages.values())
    evaluator.observe(_snapshot(1, item_picked=True))
    evaluator.observe(_snapshot(2, item_placed=True, target_color_matches=True))
    result = evaluator.observe(_snapshot(3, item_placed=True, target_color_matches=True))
    assert result.success


@pytest.mark.parametrize(
    ("memory", "domain_success", "violation", "expected"),
    ((False, True, False, False), (True, False, False, False),
     (True, True, True, False), (True, True, False, True)),
)
def test_expert_recording_still_requires_memory_and_domain_protocol(
    memory, domain_success, violation, expected,
):
    evaluator = WR07Evaluator()
    evaluator.reset(_snapshot(0, require_expert_protocol=True))
    evaluator.observe(_snapshot(1, require_expert_protocol=True, item_picked=True))
    result = evaluator.observe(_snapshot(
        2, require_expert_protocol=True, item_picked=True, item_placed=True,
        target_color_matches=True, source_color_memory_latched=memory,
        protocol_violation=violation, domain_success=domain_success,
    ))
    assert result.success is expected
    assert not result.stop_execution
    assert result.details["evaluation"]["mode"] == "expert_recording"
    _validate(result)


def test_expert_recording_keeps_live_checks_after_candidate_success():
    evaluator = WR07Evaluator()
    evaluator.reset(_snapshot(0, require_expert_protocol=True))
    evaluator.observe(_snapshot(1, require_expert_protocol=True, item_picked=True))
    sample = _snapshot(
        2, require_expert_protocol=True, item_picked=True, item_placed=True,
        target_color_matches=True, source_color_memory_latched=True, domain_success=True,
    )
    result = evaluator.observe(sample)
    assert result.success
    assert not result.stop_execution
    assert not evaluator.observe(replace(sample, step_id=3, domain_success=False)).success


@pytest.mark.parametrize("matched", (False, True))
def test_runtime_success_diagnostics_and_reports_share_one_evaluator(matched):
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, item_picked=True))
    placement = _snapshot(
        2, item_placed=True, target_slot_attempted="target_left", target_color_matches=matched,
    )
    evaluator.observe(placement)
    final = replace(placement, step_id=3)
    env = SimpleNamespace(evaluation_snapshot=lambda: final, _wr07_evaluator=evaluator)
    runtime = SimpleNamespace(
        env=env, metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )
    assert success_for_runtime(runtime) is matched
    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["success"] is matched
    assert diagnostics["terminal"]
    assert diagnostics["stop_execution"]
    assert diagnostics["failure_code"] == (None if matched else "wrong_target_placement")
    assert report_stages_for_runtime(runtime)["stages"]["success"]["passed"] is matched
    assert evaluator.result().details["evaluation"]["terminal_step"] == 3


@pytest.mark.parametrize("dwell", (0, -1, 1.5))
def test_invalid_dwell_is_rejected(dwell):
    with pytest.raises(ValueError, match="positive integer"):
        WR07Evaluator(placement_dwell_steps=dwell)
