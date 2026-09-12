"""Preserve ordered WR-08 expert verification while updating public reports."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr08.domain.contracts import ITEM_IDS, TARGET_SLOTS
from bench.tasks.wr08.domain.episode import resolve_episode_plan
from bench.tasks.wr08.evaluation import (
    EXPERT_STAGE_ORDER,
    WR08EvaluationSnapshot,
    WR08Evaluator,
)


PLAN = resolve_episode_plan(seed=0)
TARGETS = {
    item: PLAN.target_slot_by_color[PLAN.source_color_by_item[item]]
    for item in ITEM_IDS
}


def _snapshot(
    step,
    *,
    order=ITEM_IDS,
    picked=(),
    latched=(),
    matched=(),
    held=(),
    success=False,
    attempts=None,
    protocol_violation=False,
):
    return WR08EvaluationSnapshot(
        step_id=step,
        transport_order=order,
        source_slot_by_item=PLAN.source_slot_by_item,
        source_color_by_item=PLAN.source_color_by_item,
        target_slot_by_item=TARGETS,
        item_picked_by_item={item: item in picked for item in ITEM_IDS},
        memory_latched_by_item={item: item in latched for item in ITEM_IDS},
        item_matched_by_item={item: item in matched for item in ITEM_IDS},
        released_by_item={item: item not in held for item in ITEM_IDS},
        target_slot_attempted_by_item=(
            {item: TARGETS[item] for item in matched} if attempts is None else attempts
        ),
        domain_success=success,
        protocol_violation=protocol_violation,
        require_expert_protocol=True,
    )


def _evaluator(order=ITEM_IDS):
    evaluator = WR08Evaluator(
        transport_order=order,
        source_slot_by_item=PLAN.source_slot_by_item,
        source_color_by_item=PLAN.source_color_by_item,
        target_slot_by_item=TARGETS,
    )
    evaluator.reset(_snapshot(0, order=order))
    return evaluator


def _ordered_snapshots(order=ITEM_IDS):
    first, _ = order
    return [
        _snapshot(1, order=order, picked=(first,), latched=(first,), held=(first,)),
        _snapshot(2, order=order, picked=(first,), latched=(first,), matched=(first,)),
        _snapshot(3, order=order, picked=order, latched=order, matched=(first,)),
        _snapshot(
            4, order=order, picked=order, latched=order, matched=order, success=True
        ),
    ]


@pytest.mark.parametrize("order", [ITEM_IDS, ITEM_IDS[::-1]])
def test_wr08_ordered_transport_qualifies_all_stages_and_each_partial_report(order):
    evaluator = _evaluator(order)
    for snapshot in _ordered_snapshots(order):
        result = evaluator.observe(snapshot)
        report = validate_report_stages(
            result.to_report_stages(),
            expected_task_id="WR-08",
            expected_profile_id="WR-08/report-v3",
            expected_success=result.success,
        )
        assert report["order"] == [
            "one_item_picked",
            "one_item_placed",
            "one_item_matched",
            "two_items_picked",
            "two_items_placed",
            "two_items_matched",
            "success",
        ]
    assert result.success
    assert result.terminal
    assert not result.stop_execution
    assert result.failure_code is None
    assert all(result.stages[name].qualified for name in EXPERT_STAGE_ORDER)
    assert result.stages["first_item_picked"].qualified_step == 1
    assert result.stages["first_item_matched"].qualified_step == 2
    assert result.stages["second_item_picked"].qualified_step == 3
    assert result.stages["second_item_matched"].qualified_step == 4


def test_wr08_completed_state_present_at_reset_is_not_a_successful_history():
    evaluator = _evaluator()
    snapshot = _snapshot(
        0, picked=ITEM_IDS, latched=ITEM_IDS, matched=ITEM_IDS, success=True
    )
    result = evaluator.reset(snapshot)
    assert not result.success
    assert all(not stage.qualified for stage in result.stages.values())
    result = evaluator.observe(replace(snapshot, step_id=1))
    assert not result.success
    assert result.stages["success"].reason == "present_at_reset"


def test_wr08_recording_keeps_live_checks_after_candidate_success():
    evaluator = _evaluator()
    for snapshot in _ordered_snapshots():
        result = evaluator.observe(snapshot)
    assert result.success and not result.stop_execution
    later = evaluator.observe(_snapshot(5, picked=ITEM_IDS, latched=ITEM_IDS))
    assert not later.success and not later.stop_execution


def test_wr08_second_item_cannot_be_credited_before_the_first_is_matched():
    evaluator = _evaluator()
    second = ITEM_IDS[1]
    result = evaluator.observe(
        _snapshot(1, picked=(second,), latched=(second,), held=(second,))
    )
    stage = result.stages["second_item_picked"]
    assert stage.observed and not stage.qualified
    assert stage.reason == "prerequisite_not_qualified:first_item_matched"
    # A persistent prematurely observed bit must not be credited later.
    for snapshot in _ordered_snapshots():
        result = evaluator.observe(
            replace(
                snapshot,
                step_id=snapshot.step_id + 1,
                item_picked_by_item={**snapshot.item_picked_by_item, second: True},
                memory_latched_by_item={
                    **snapshot.memory_latched_by_item,
                    second: True,
                },
            )
        )
    assert not result.success
    assert not result.stages["second_item_picked"].qualified


def test_wr08_memory_latch_cannot_precede_picking():
    evaluator = _evaluator()
    result = evaluator.observe(_snapshot(1, latched=(ITEM_IDS[0],)))
    assert not result.stages["first_item_color_latched"].qualified
    assert not result.success


def test_wr08_wrong_target_failure_cannot_be_repaired_by_later_good_bits():
    evaluator = _evaluator()
    first = ITEM_IDS[0]
    evaluator.observe(_ordered_snapshots()[0])
    wrong = next(slot for slot in TARGET_SLOTS if slot != TARGETS[first])
    result = evaluator.observe(
        _snapshot(
            2,
            picked=(first,),
            latched=(first,),
            matched=(first,),
            attempts={first: wrong},
        )
    )
    assert not result.success
    assert result.terminal
    assert result.failure_code == "wrong_target_placement"
    result = evaluator.observe(
        _snapshot(3, picked=ITEM_IDS, latched=ITEM_IDS, matched=ITEM_IDS, success=True)
    )
    assert not result.success
    assert result.failure_code == "wrong_target_placement"


def test_wr08_held_carton_is_not_a_completed_placement():
    evaluator = _evaluator()
    first = ITEM_IDS[0]
    evaluator.observe(_ordered_snapshots()[0])
    result = evaluator.observe(
        _snapshot(2, picked=(first,), latched=(first,), matched=(first,), held=(first,))
    )
    assert not result.stages["first_item_matched"].qualified
    assert result.stages["first_item_matched"].reason == "item_not_released"
    assert not result.success


def test_wr08_protocol_violation_overrides_an_inconsistent_domain_success():
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(
            1,
            picked=ITEM_IDS,
            latched=ITEM_IDS,
            matched=ITEM_IDS,
            success=True,
            protocol_violation=True,
        )
    )
    assert result.terminal
    assert result.failure_code == "protocol_violation"
    assert not result.success
    assert not result.stages["success"].qualified


def test_wr08_domain_success_still_requires_the_ordered_transport_history():
    evaluator = _evaluator()
    result = evaluator.observe(_snapshot(1, success=True))
    assert not result.success
    assert not result.stages["success"].qualified


def test_wr08_duplicate_steps_are_idempotent_and_backwards_steps_are_rejected():
    evaluator = _evaluator()
    first = _ordered_snapshots()[0]
    result = evaluator.observe(first)
    assert evaluator.observe(first) is result
    with pytest.raises(ValueError, match="backwards"):
        evaluator.observe(replace(first, step_id=0))


def test_wr08_episode_bindings_cannot_change_during_evaluation():
    evaluator = _evaluator()
    first = ITEM_IDS[0]
    with pytest.raises(ValueError, match="configuration changed"):
        evaluator.observe(
            replace(
                _snapshot(1),
                source_color_by_item={**PLAN.source_color_by_item, first: "cyan"},
            )
        )
