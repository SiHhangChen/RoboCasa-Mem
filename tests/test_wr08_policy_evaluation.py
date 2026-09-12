"""Policy histories, two-carton outcomes, and irreversible WR-08 stops."""

from dataclasses import replace

import pytest

from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr08.domain.contracts import ITEM_IDS, TARGET_SLOTS
from bench.tasks.wr08.domain.episode import resolve_episode_plan
from bench.tasks.wr08.evaluation import STAGE_ORDER, WR08EvaluationSnapshot, WR08Evaluator


PLAN = resolve_episode_plan(seed=0)
TARGETS = {item: PLAN.target_slot_by_color[PLAN.source_color_by_item[item]] for item in ITEM_IDS}


def _snapshot(step, *, picked=(), placements=None, held=(), **kwargs):
    placements = dict(placements or {})
    return WR08EvaluationSnapshot(
        step_id=step,
        transport_order=ITEM_IDS,
        source_slot_by_item=PLAN.source_slot_by_item,
        source_color_by_item=PLAN.source_color_by_item,
        target_slot_by_item=TARGETS,
        item_picked_by_item={item: item in picked for item in ITEM_IDS},
        item_placed_by_item={item: item in placements for item in ITEM_IDS},
        item_matched_by_item={item: placements.get(item) == TARGETS[item] for item in ITEM_IDS},
        released_by_item={item: item not in held for item in ITEM_IDS},
        target_slot_attempted_by_item={item: slot for item, slot in placements.items() if slot},
        **kwargs,
    )


def _evaluator(**kwargs):
    evaluator = WR08Evaluator(**kwargs)
    evaluator.reset(_snapshot(0))
    return evaluator


def _sequence(order=ITEM_IDS):
    first, second = order
    first_placement = {first: TARGETS[first]}
    return [
        _snapshot(1, picked=(first,), held=(first,)),
        _snapshot(2, placements=first_placement),
        _snapshot(3, placements=first_placement),
        _snapshot(4, picked=(second,), held=(second,), placements=first_placement),
        _snapshot(5, placements=TARGETS),
        _snapshot(6, placements=TARGETS),
    ]


@pytest.mark.parametrize("order", (ITEM_IDS, ITEM_IDS[::-1]))
def test_both_transport_orders_work_without_expert_memory_or_callbacks(order):
    evaluator = _evaluator()
    for snapshot in _sequence(order):
        result = evaluator.observe(snapshot)
        validate_report_stages(
            result.to_report_stages(), expected_task_id="WR-08",
            expected_profile_id="WR-08/report-v3", expected_success=result.success,
        )
        assert result.stop_execution is (snapshot.step_id == 6)
        if snapshot.step_id == 3:
            assert result.stages["one_item_matched"].qualified
            assert not result.terminal
    assert result.success and result.terminal and result.failure_code is None
    assert [result.stages[name].qualified_step for name in STAGE_ORDER] == [1, 3, 3, 4, 6, 6, 6]
    diagnostics = result.to_diagnostics()
    assert diagnostics["stop_execution"]
    assert diagnostics["evaluation"]["terminal_step"] == 6
    assert diagnostics["evaluation"]["observed_pick_order"] == list(order)
    assert diagnostics["evaluation"]["transport_order_required"] is False
    assert not any("color_latched" in name for name in diagnostics["stage_checks"])


def test_expert_protocol_flags_do_not_gate_the_policy():
    evaluator = _evaluator()
    for snapshot in _sequence():
        result = evaluator.observe(replace(snapshot, protocol_violation=True, domain_success=False))
    assert result.success and result.stop_execution


@pytest.mark.parametrize("reset_completed", (False, True))
def test_resting_at_targets_or_domain_success_cannot_bypass_pickup(reset_completed):
    evaluator = _evaluator()
    completed = _snapshot(
        0, picked=ITEM_IDS, placements=TARGETS,
        memory_latched_by_item={item: True for item in ITEM_IDS}, domain_success=True,
    )
    if reset_completed:
        evaluator.reset(completed)
    for step in (1, 2, 3):
        result = evaluator.observe(replace(completed, step_id=step))
        assert not result.terminal and not result.stop_execution
        assert not any(stage.qualified for stage in result.stages.values())


@pytest.mark.parametrize("item_index", (0, 1), ids=("first_carton", "second_carton"))
def test_any_wrong_target_is_terminal_and_cannot_be_corrected(item_index):
    evaluator = _evaluator()
    for snapshot in _sequence()[:1 if item_index == 0 else 4]:
        evaluator.observe(snapshot)
    item = ITEM_IDS[item_index]
    wrong = next(slot for slot in TARGET_SLOTS if slot != TARGETS[item])
    step = 2 if item_index == 0 else 5
    placements = {} if item_index == 0 else {ITEM_IDS[0]: TARGETS[ITEM_IDS[0]]}
    placements[item] = wrong
    assert not evaluator.observe(_snapshot(step, placements=placements)).terminal
    failed = evaluator.observe(_snapshot(step + 1, placements=placements))
    assert failed.terminal and failed.stop_execution and not failed.success
    assert failed.failure_code == "wrong_target_placement"
    assert failed.stages["one_item_placed"].qualified
    assert failed.details["evaluation"]["failure_item"] == item
    assert evaluator.observe(_snapshot(step + 2, placements=TARGETS, domain_success=True)) is failed


def test_success_is_frozen_until_reset():
    evaluator = _evaluator()
    for snapshot in _sequence():
        final = evaluator.observe(snapshot)
    assert final.success and final.stop_execution
    assert evaluator.observe(_snapshot(7, protocol_violation=True)) is final
    reset = evaluator.reset(_snapshot(0))
    assert not reset.success and not reset.stop_execution
    assert all(not stage.qualified for stage in reset.stages.values())


@pytest.mark.parametrize("target_correct", (False, True))
def test_carried_carton_contacting_a_target_is_not_a_placement(target_correct):
    evaluator = _evaluator()
    first = ITEM_IDS[0]
    slot = TARGETS[first] if target_correct else next(s for s in TARGET_SLOTS if s != TARGETS[first])
    evaluator.observe(_snapshot(1, picked=(first,), held=(first,)))
    for step in (2, 3, 4):
        result = evaluator.observe(_snapshot(step, placements={first: slot}, held=(first,)))
    assert not result.stages["one_item_placed"].qualified
    assert not result.terminal


def test_source_or_counter_putdown_counts_without_being_a_wrong_target():
    evaluator = _evaluator()
    first = ITEM_IDS[0]
    evaluator.observe(_snapshot(1, picked=(first,)))
    for step in (2, 3):
        result = evaluator.observe(_snapshot(step, placements={first: None}))
    assert result.stages["one_item_placed"].qualified
    assert not result.stages["one_item_matched"].qualified
    assert not result.terminal
    for step in (4, 5):
        result = evaluator.observe(_snapshot(step, placements={first: TARGETS[first]}))
    assert result.stages["one_item_matched"].qualified
    assert not result.terminal


@pytest.mark.parametrize("interruption", ("contact_loss", "held", "different_wrong_plate", "step_gap"))
def test_wrong_target_requires_the_same_plate_on_consecutive_steps(interruption):
    evaluator = _evaluator()
    item = ITEM_IDS[0]
    wrong, other = [slot for slot in TARGET_SLOTS if slot != TARGETS[item]]
    evaluator.observe(_snapshot(1, picked=(item,)))
    evaluator.observe(_snapshot(2, placements={item: wrong}))
    if interruption == "contact_loss":
        evaluator.observe(_snapshot(3))
    elif interruption == "held":
        evaluator.observe(_snapshot(3, placements={item: wrong}, held=(item,)))
    elif interruption == "different_wrong_plate":
        evaluator.observe(_snapshot(3, placements={item: other}))
    assert not evaluator.observe(_snapshot(4, placements={item: wrong})).terminal
    assert evaluator.observe(_snapshot(5, placements={item: wrong})).stop_execution


def test_duplicate_reads_do_not_accumulate_dwell_and_backwards_steps_fail():
    evaluator = _evaluator()
    item = ITEM_IDS[0]
    evaluator.observe(_snapshot(1, picked=(item,)))
    snapshot = _snapshot(2, placements={item: TARGETS[item]})
    first = evaluator.observe(snapshot)
    for _ in range(5):
        assert evaluator.observe(snapshot) is first
    assert not first.stages["one_item_placed"].qualified
    with pytest.raises(ValueError, match="backwards"):
        evaluator.observe(replace(snapshot, step_id=1))
    assert evaluator.observe(replace(snapshot, step_id=3)).stages["one_item_matched"].qualified


def test_both_cartons_must_still_be_on_their_targets_at_final_success():
    evaluator = _evaluator()
    first, second = ITEM_IDS
    for snapshot in _sequence()[:4]:
        evaluator.observe(snapshot)
    for step in (5, 6):
        result = evaluator.observe(_snapshot(step, placements={second: TARGETS[second]}))
    assert result.stages["two_items_matched"].qualified
    assert not result.success and not result.stop_execution
    assert not evaluator.observe(_snapshot(7, placements=TARGETS)).success
    assert evaluator.observe(_snapshot(8, placements=TARGETS)).success


def test_first_carton_wrongly_replaced_while_moving_second_stops_everything():
    evaluator = _evaluator()
    first, second = ITEM_IDS
    for snapshot in _sequence()[:3]:
        evaluator.observe(snapshot)
    wrong = next(slot for slot in TARGET_SLOTS if slot != TARGETS[first])
    evaluator.observe(_snapshot(4, picked=(second,), placements={first: wrong}))
    result = evaluator.observe(_snapshot(5, placements={first: wrong}))
    assert result.stop_execution and result.failure_code == "wrong_target_placement"
    assert result.details["evaluation"]["failure_item"] == first


def test_interleaved_pickups_count_each_carton_once_and_keep_a_valid_report_funnel():
    evaluator = _evaluator()
    first, second = ITEM_IDS
    evaluator.observe(_snapshot(1, picked=(first,)))
    evaluator.observe(_snapshot(2, picked=(second,)))
    for step in (3, 4):
        result = evaluator.observe(_snapshot(step, placements={second: TARGETS[second]}))
    assert result.stages["one_item_matched"].qualified
    assert result.stages["two_items_picked"].qualified
    assert result.details["evaluation"]["items"][first]["picked_step"] == 1
    assert result.details["evaluation"]["items"][second]["picked_step"] == 2
    for step in (5, 6):
        result = evaluator.observe(_snapshot(step, placements=TARGETS))
    assert result.success
    validate_report_stages(result.to_report_stages(), expected_success=True)


def test_matching_bit_without_explicit_support_fails_closed():
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, picked=ITEM_IDS))
    for step in (2, 3):
        result = evaluator.observe(replace(_snapshot(step, placements=TARGETS), item_placed_by_item={}))
    assert not result.terminal
    assert not result.stages["one_item_placed"].qualified


@pytest.mark.parametrize("invalid", (0, -1, 1.5))
def test_invalid_placement_dwell_is_rejected(invalid):
    with pytest.raises(ValueError, match="positive integer"):
        WR08Evaluator(placement_dwell_steps=invalid)


def test_mode_changes_require_an_explicit_reset():
    evaluator = _evaluator()
    with pytest.raises(ValueError, match="mode changed"):
        evaluator.observe(replace(_snapshot(1), require_expert_protocol=True))
