"""Contract tests for the ordered, seed-dependent TS01 evaluator.

The simulator-facing detector is expected to debounce containment and expose a
``confirmed_sugar_count``.  These tests intentionally start at that boundary:
they specify task ordering, terminal over-add behavior, button qualification,
and the padded three-stage view used by aggregate reports without importing
MuJoCo.
"""

from __future__ import annotations

import pytest

from bench.tasks.ts01.evaluation import (
    SUGAR_STAGE_NAMES,
    TS01EvaluationSnapshot,
    TS01Evaluator,
)


def _snapshot(
    step: int,
    *,
    target: int,
    total: int = 5,
    count: int = 0,
    objects_released: bool = True,
    button_pressed: bool = False,
) -> TS01EvaluationSnapshot:
    return TS01EvaluationSnapshot(
        step_id=step,
        target_sugar_count=target,
        total_sugar_count=total,
        confirmed_sugar_count=count,
        objects_released=objects_released,
        completion_button_pressed=button_pressed,
    )


def _evaluator(target: int) -> TS01Evaluator:
    evaluator = TS01Evaluator(target_sugar_count=target, total_sugar_count=5)
    evaluator.reset(_snapshot(0, target=target))
    return evaluator


@pytest.mark.parametrize("target", (1, 2, 3))
def test_ordered_funnel_adapts_to_each_episode_target(target: int) -> None:
    evaluator = _evaluator(target)

    result = evaluator.result()
    for count in range(1, target + 1):
        result = evaluator.observe(_snapshot(count, target=target, count=count))

    assert {
        name: result.stages[name].applicable for name in SUGAR_STAGE_NAMES
    } == {
        "sugar_1": True,
        "sugar_2": target >= 2,
        "sugar_3": target >= 3,
    }
    assert {
        name: result.stages[name].qualified_step for name in SUGAR_STAGE_NAMES
    } == {
        "sugar_1": 1,
        "sugar_2": 2 if target >= 2 else None,
        "sugar_3": 3 if target >= 3 else None,
    }
    assert not result.success

    result = evaluator.observe(
        _snapshot(
            target + 1,
            target=target,
            count=target,
            button_pressed=True,
        )
    )

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert result.stages["button"].qualified_step == target + 1
    assert result.stages["success"].qualified_step == target + 1


def test_sugar_progress_depends_only_on_confirmed_count_not_pick_identity() -> None:
    evaluator = _evaluator(3)

    # Any number of failed picks leave the evaluator at count zero.  When any
    # cube is eventually confirmed in the saucepan, it satisfies sugar_1; the
    # evaluator does not know or care which attempted cube it was.
    evaluator.observe(_snapshot(1, target=3, count=0))
    evaluator.observe(_snapshot(2, target=3, count=0))
    result = evaluator.observe(_snapshot(3, target=3, count=1))

    assert result.stages["sugar_1"].qualified
    assert result.stages["sugar_1"].qualified_step == 3
    assert not result.stages["sugar_2"].qualified


def test_count_jump_cannot_bypass_the_previous_sugar_stage() -> None:
    evaluator = _evaluator(3)

    jumped = evaluator.observe(_snapshot(1, target=3, count=2))

    assert jumped.stages["sugar_1"].qualified
    assert jumped.stages["sugar_2"].observed
    assert not jumped.stages["sugar_2"].qualified
    assert jumped.stages["sugar_2"].invalidated
    assert jumped.stages["sugar_2"].reason == "prerequisite_not_qualified:sugar_1"

    # A level that remains high is not a new ordered observation.
    held = evaluator.observe(_snapshot(2, target=3, count=2))
    assert not held.stages["sugar_2"].qualified

    # The count must fall below the milestone and rise again after sugar_1 was
    # qualified before sugar_2 can be credited.
    evaluator.observe(_snapshot(3, target=3, count=1))
    recovered = evaluator.observe(_snapshot(4, target=3, count=2))
    assert recovered.stages["sugar_2"].qualified
    assert recovered.stages["sugar_2"].qualified_step == 4


def test_qualified_sugar_history_is_latched_but_button_uses_current_count() -> None:
    evaluator = _evaluator(1)

    evaluator.observe(_snapshot(1, target=1, count=1))
    missing_at_press = evaluator.observe(
        _snapshot(2, target=1, count=0, button_pressed=True)
    )

    assert missing_at_press.stages["sugar_1"].qualified
    assert missing_at_press.stages["button"].observed
    assert not missing_at_press.stages["button"].qualified
    assert missing_at_press.stages["button"].reason == "sugar_goal_not_ready"

    # Restoring the sugar while the same press is held cannot retroactively
    # qualify the button.  It must be released and pressed again.
    evaluator.observe(_snapshot(3, target=1, count=1, button_pressed=True))
    evaluator.observe(_snapshot(4, target=1, count=1, button_pressed=False))
    recovered = evaluator.observe(
        _snapshot(5, target=1, count=1, button_pressed=True)
    )
    assert recovered.success
    assert recovered.stages["button"].qualified_step == 5


def test_button_pressed_before_required_sugar_stages_must_be_pressed_again() -> None:
    evaluator = _evaluator(2)

    early = evaluator.observe(
        _snapshot(1, target=2, count=0, button_pressed=True)
    )
    assert early.stages["button"].observed
    assert not early.stages["button"].qualified
    assert early.stages["button"].reason == "prerequisite_not_qualified:sugar_2"

    evaluator.observe(_snapshot(2, target=2, count=1, button_pressed=True))
    ready_but_held = evaluator.observe(
        _snapshot(3, target=2, count=2, button_pressed=True)
    )
    assert ready_but_held.stages["sugar_2"].qualified
    assert not ready_but_held.stages["button"].qualified

    evaluator.observe(_snapshot(4, target=2, count=2, button_pressed=False))
    retried = evaluator.observe(
        _snapshot(5, target=2, count=2, button_pressed=True)
    )
    assert retried.success
    assert retried.stages["button"].qualified_step == 5


def test_final_sugar_and_button_cannot_qualify_in_the_same_observation() -> None:
    evaluator = _evaluator(1)

    same_step = evaluator.observe(
        _snapshot(1, target=1, count=1, button_pressed=True)
    )

    assert same_step.stages["sugar_1"].qualified
    assert not same_step.stages["button"].qualified
    assert same_step.stages["button"].reason == "prerequisite_not_qualified:sugar_1"

    evaluator.observe(_snapshot(2, target=1, count=1, button_pressed=False))
    retried = evaluator.observe(
        _snapshot(3, target=1, count=1, button_pressed=True)
    )
    assert retried.success


def test_button_requires_all_objects_to_be_released() -> None:
    evaluator = _evaluator(1)

    evaluator.observe(
        _snapshot(1, target=1, count=1, objects_released=False)
    )
    held = evaluator.observe(
        _snapshot(
            2,
            target=1,
            count=1,
            objects_released=False,
            button_pressed=True,
        )
    )

    assert held.stages["sugar_1"].qualified
    assert not held.stages["button"].qualified
    assert held.stages["button"].reason == "sugar_goal_not_ready"

    evaluator.observe(
        _snapshot(
            3,
            target=1,
            count=1,
            objects_released=True,
            button_pressed=False,
        )
    )
    released = evaluator.observe(
        _snapshot(
            4,
            target=1,
            count=1,
            objects_released=True,
            button_pressed=True,
        )
    )
    assert released.success


@pytest.mark.parametrize(
    ("target", "count_before_over_add", "over_added_count"),
    ((1, 1, 2), (2, 1, 3), (3, 2, 4)),
)
def test_over_add_is_immediate_terminal_and_blocks_later_stages(
    target: int,
    count_before_over_add: int,
    over_added_count: int,
) -> None:
    evaluator = _evaluator(target)

    for count in range(1, count_before_over_add + 1):
        evaluator.observe(_snapshot(count, target=target, count=count))
    result = evaluator.observe(
        _snapshot(
            count_before_over_add + 1,
            target=target,
            count=over_added_count,
            button_pressed=True,
        )
    )

    assert not result.success
    assert result.terminal
    assert result.failure_code == "over_add"
    assert result.details["over_add"]["confirmed_sugar_count"] == over_added_count
    assert not result.stages["button"].qualified
    assert not result.stages["success"].qualified
    for index in range(count_before_over_add + 1, 4):
        assert not result.stages[f"sugar_{index}"].qualified


def test_over_add_takes_precedence_before_any_same_step_stage_qualification() -> None:
    evaluator = _evaluator(1)

    result = evaluator.observe(_snapshot(1, target=1, count=2))

    assert result.terminal
    assert result.failure_code == "over_add"
    assert not result.stages["sugar_1"].qualified


@pytest.mark.parametrize("target", (1, 2, 3))
def test_summary_pads_only_unrequired_sugar_stages_after_the_real_goal(
    target: int,
) -> None:
    evaluator = _evaluator(target)

    initial = evaluator.result().to_diagnostics()["stages"]
    for index in range(target + 1, 4):
        assert not initial[f"sugar_{index}"]["effective_success"]

    result = evaluator.result()
    for count in range(1, target + 1):
        result = evaluator.observe(_snapshot(count, target=target, count=count))
    stages = result.to_diagnostics()["stages"]

    for index in range(1, 4):
        stage = stages[f"sugar_{index}"]
        if index <= target:
            assert stage["applicable"]
            assert stage["qualified"]
            assert stage["effective_success"]
            assert stage["completion_mode"] == "physical"
            assert stage["derived_from"] is None
        else:
            assert not stage["applicable"]
            assert not stage["qualified"]
            assert stage["qualified_step"] is None
            assert stage["effective_success"]
            assert stage["completion_mode"] == "not_required"
            assert stage["derived_from"] == f"sugar_{target}"


@pytest.mark.parametrize("target", (1, 2))
def test_over_add_revokes_summary_padding_but_not_real_stage_history(
    target: int,
) -> None:
    evaluator = _evaluator(target)

    for count in range(1, target + 1):
        evaluator.observe(_snapshot(count, target=target, count=count))
    before = evaluator.result().to_diagnostics()["stages"]
    assert all(before[f"sugar_{index}"]["effective_success"] for index in range(1, 4))

    over_added = evaluator.observe(
        _snapshot(target + 1, target=target, count=target + 1)
    )
    after = over_added.to_diagnostics()["stages"]

    for index in range(1, target + 1):
        assert after[f"sugar_{index}"]["qualified"]
        assert after[f"sugar_{index}"]["effective_success"]
    for index in range(target + 1, 4):
        assert not after[f"sugar_{index}"]["qualified"]
        assert not after[f"sugar_{index}"]["effective_success"]
        assert after[f"sugar_{index}"]["completion_mode"] == "blocked_by_over_add"


def test_snapshot_target_cannot_change_during_an_episode() -> None:
    evaluator = _evaluator(1)

    with pytest.raises(ValueError, match="target_sugar_count"):
        evaluator.observe(_snapshot(1, target=2, count=1))


def test_snapshot_total_cannot_change_during_an_episode() -> None:
    evaluator = _evaluator(1)

    with pytest.raises(ValueError, match="total_sugar_count"):
        evaluator.observe(_snapshot(1, target=1, total=4, count=1))


def test_snapshot_rejects_confirmed_count_above_total() -> None:
    with pytest.raises(ValueError, match="cannot exceed total_sugar_count"):
        _snapshot(0, target=1, total=1, count=2)
