from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bench.evaluation.expert_prefix import ExpertPrefixArtifact
from bench.evaluation.stage_integration import get_stage_evaluation_integration
from bench.evaluation.stage_rollout import StageTracker
from bench.tasks.ts04.domain.contracts import FIXTURE_NAMES
from bench.tasks.ts04.evaluation import TS04EvaluationSnapshot, TS04Evaluator
from bench.tasks.ts04.stage_evaluation import (
    BEFORE_TARGET_DRAWER,
    TARGET_DRAWER_RETURNED,
    build_expert_stage_boundaries,
    build_stage_specs,
    privileged_stage_state,
    stage_completed,
)


def _snapshot(
    step: int,
    *,
    history: tuple[str, ...] = (),
    button_steps: tuple[int, ...] = (),
    actual_button_count: int | None = None,
    success: bool = False,
    failure_reason: str | None = None,
    search_after_target: bool = False,
) -> TS04EvaluationSnapshot:
    target = FIXTURE_NAMES[0]
    found = target in history
    return TS04EvaluationSnapshot(
        step_id=step,
        fixture_names=FIXTURE_NAMES,
        target_fixture=target,
        completed_fixture_history=history,
        completed_fixture_steps={name: index for index, name in enumerate(history, start=1)},
        target_found=found,
        search_after_target_detected=search_after_target,
        valid_button_press_steps=button_steps,
        actual_button_press_count=(
            len(button_steps) if actual_button_count is None else actual_button_count
        ),
        required_button_press_count=len(history) if found else 0,
        domain_success=success,
        failure_reason=failure_reason,
    )


def _evaluator() -> TS04Evaluator:
    evaluator = TS04Evaluator(
        fixture_names=FIXTURE_NAMES,
        target_fixture=FIXTURE_NAMES[0],
    )
    evaluator.reset(_snapshot(0))
    return evaluator


def test_ts04_stage_integration_matches_ts02_handoff_protocol() -> None:
    integration = get_stage_evaluation_integration("TS04")
    ts02 = get_stage_evaluation_integration("TS02")

    assert integration.task_id == "TS04"
    assert (
        integration.target_stage_ids
        == ts02.target_stage_ids
        == (
            "one_drawer_checked",
            "target_found",
            "search_stopped_after_target",
            "button_pressed",
            "correct_press_count",
        )
    )
    assert (
        integration.prefix_stage_by_target
        == ts02.prefix_stage_by_target
        == {
            "one_drawer_checked": None,
            "target_found": BEFORE_TARGET_DRAWER,
            "search_stopped_after_target": BEFORE_TARGET_DRAWER,
            "button_pressed": TARGET_DRAWER_RETURNED,
            "correct_press_count": TARGET_DRAWER_RETURNED,
        }
    )
    assert integration.capture_stage_ids == ts02.capture_stage_ids
    assert integration.completion_stage_by_target == ts02.completion_stage_by_target
    integration.validate_stage_suite(integration.target_stage_ids)

    for target in integration.target_stage_ids:
        (spec,) = integration.build_stage_specs(None, target)
        (ts02_spec,) = ts02.build_stage_specs(None, target)
        assert spec.id == target
        assert spec.model_step_budget == ts02_spec.model_step_budget
        assert spec.success_hold_steps == ts02_spec.success_hold_steps == 1


@pytest.mark.parametrize("order", (FIXTURE_NAMES, tuple(reversed(FIXTURE_NAMES))))
@pytest.mark.parametrize("search_length", (1, 2))
def test_ts04_handoffs_follow_actual_expert_search_history(
    order: tuple[str, ...],
    search_length: int,
) -> None:
    checked = order[:search_length]
    plan = SimpleNamespace(metadata={"fixture_order": checked, "target_fixture": checked[-1]})
    stage_end_steps = {
        f"return_arm_after_{fixture}": 100 * index for index, fixture in enumerate(checked, start=1)
    }
    stage_end_steps[f"close_{checked[-1]}"] = 100 * search_length - 20
    boundaries = build_expert_stage_boundaries(None, plan, stage_end_steps)
    assert boundaries == {
        BEFORE_TARGET_DRAWER: 100 * (search_length - 1),
        TARGET_DRAWER_RETURNED: 100 * search_length,
    }

    actions = np.arange((100 * search_length + 1) * 13, dtype=np.float32).reshape(-1, 13)
    artifact = ExpertPrefixArtifact(
        task_id="TS04",
        seed=0,
        episode_identity="episode",
        expert_plan_version="expert",
        stage_boundary_steps=boundaries,
        actions=actions,
        success=True,
    )
    integration = get_stage_evaluation_integration("TS04")
    expected_lengths = {
        "one_drawer_checked": 0,
        "target_found": 100 * (search_length - 1),
        "search_stopped_after_target": 100 * (search_length - 1),
        "button_pressed": 100 * search_length,
        "correct_press_count": 100 * search_length,
    }
    for target, length in expected_lengths.items():
        np.testing.assert_array_equal(
            integration.prefix_actions(artifact, target), actions[:length]
        )


def test_ts04_does_not_invent_missing_arm_return_boundaries() -> None:
    first, target = FIXTURE_NAMES
    plan = SimpleNamespace(metadata={"fixture_order": FIXTURE_NAMES, "target_fixture": target})
    assert build_expert_stage_boundaries(None, plan, {f"close_{target}": 200}) == {}
    assert build_expert_stage_boundaries(
        None, plan, {f"return_arm_after_{first}": 100, f"close_{target}": 200}
    ) == {BEFORE_TARGET_DRAWER: 100}


@pytest.mark.parametrize("order", ((), tuple(reversed(FIXTURE_NAMES))))
def test_ts04_expert_prefix_must_end_its_search_at_the_target(order) -> None:
    plan = SimpleNamespace(metadata={"fixture_order": order, "target_fixture": FIXTURE_NAMES[1]})
    with pytest.raises(ValueError, match="must end at target_fixture"):
        build_expert_stage_boundaries(None, plan, {})


def test_ts04_stage_specs_accept_budget_overrides_and_validate_targets() -> None:
    (spec,) = build_stage_specs({"correct_press_count": 25})
    assert spec.id == "correct_press_count"
    assert spec.model_step_budget == 25

    with pytest.raises(ValueError, match="unknown TS04 stage budgets"):
        build_stage_specs({"success": 10})
    with pytest.raises(KeyError, match="unknown TS04 target stage"):
        build_stage_specs(target_stage="success")
    with pytest.raises(ValueError, match="model_step_budget must be positive"):
        build_stage_specs({"correct_press_count": 0})


def test_ts04_stage_predicates_use_qualified_verdicts_not_observed_history() -> None:
    evaluator = _evaluator()
    target = FIXTURE_NAMES[0]
    found = evaluator.observe(_snapshot(1, history=(target,)))
    for stage in ("one_drawer_checked", "target_found", "search_stopped_after_target"):
        assert stage_completed(found, stage)
    assert not stage_completed(found, "button_pressed")
    assert not stage_completed(found, "correct_press_count")
    for boundary in (BEFORE_TARGET_DRAWER, TARGET_DRAWER_RETURNED):
        assert not stage_completed(found, boundary)
    with pytest.raises(KeyError, match="unknown TS04 stage"):
        stage_completed(found, "success")

    passed = evaluator.observe(_snapshot(2, history=(target,), button_steps=(2,), success=True))
    assert stage_completed(passed, "correct_press_count")
    extra = evaluator.observe(
        _snapshot(
            3,
            history=(target,),
            button_steps=(2,),
            actual_button_count=2,
            failure_reason="completion_button_pressed_too_many_times",
        )
    )
    assert extra.stages["correct_press_count"].observed
    assert not stage_completed(extra, "correct_press_count")
    tracker = StageTracker(build_stage_specs(target_stage="correct_press_count"))
    tracker.reset(found)
    tracker.update(extra)
    assert tracker.target_failed("correct_press_count")
    assert not tracker.target_completed("correct_press_count")


def test_ts04_stage_predicates_preserve_search_stop_revocation() -> None:
    evaluator = _evaluator()
    target = FIXTURE_NAMES[0]
    found = evaluator.observe(_snapshot(1, history=(target,)))
    (spec,) = build_stage_specs(target_stage="search_stopped_after_target")
    assert spec.completion_predicate(found)
    continued = evaluator.observe(
        _snapshot(
            2,
            history=(target,),
            search_after_target=True,
            failure_reason="search_continued_after_target_found",
        )
    )
    assert not spec.completion_predicate(continued)
    assert spec.failure_predicate(continued)


@pytest.mark.parametrize("wrapped", (False, True))
def test_ts04_privileged_stage_state_uses_task_local_evaluator(wrapped: bool) -> None:
    evaluator = _evaluator()
    snapshot = _snapshot(1, history=(FIXTURE_NAMES[0],))
    shared = SimpleNamespace(
        metadata={"task_evaluator": evaluator},
        env=SimpleNamespace(evaluation_snapshot=lambda: snapshot),
    )
    runtime = SimpleNamespace(metadata={"shared_runtime": shared}) if wrapped else shared
    result = privileged_stage_state(runtime)

    assert result is evaluator.result()
    assert result.stages["target_found"].qualified
    assert not result.success
    assert result.to_report_stages()["task_id"] == "TS04"
