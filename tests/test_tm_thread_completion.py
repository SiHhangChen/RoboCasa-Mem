"""Order-independent, five-stage reporting for TM01 and TM02."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bench.evaluation.report_profiles import get_report_stage_profile
from bench.evaluation.report_stages import validate_report_stages
from bench.evaluation.summary import aggregate_episode_results
from bench.tasks.tm01.evaluation import TM01EvaluationSnapshot, TM01Evaluator
from bench.tasks.tm02.evaluation import TM02EvaluationSnapshot, TM02Evaluator


@pytest.fixture(params=("TM01", "TM02"))
def task_id(request) -> str:
    return request.param


def _snapshot(task_id: str, step: int, events: dict[str, int], **kwargs):
    common = {
        "step_id": step,
        "event_steps": events,
        "completion_button_pressed": "completion_button" in events,
        **kwargs,
    }
    if task_id == "TM01":
        return TM01EvaluationSnapshot(
            burner_on_completed="A1" in events,
            one_drawer_inspected="B1" in events,
            boiling_thread_completed="A2" in events,
            drawer_thread_completed="B2" in events,
            **common,
        )
    return TM02EvaluationSnapshot(
        faucet_on_completed="A1" in events,
        pot_filled_and_can_retrieved="B1" in events,
        fill_pot_thread_completed="A2" in events,
        retrieve_can_thread_completed="B2" in events,
        **common,
    )


def _evaluator(task_id: str):
    evaluator = TM01Evaluator() if task_id == "TM01" else TM02Evaluator()
    evaluator.reset(_snapshot(task_id, 0, {}))
    return evaluator


@pytest.mark.parametrize("first_event", ("A2", "B2"))
def test_each_tail_event_advances_one_stage_without_imposing_order(
    task_id: str, first_event: str,
) -> None:
    evaluator = _evaluator(task_id)
    events = {"A1": 2, "B1": 5}
    before = evaluator.observe(_snapshot(task_id, 5, events))
    assert not before.stages["one_thread_completed"].qualified
    assert before.to_diagnostics()["thread_completion"]["completed_count"] == 0

    events = {**events, first_event: 8}
    snapshot = _snapshot(task_id, 8, events)
    first = evaluator.observe(snapshot)
    assert evaluator.observe(snapshot) is first
    assert first.stages["one_thread_completed"].qualified
    assert first.stages["one_thread_completed"].qualified_step == 8
    assert not first.stages["both_threads_completed"].qualified
    assert not first.success
    assert not first.terminal
    second_event = "B2" if first_event == "A2" else "A2"
    progress = first.to_diagnostics()["thread_completion"]
    assert progress["completed_count"] == 1
    assert progress[first_event] == {"completed": True, "completed_step": 8}
    assert progress[second_event] == {"completed": False, "completed_step": None}
    validate_report_stages(first.to_report_stages(), expected_success=False)

    events = {**events, second_event: 16}
    both = evaluator.observe(_snapshot(task_id, 16, events))
    assert both.stages["one_thread_completed"].qualified_step == 8
    assert both.stages["one_thread_completed"].qualified
    assert both.stages["both_threads_completed"].qualified_step == 16
    assert both.to_diagnostics()["thread_completion"]["completed_count"] == 2
    assert not both.success
    assert not both.terminal

    success = evaluator.observe(_snapshot(
        task_id, 20, {**events, "completion_button": 20}, domain_success=True,
    ))
    assert success.success and success.terminal
    assert all(stage.qualified for stage in success.stages.values())
    assert len(success.stages) == 5
    assert success.to_report_stages()["profile_id"] == f"{task_id}/report-v2"
    validate_report_stages(success.to_report_stages(), expected_success=True)


@pytest.mark.parametrize("first_event", ("A2", "B2"))
def test_late_read_uses_earlier_and_later_semantic_event_steps(
    task_id: str, first_event: str,
) -> None:
    other = "B2" if first_event == "A2" else "A2"
    result = _evaluator(task_id).observe(_snapshot(
        task_id, 200, {"A1": 2, "B1": 5, first_event: 100, other: 160},
    ))
    assert result.stages["one_thread_completed"].qualified_step == 100
    assert result.stages["both_threads_completed"].qualified_step == 160


@pytest.mark.parametrize("event", ("A2", "B2"))
def test_tail_flags_do_not_bypass_b1_prerequisite(task_id: str, event: str) -> None:
    result = _evaluator(task_id).observe(_snapshot(task_id, 8, {"A1": 2, event: 8}))
    stage = result.stages["one_thread_completed"]
    assert stage.observed and not stage.qualified
    assert stage.reason.startswith("prerequisite_not_qualified:")
    assert not result.stages["both_threads_completed"].qualified
    assert not result.success


def test_s3_evidence_ignores_an_event_without_a_completed_thread(task_id: str) -> None:
    snapshot = _snapshot(task_id, 100, {"A1": 2, "B1": 5, "A2": 80, "B2": 10})
    field = "drawer_thread_completed" if task_id == "TM01" else "retrieve_can_thread_completed"
    snapshot = replace(snapshot, **{field: False})
    result = _evaluator(task_id).observe(snapshot)
    assert result.stages["one_thread_completed"].qualified_step == 80
    assert result.to_diagnostics()["thread_completion"]["B2"]["completed_step"] is None


def test_failure_preserves_partial_progress_without_granting_success(task_id: str) -> None:
    evaluator = _evaluator(task_id)
    events = {"A1": 2, "B1": 5, "B2": 8}
    evaluator.observe(_snapshot(task_id, 8, events))
    result = evaluator.observe(_snapshot(
        task_id, 9, events, absorbing_failure=True, failure_code="premature_completion",
    ))
    assert result.terminal and not result.success
    assert result.failure_code == "premature_completion"
    assert result.stages["one_thread_completed"].qualified_step == 8
    assert not result.stages["both_threads_completed"].qualified
    validate_report_stages(result.to_report_stages(), expected_success=False)


def test_tm_profile_v1_is_readable_but_not_the_default(task_id: str) -> None:
    current = get_report_stage_profile(task_id)
    legacy = get_report_stage_profile(task_id, profile_id=f"{task_id}/report-v1")
    assert current.profile_id == f"{task_id}/report-v2"
    assert len(current.stage_ids) == 5
    assert len(legacy.stage_ids) == 4
    assert "one_thread_completed" not in legacy.stage_ids
    report = {
        "schema_version": "membench.report_stages/v1",
        "task_id": task_id,
        "profile_id": legacy.profile_id,
        "order": list(legacy.stage_ids),
        "stages": {
            name: {"applicable": True, "passed": True, "first_pass_step": i}
            for i, name in enumerate(legacy.stage_ids, start=1)
        },
    }
    validate_report_stages(report, expected_success=True)
    with pytest.raises(ValueError, match="profile mismatch"):
        validate_report_stages(report, expected_profile_id=current.profile_id)
    other_task = "TM02" if task_id == "TM01" else "TM01"
    with pytest.raises(KeyError):
        get_report_stage_profile(other_task, profile_id=legacy.profile_id)

    current_report = _evaluator(task_id).observe(_snapshot(
        task_id, 20,
        {"A1": 2, "B1": 5, "A2": 8, "B2": 16, "completion_button": 20},
        domain_success=True,
    )).to_report_stages()
    records = [
        {"task_id": task_id, "seed": seed, "status": "success",
         "success": True, "report_stages": value}
        for seed, value in enumerate((report, current_report))
    ]
    legacy_summary = aggregate_episode_results(records[:1])["report_stages"]
    assert legacy_summary["profile_id"] == legacy.profile_id
    with pytest.raises(ValueError, match="incompatible report-stage profiles"):
        aggregate_episode_results(records)
