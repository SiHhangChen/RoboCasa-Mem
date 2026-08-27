from __future__ import annotations

from types import SimpleNamespace

import pytest

import bench.tasks.tm01.runtime as tm01_runtime
import bench.tasks.tm02.runtime as tm02_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.tm01.domain.semantics import SemanticState as TM01State
from bench.tasks.tm01.evaluation import (
    TM01EvaluationSnapshot,
    TM01Evaluator,
    report_stages_for_runtime as tm01_report,
)
from bench.tasks.tm02.domain.semantics import SemanticState as TM02State
from bench.tasks.tm02.evaluation import (
    TM02EvaluationSnapshot,
    TM02Evaluator,
    report_stages_for_runtime as tm02_report,
)


class _Environment:
    def __init__(self, state: object, event_steps: dict[str, int], success: bool):
        self.physical_observer = SimpleNamespace(
            state=state,
            event_steps=dict(event_steps),
        )
        self._success = bool(success)
        self._evaluation_snapshot = None
        if isinstance(state, TM01State):
            self._evaluation_snapshot = TM01EvaluationSnapshot(
                step_id=max(event_steps.values(), default=1),
                burner_on_completed=state.a1_complete,
                one_drawer_inspected=state.b1_complete,
                boiling_thread_completed=state.a2_complete,
                drawer_thread_completed=state.b2_complete,
                completion_button_pressed="completion_button" in event_steps,
                domain_success=success,
                event_steps=event_steps,
            )
            evaluator = TM01Evaluator()
            evaluator.reset(TM01EvaluationSnapshot(step_id=0))
            self._tm01_evaluator = evaluator
        elif isinstance(state, TM02State):
            self._evaluation_snapshot = TM02EvaluationSnapshot(
                step_id=max(event_steps.values(), default=1),
                faucet_on_completed=state.a1_complete,
                pot_filled_and_can_retrieved=state.b1_complete,
                fill_pot_thread_completed=state.a2_complete,
                retrieve_can_thread_completed=state.b2_complete,
                completion_button_pressed="completion_button" in event_steps,
                domain_success=success,
                event_steps=event_steps,
            )
            evaluator = TM02Evaluator()
            evaluator.reset(TM02EvaluationSnapshot(step_id=0))
            self._tm02_evaluator = evaluator

    def _check_success(self) -> bool:
        return self._success

    def evaluation_snapshot(
        self,
    ) -> TM01EvaluationSnapshot | TM02EvaluationSnapshot:
        if self._evaluation_snapshot is None:
            raise RuntimeError("fixture has no evaluation snapshot")
        return self._evaluation_snapshot


@pytest.mark.parametrize(
    ("a2_step", "b2_step"),
    ((8, 9), (9, 8)),
)
def test_tm01_report_accepts_either_thread_completion_order(
    a2_step: int,
    b2_step: int,
) -> None:
    state = TM01State(
        a1_complete=True,
        a2_complete=True,
        b1_complete=True,
        b2_complete=True,
    )
    runtime = SimpleNamespace(
        env=_Environment(
            state,
            {
                "A1": 2,
                "B1": 5,
                "A2": a2_step,
                "B2": b2_step,
                "completion_button": 10,
            },
            True,
        )
    )

    report = tm01_report(runtime)

    assert report["order"] == [
        "burner_on",
        "one_drawer_inspected",
        "both_threads_completed",
        "success",
    ]
    assert report["stages"]["burner_on"]["first_pass_step"] == 2
    assert report["stages"]["one_drawer_inspected"]["first_pass_step"] == 5
    assert report["stages"]["both_threads_completed"]["first_pass_step"] == 9
    assert report["stages"]["success"]["first_pass_step"] == 10
    validate_report_stages(report, expected_success=True)


def test_tm02_report_combines_fill_and_retrieval_before_both_threads() -> None:
    state = TM02State(
        a1_complete=True,
        a2_complete=True,
        b1_complete=True,
        b2_complete=True,
        pot_fill_ready=True,
        can_removed_from_cabinet=True,
    )
    runtime = SimpleNamespace(
        env=_Environment(
            state,
            {
                "A1": 3,
                "B1": 7,
                "B2": 11,
                "A2": 12,
                "completion_button": 13,
            },
            True,
        )
    )

    report = tm02_report(runtime)

    assert report["order"] == [
        "faucet_on",
        "pot_filled_and_can_retrieved",
        "both_threads_completed",
        "success",
    ]
    assert report["stages"]["pot_filled_and_can_retrieved"][
        "first_pass_step"
    ] == 7
    assert report["stages"]["both_threads_completed"]["first_pass_step"] == 12
    validate_report_stages(report, expected_success=True)


@pytest.mark.parametrize(
    ("report_builder", "state"),
    (
        (
            tm01_report,
            TM01State(
                a1_complete=True,
                a2_complete=True,
                b1_complete=True,
                b2_complete=True,
            ),
        ),
        (
            tm02_report,
            TM02State(
                a1_complete=True,
                a2_complete=True,
                b1_complete=True,
                b2_complete=True,
            ),
        ),
    ),
)
def test_tm_report_preserves_thread_history_when_final_success_fails(
    report_builder,
    state,
) -> None:
    runtime = SimpleNamespace(
        env=_Environment(
            state,
            {
                "A1": 1,
                "B1": 2,
                "A2": 3,
                "B2": 4,
                "completion_button": 5,
            },
            False,
        )
    )

    report = report_builder(runtime)

    assert report["stages"][report["order"][-2]]["passed"] is True
    assert report["stages"]["success"]["passed"] is False
    validate_report_stages(report, expected_success=False)


def test_tm_runtimes_register_their_public_projectors() -> None:
    assert tm01_runtime.TASK_RUNTIME_API.report_stage_profile_id == "TM01/report-v1"
    assert callable(tm01_runtime.TASK_RUNTIME_API.report_stages)
    assert tm02_runtime.TASK_RUNTIME_API.report_stage_profile_id == "TM02/report-v1"
    assert callable(tm02_runtime.TASK_RUNTIME_API.report_stages)
