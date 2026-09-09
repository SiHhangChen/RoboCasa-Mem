from __future__ import annotations

from bench.evaluation.stage_integration import get_stage_evaluation_integration
from bench.tasks.tm02.evaluation import TM02EvaluationSnapshot, TM02Evaluator
from bench.tasks.tm02.stage_evaluation import (
    B1_DIAGNOSTIC_STAGE_ORDER,
    build_b1_diagnostic_stage_specs,
    build_stage_specs,
    stage_completed,
)


def _snapshot(
    step: int,
    *,
    a1: bool = False,
    b1: bool = False,
    a2: bool = False,
    b2: bool = False,
    button: bool = False,
    success: bool = False,
    failure: str | None = None,
) -> TM02EvaluationSnapshot:
    return TM02EvaluationSnapshot(
        step_id=step,
        faucet_on_completed=a1,
        pot_filled_and_can_retrieved=b1,
        fill_pot_thread_completed=a2,
        retrieve_can_thread_completed=b2,
        completion_button_pressed=button,
        absorbing_failure=failure is not None,
        failure_code=failure,
        domain_success=success,
    )


def test_tm02_stage_specs_reuse_authoritative_evaluator() -> None:
    evaluator = TM02Evaluator()
    evaluator.reset(_snapshot(0))
    faucet = evaluator.observe(_snapshot(1, a1=True))
    combined = evaluator.observe(_snapshot(2, a1=True, b1=True))
    threads = evaluator.observe(_snapshot(3, a1=True, b1=True, a2=True, b2=True))
    success = evaluator.observe(
        _snapshot(4, a1=True, b1=True, a2=True, b2=True, button=True, success=True)
    )

    assert tuple(spec.id for spec in build_stage_specs()) == (
        "faucet_on",
        "pot_filled_and_can_retrieved",
        "both_threads_completed",
        "success",
    )
    assert stage_completed(faucet, "faucet_on")
    assert stage_completed(combined, "pot_filled_and_can_retrieved")
    assert stage_completed(threads, "both_threads_completed")
    assert stage_completed(success, "success")


def test_tm02_stage_integration_uses_previous_stage_handoffs() -> None:
    integration = get_stage_evaluation_integration("TM02")

    assert integration.prefix_stage_by_target == {
        "faucet_on": None,
        "pot_filled_and_can_retrieved": "faucet_on",
        "both_threads_completed": "pot_filled_and_can_retrieved",
        "success": "both_threads_completed",
    }


def test_tm02_absorbing_failure_fails_all_stage_specs() -> None:
    evaluator = TM02Evaluator()
    evaluator.reset(_snapshot(0))
    failed = evaluator.observe(_snapshot(1, failure="early_A2"))

    assert all(spec.failure_predicate(failed) for spec in build_stage_specs())


def test_tm02_b1_diagnostic_profile_uses_branch_specific_handoffs() -> None:
    integration = get_stage_evaluation_integration(
        "TM02",
        stage_ids=B1_DIAGNOSTIC_STAGE_ORDER,
    )

    assert integration.prefix_stage_by_target == {
        "faucet_on": None,
        "pot_filled": "faucet_on",
        "cabinet_opened": "faucet_on",
        "can_removed_from_cabinet": "cabinet_opened",
        "can_released_in_place_region": "can_removed_from_cabinet",
    }
    assert tuple(
        spec.id
        for spec in build_b1_diagnostic_stage_specs(
            target_stage="cabinet_opened"
        )
    ) == ("faucet_on", "cabinet_opened")


def test_tm02_b1_diagnostic_predicates_use_privileged_details() -> None:
    state = TM02Evaluator().reset(_snapshot(0))
    details = dict(state.details)
    details.update(
        {
            "semantic": {
                "pot_fill_ready": True,
                "cabinet_opened": True,
                "can_removed_from_cabinet": True,
            },
            "physical": {
                "pot_filled": True,
                "can_in_place_region": True,
                "can_released": True,
            },
        }
    )
    diagnostic_state = type(state)(
        success=state.success,
        terminal=state.terminal,
        failure_code=state.failure_code,
        stages=state.stages,
        details=details,
    )

    for stage_id in B1_DIAGNOSTIC_STAGE_ORDER[1:]:
        assert stage_completed(diagnostic_state, stage_id)
