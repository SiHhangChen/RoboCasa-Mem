"""Task-level evaluation tests for WR-05."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import bench.tasks.wr05.environment as wr05_environment
import bench.tasks.wr05.runtime as wr05_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr05.domain.semantics import SemanticState
from bench.tasks.wr05.evaluation import (
    REPORT_STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WR05EvaluationSnapshot,
    WR05Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


CONDIMENTS = ("ketchup", "mustard")


def _flags(*names: str) -> dict[str, bool]:
    return {name: name in names for name in CONDIMENTS}


def _snapshot(
    step: int,
    *,
    staged: tuple[str, ...] = (),
    home: tuple[str, ...] = (),
    released: tuple[str, ...] = (),
    relation_correct: bool = False,
    success: bool = False,
) -> WR05EvaluationSnapshot:
    return WR05EvaluationSnapshot(
        step_id=step,
        staged_at_plate=_flags(*staged),
        home_correct_by_condiment=_flags(*home),
        released_by_condiment=_flags(*released),
        shelf_level_relation_correct=relation_correct,
        domain_success=success,
        details={"legacy_check": True},
    )


def _evaluator() -> WR05Evaluator:
    evaluator = WR05Evaluator()
    evaluator.reset(
        _snapshot(
            0,
            home=CONDIMENTS,
            released=CONDIMENTS,
            relation_correct=True,
        )
    )
    return evaluator


def test_wr05_evaluator_qualifies_the_public_funnel() -> None:
    evaluator = _evaluator()
    evaluator.observe(
        _snapshot(
            1,
            staged=("ketchup",),
            home=("mustard",),
            released=CONDIMENTS,
        )
    )
    evaluator.observe(
        _snapshot(
            2,
            staged=CONDIMENTS,
            released=CONDIMENTS,
        )
    )
    evaluator.observe(
        _snapshot(
            3,
            staged=CONDIMENTS,
            home=("ketchup",),
            released=CONDIMENTS,
        )
    )
    result = evaluator.observe(
        _snapshot(
            4,
            staged=CONDIMENTS,
            home=CONDIMENTS,
            released=CONDIMENTS,
            relation_correct=True,
            success=True,
        )
    )

    assert result.success
    assert result.terminal
    assert {
        name: result.stages[name].qualified_step for name in REPORT_STAGE_ORDER
    } == {
        "one_item_staged": 1,
        "two_items_staged": 2,
        "one_item_correctly_returned": 3,
        "success": 4,
    }
    diagnostics = result.to_diagnostics()
    assert diagnostics["legacy_check"]
    assert diagnostics["stage_checks"] == {
        name: True for name in REPORT_STAGE_ORDER
    }

    report = result.to_report_stages()
    assert report == {
        "schema_version": "membench.report_stages/v1",
        "task_id": "WR-05",
        "profile_id": "WR-05/report-v1",
        "order": [
            "one_item_staged",
            "two_items_staged",
            "one_item_correctly_returned",
            "success",
        ],
        "stages": {
            "one_item_staged": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 1,
            },
            "two_items_staged": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 2,
            },
            "one_item_correctly_returned": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 3,
            },
            "success": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 4,
            },
        },
    }
    assert validate_report_stages(
        report,
        expected_task_id="WR-05",
        expected_profile_id="WR-05/report-v1",
        expected_success=True,
    ) == report


def test_initial_home_positions_do_not_count_as_returns() -> None:
    evaluator = _evaluator()
    initial = evaluator.result()

    assert not initial.stages["one_item_correctly_returned"].qualified
    assert not initial.stages["one_item_correctly_returned"].observed
    assert initial.stages["one_item_correctly_returned"].reason == (
        "waiting_for_prerequisite:two_items_staged"
    )


def test_return_before_two_staged_does_not_qualify_s3() -> None:
    evaluator = _evaluator()
    evaluator.observe(
        _snapshot(
            1,
            staged=("ketchup",),
            home=("mustard",),
            released=CONDIMENTS,
        )
    )
    early_return = evaluator.observe(
        _snapshot(
            2,
            staged=("ketchup",),
            home=CONDIMENTS,
            released=CONDIMENTS,
        )
    )
    both_staged = evaluator.observe(
        _snapshot(
            3,
            staged=CONDIMENTS,
            home=("ketchup",),
            released=CONDIMENTS,
        )
    )
    completed = evaluator.observe(
        _snapshot(
            4,
            staged=CONDIMENTS,
            home=CONDIMENTS,
            released=CONDIMENTS,
            relation_correct=True,
            success=True,
        )
    )

    assert not early_return.stages["one_item_correctly_returned"].qualified
    assert early_return.stages["one_item_correctly_returned"].reason == (
        "return_before_two_items_staged:ketchup"
    )
    assert not both_staged.stages["one_item_correctly_returned"].qualified
    assert completed.stages["one_item_correctly_returned"].qualified_step == 4
    assert completed.stages["success"].qualified_step == 4


def test_return_edge_in_same_step_as_second_staging_is_rejected() -> None:
    evaluator = _evaluator()
    evaluator.observe(
        _snapshot(
            1,
            staged=("ketchup",),
            released=CONDIMENTS,
        )
    )
    same_step = evaluator.observe(
        _snapshot(
            2,
            staged=CONDIMENTS,
            home=("ketchup",),
            released=CONDIMENTS,
        )
    )
    unchanged = evaluator.observe(
        _snapshot(
            3,
            staged=CONDIMENTS,
            home=("ketchup",),
            released=CONDIMENTS,
        )
    )

    assert same_step.stages["two_items_staged"].qualified_step == 2
    assert not same_step.stages["one_item_correctly_returned"].qualified
    assert same_step.stages["one_item_correctly_returned"].reason == (
        "return_before_two_items_staged:ketchup"
    )
    assert not unchanged.stages["one_item_correctly_returned"].qualified


def test_held_item_in_home_region_is_not_a_return() -> None:
    evaluator = _evaluator()
    evaluator.observe(
        _snapshot(1, staged=("ketchup",), released=("mustard",))
    )
    evaluator.observe(_snapshot(2, staged=CONDIMENTS))
    held_at_home = evaluator.observe(
        _snapshot(
            3,
            staged=CONDIMENTS,
            home=("ketchup",),
            released=("mustard",),
        )
    )
    released_at_home = evaluator.observe(
        _snapshot(
            4,
            staged=CONDIMENTS,
            home=("ketchup",),
            released=CONDIMENTS,
        )
    )

    assert not held_at_home.stages["one_item_correctly_returned"].qualified
    assert released_at_home.stages["one_item_correctly_returned"].qualified_step == 4


def test_success_is_revoked_when_final_state_regresses() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, staged=("ketchup",)))
    evaluator.observe(_snapshot(2, staged=CONDIMENTS))
    success = evaluator.observe(
        _snapshot(
            3,
            staged=CONDIMENTS,
            home=CONDIMENTS,
            released=CONDIMENTS,
            relation_correct=True,
            success=True,
        )
    )
    regressed = evaluator.observe(
        _snapshot(
            4,
            staged=CONDIMENTS,
            home=("ketchup",),
            released=CONDIMENTS,
        )
    )

    assert success.success
    assert not regressed.success
    assert regressed.stages["one_item_correctly_returned"].qualified
    assert not regressed.stages["success"].qualified
    assert regressed.stages["success"].reason == "final_state_regressed"
    assert not regressed.to_report_stages()["stages"]["success"]["passed"]


def test_snapshot_rejects_inconsistent_domain_success() -> None:
    with pytest.raises(ValueError, match="domain_success requires"):
        _snapshot(1, success=True)


def test_environment_latches_plate_staging_only_after_release(monkeypatch) -> None:
    env = object.__new__(wr05_environment.WR05ReturnCondimentPair)
    env.condiment_names = CONDIMENTS
    env._semantic_state = SemanticState()
    env._condiment_at_plate_side = lambda condiment_id: True
    env._condiment_in_return_shelf = lambda condiment_id: False
    env._shelf_level_relation_correct = lambda: False
    env._condiment_relation_diagnostics = lambda: {"valid": False}
    env._obj_pos = lambda condiment_id: np.zeros(3, dtype=float)
    monkeypatch.setattr(
        wr05_environment.OU,
        "gripper_obj_far",
        lambda task, condiment_id, th: condiment_id == "mustard",
    )

    env._update_housemem_state()

    assert env._semantic_state.staged_at_plate == {
        "ketchup": False,
        "mustard": True,
    }


def test_runtime_callbacks_share_the_initialized_evaluator() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        snapshot = _snapshot(1, staged=("ketchup",))

        def evaluation_snapshot(self) -> WR05EvaluationSnapshot:
            return self.snapshot

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["one_item_staged"]
    assert not success_for_runtime(runtime, {})
    assert report_stages_for_runtime(runtime)["order"] == list(
        REPORT_STAGE_ORDER
    )


def test_runtime_initialize_mounts_a_fresh_evaluator(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeEvaluator:
        def reset(self, snapshot: object) -> None:
            calls["snapshot"] = snapshot

    monkeypatch.setattr(wr05_runtime, "WR05Evaluator", FakeEvaluator)
    monkeypatch.setattr(
        wr05_runtime,
        "_initialize_shelf",
        lambda descriptor, env, observation: (
            "updated",
            "initialization",
            {"existing": True},
            "targets",
        ),
    )

    class FakeEnv:
        _wr05_evaluator = None

        def evaluation_snapshot(self) -> str:
            return "SNAPSHOT"

    env = FakeEnv()
    updated, initialization, metadata, targets = wr05_runtime._initialize(
        object(),
        env,
        {},
    )

    assert updated == "updated"
    assert initialization == "initialization"
    assert targets == "targets"
    assert metadata["existing"] is True
    assert metadata[TASK_EVALUATOR_METADATA_KEY] is env._wr05_evaluator
    assert calls["snapshot"] == "SNAPSHOT"


def test_runtime_registers_public_report_stage_projector() -> None:
    assert wr05_runtime.TASK_RUNTIME_API.report_stage_profile_id == (
        "WR-05/report-v1"
    )
    assert callable(wr05_runtime.TASK_RUNTIME_API.report_stages)
