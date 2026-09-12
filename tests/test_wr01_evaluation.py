from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import bench.tasks.wr01.runtime as wr01_runtime
from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wr01.evaluation import (
    REPORT_STAGE_ORDER,
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WR01EvaluationSnapshot,
    WR01Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    order: tuple[str, str] = ("blue_can", "red_can"),
    target: str = "blue_can",
    target_level: str = "lower",
    grasped: tuple[str, ...] = (),
    stored: dict[str, str] | None = None,
    upper_closed: bool = False,
    lower_closed: bool = False,
    opened: str | None = None,
    retrieved: str | None = None,
    distractor_stored: bool = True,
    storage_hidden: bool = False,
    final_placement_verified: bool = False,
) -> WR01EvaluationSnapshot:
    return WR01EvaluationSnapshot(
        step_id=step,
        storage_order=order,
        target_item_id=target,
        target_level=target_level,
        item_grasped={item_id: item_id in grasped for item_id in order},
        stored_levels_by_item=stored or {},
        storage_hidden=storage_hidden,
        upper_door_closed=upper_closed,
        lower_door_closed=lower_closed,
        opened_level=opened,
        retrieved_item_id=retrieved,
        distractor_still_stored=distractor_stored,
        final_placement_verified=final_placement_verified,
    )


def _evaluator() -> WR01Evaluator:
    evaluator = WR01Evaluator(
        storage_order=("blue_can", "red_can"),
        target_item_id="blue_can",
        target_level="lower",
    )
    evaluator.reset(_snapshot(0))
    return evaluator


def _close_both_doors(evaluator: WR01Evaluator) -> dict[str, str]:
    both_stored = {"blue_can": "lower", "red_can": "upper"}
    evaluator.observe(_snapshot(1, stored=both_stored, upper_closed=True))
    evaluator.observe(
        _snapshot(
            2,
            stored=both_stored,
            storage_hidden=True,
            upper_closed=True,
            lower_closed=True,
        )
    )
    return both_stored


def _episode_snapshots() -> tuple[WR01EvaluationSnapshot, ...]:
    first_closed = _snapshot(
        2, stored={"blue_can": "lower", "red_can": "upper"}, upper_closed=True,
    )
    hidden = replace(first_closed, step_id=3, lower_door_closed=True, storage_hidden=True)
    opened = replace(hidden, step_id=4, lower_door_closed=False, opened_level="lower")
    retrieved = replace(opened, step_id=5, retrieved_item_id="blue_can")
    placed = replace(retrieved, step_id=6, final_placement_verified=True)
    return (_snapshot(0), _snapshot(1), first_closed, hidden, opened, retrieved, placed)


def test_wr01_evaluator_completes_without_a_neutral_reset_stage() -> None:
    evaluator = _evaluator()
    both_stored = _close_both_doors(evaluator)
    opened = evaluator.observe(
        _snapshot(
            3,
            stored=both_stored,
            upper_closed=True,
            storage_hidden=True,
            opened="lower",
        )
    )
    assert opened.stages["open_cabinet"].qualified
    assert not opened.success
    result = evaluator.observe(
        _snapshot(
            4,
            stored=both_stored,
            upper_closed=True,
            storage_hidden=True,
            opened="lower",
            retrieved="blue_can",
        )
    )
    assert result.stages["retrieve_item"].qualified
    assert not result.success
    result = evaluator.observe(
        _snapshot(
            5,
            stored=both_stored,
            upper_closed=True,
            storage_hidden=True,
            opened="lower",
            retrieved="blue_can",
            final_placement_verified=True,
        )
    )

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert {
        name: result.stages[name].qualified_step for name in STAGE_ORDER
    } == {
        "close_first_door": 1,
        "close_second_door": 2,
        "open_cabinet": 3,
        "retrieve_item": 4,
        "place_on_table": 5,
    }
    diagnostics = result.to_diagnostics()
    assert diagnostics["stage_checks"] == {name: True for name in STAGE_ORDER}
    assert diagnostics["evaluation"]["stage_bindings"] == {
        "first_can": "blue_can",
        "first_store_level": "lower",
        "second_can": "red_can",
        "second_store_level": "upper",
        "retrieval_target": "blue_can",
        "retrieval_level": "lower",
    }

    report = result.to_report_stages()
    assert report == {
        "schema_version": "membench.report_stages/v1",
        "task_id": "WR-01",
        "profile_id": "WR-01/report-v2",
        "order": [
            "one_can_stored",
            "two_cans_stored",
            "correct_cabinet_opened",
            "success",
        ],
        "stages": {
            "one_can_stored": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 1,
            },
            "two_cans_stored": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 2,
            },
            "correct_cabinet_opened": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 3,
            },
            "success": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 5,
            },
        },
    }
    assert not set(STAGE_ORDER) & set(report["stages"])
    assert validate_report_stages(
        report,
        expected_task_id="WR-01",
        expected_profile_id="WR-01/report-v2",
        expected_success=True,
    ) == report


@pytest.mark.parametrize("first_level", ["lower", "upper"])
def test_wr01_evaluator_accepts_either_door_closed_first(first_level: str) -> None:
    evaluator = _evaluator()
    both_stored = {"blue_can": "lower", "red_can": "upper"}

    first = evaluator.observe(
        _snapshot(
            1, stored=both_stored,
            lower_closed=first_level == "lower",
            upper_closed=first_level == "upper",
        )
    )
    assert first.stages["close_first_door"].qualified
    assert not first.stages["close_second_door"].qualified
    assert first.failure_code is None

    second = evaluator.observe(
        _snapshot(2, stored=both_stored, upper_closed=True, lower_closed=True)
    )
    assert second.stages["close_first_door"].qualified_step == 1
    assert second.stages["close_second_door"].qualified_step == 2
    assert second.failure_code is None


def test_wr01_evaluator_accepts_both_doors_closed_on_the_same_step() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(
            1, stored={"blue_can": "lower", "red_can": "upper"},
            upper_closed=True, lower_closed=True,
        )
    )
    assert result.stages["close_first_door"].qualified_step == 1
    assert result.stages["close_second_door"].qualified_step == 1


def test_wr01_late_storage_confirmation_does_not_require_reclosing_the_doors() -> None:
    evaluator = _evaluator()
    pending = evaluator.observe(_snapshot(1, upper_closed=True, lower_closed=True))
    assert not pending.stages["close_first_door"].qualified
    assert not pending.stages["close_second_door"].qualified

    result = evaluator.observe(
        _snapshot(
            2, stored={"blue_can": "lower", "red_can": "upper"},
            upper_closed=True, lower_closed=True,
        )
    )
    assert result.stages["close_first_door"].qualified_step == 2
    assert result.stages["close_second_door"].qualified_step == 2


@pytest.mark.parametrize("first_level", ["lower", "upper"])
def test_wr01_closing_doors_at_separate_times_does_not_count_both_closed(
    first_level: str,
) -> None:
    evaluator = _evaluator()
    both_stored = {"blue_can": "lower", "red_can": "upper"}
    first = _snapshot(
        1, stored=both_stored,
        lower_closed=first_level == "lower", upper_closed=first_level == "upper",
    )
    evaluator.observe(first)
    evaluator.observe(_snapshot(2, stored=both_stored))
    repeated = evaluator.observe(replace(first, step_id=3))
    assert not repeated.stages["close_second_door"].qualified

    other_only = evaluator.observe(
        _snapshot(
            4, stored=both_stored,
            lower_closed=first_level == "upper", upper_closed=first_level == "lower",
        )
    )
    assert not other_only.stages["close_second_door"].qualified

    both = evaluator.observe(
        _snapshot(5, stored=both_stored, lower_closed=True, upper_closed=True)
    )
    assert both.stages["close_second_door"].qualified_step == 5


def test_wr01_non_target_cabinet_can_be_opened_before_the_target() -> None:
    evaluator = _evaluator()
    both_stored = _close_both_doors(evaluator)

    result = evaluator.observe(
        _snapshot(
            3,
            stored=both_stored,
            storage_hidden=True,
            lower_closed=True,
            opened="upper",
        )
    )

    assert not result.terminal
    assert not result.success
    assert result.failure_code is None
    assert not result.stages["open_cabinet"].qualified

    report = result.to_report_stages()
    assert [
        report["stages"][name]["passed"] for name in REPORT_STAGE_ORDER
    ] == [True, True, False, False]
    assert report["stages"]["correct_cabinet_opened"]["first_pass_step"] is None
    assert report["stages"]["success"]["first_pass_step"] is None

    # No intermediate "no door open" observation is required before recovery.
    target_open = _snapshot(4, stored=both_stored, storage_hidden=True, opened="lower")
    opened = evaluator.observe(target_open)
    assert opened.stages["open_cabinet"].qualified_step == 4
    evaluator.observe(replace(target_open, step_id=5, retrieved_item_id="blue_can"))
    completed = evaluator.observe(
        replace(target_open, step_id=6, retrieved_item_id="blue_can", final_placement_verified=True)
    )
    assert completed.success
    assert completed.failure_code is None


def test_wr01_opening_still_requires_a_physical_memory_barrier() -> None:
    evaluator = _evaluator()
    both_stored = {"blue_can": "lower", "red_can": "upper"}
    evaluator.observe(_snapshot(1, stored=both_stored, upper_closed=True))
    evaluator.observe(
        _snapshot(2, stored=both_stored, upper_closed=True, lower_closed=True)
    )
    result = evaluator.observe(
        _snapshot(3, stored=both_stored, upper_closed=True, opened="lower")
    )

    assert not result.terminal
    assert not result.success
    assert result.failure_code is None
    assert not result.stages["open_cabinet"].qualified

    evaluator.observe(
        _snapshot(4, stored=both_stored, storage_hidden=True, upper_closed=True, lower_closed=True)
    )
    opened = evaluator.observe(
        _snapshot(5, stored=both_stored, storage_hidden=True, opened="lower")
    )
    assert opened.stages["open_cabinet"].qualified_step == 5


@pytest.mark.parametrize(
    ("retrieved", "distractor_stored", "failure_code"),
    [("red_can", True, "wrong_item_retrieved"), ("blue_can", False, "distractor_not_stored")],
)
def test_wr01_unordered_doors_still_require_the_correct_item_and_stored_distractor(
    retrieved: str, distractor_stored: bool, failure_code: str,
) -> None:
    evaluator = _evaluator()
    both_stored = _close_both_doors(evaluator)
    target_open = _snapshot(3, stored=both_stored, storage_hidden=True, opened="lower")
    evaluator.observe(target_open)
    result = evaluator.observe(
        replace(
            target_open, step_id=4, retrieved_item_id=retrieved,
            distractor_still_stored=distractor_stored,
        )
    )
    assert result.terminal
    assert not result.success
    assert result.failure_code == failure_code


@pytest.mark.parametrize(
    "violation_step", (1, 2, 3, 4, 5, 6),
    ids=("before_storage_confirmation", "first_close", "second_close", "open", "retrieve", "place"),
)
def test_wr01_distractor_departure_is_terminal_at_every_stage(violation_step: int) -> None:
    evaluator = _evaluator()
    snapshots = _episode_snapshots()
    for snapshot in snapshots[1:violation_step]:
        assert not evaluator.observe(snapshot).terminal
    previous = evaluator.result()

    failed = evaluator.observe(
        replace(snapshots[violation_step], distractor_still_stored=False)
    )

    assert failed.terminal
    assert not failed.success
    assert failed.failure_code == "distractor_not_stored"
    # A global violation takes precedence over newly completed stages, including
    # placing the target on the table on the very same control step.
    assert failed.stages == previous.stages
    assert failed.to_diagnostics()["evaluation"]["step_id"] == violation_step

    restored = evaluator.observe(replace(snapshots[-1], step_id=violation_step + 10))
    assert restored is failed
    assert not restored.to_report_stages()["stages"]["success"]["passed"]


def test_wr01_distractor_departure_is_checked_without_a_new_stage_event() -> None:
    evaluator = _evaluator()
    snapshots = _episode_snapshots()
    for snapshot in snapshots[1:-1]:
        evaluator.observe(snapshot)
    assert evaluator.result().stages["retrieve_item"].qualified

    departure = replace(snapshots[-2], step_id=6, distractor_still_stored=False)
    assert departure.raw_stage_values() == snapshots[-2].raw_stage_values()
    failed = evaluator.observe(departure)

    assert failed.terminal
    assert failed.failure_code == "distractor_not_stored"
    assert not failed.success
    assert failed.stages["retrieve_item"].qualified
    assert not failed.stages["place_on_table"].qualified
    assert evaluator.observe(replace(snapshots[-1], step_id=7)) is failed


def test_wr01_distractor_must_be_in_its_cabinet_at_initialization() -> None:
    evaluator = _evaluator()
    failed = evaluator.reset(_snapshot(0, distractor_stored=False))

    assert failed.terminal
    assert not failed.success
    assert failed.failure_code == "distractor_not_stored"
    assert failed.to_diagnostics()["evaluation"]["step_id"] == 0
    assert evaluator.observe(_snapshot(1)) is failed


def test_wr01_new_episode_clears_the_distractor_violation() -> None:
    evaluator = _evaluator()
    failed = evaluator.observe(_snapshot(1, distractor_stored=False))
    assert failed.terminal

    snapshots = _episode_snapshots()
    reset = evaluator.reset(snapshots[0])
    assert not reset.terminal
    assert reset.failure_code is None
    for snapshot in snapshots[1:]:
        result = evaluator.observe(snapshot)
    assert result.success
    assert result.failure_code is None


def test_wr01_distractor_guard_ends_when_the_episode_succeeds() -> None:
    evaluator = _evaluator()
    snapshots = _episode_snapshots()
    for snapshot in snapshots[1:]:
        result = evaluator.observe(snapshot)
    assert result.success

    # Completion ends this episode; observations after terminal success cannot
    # reopen it. A new episode must explicitly reset the evaluator.
    assert evaluator.observe(
        replace(snapshots[-1], step_id=7, distractor_still_stored=False)
    ) is result


def test_wr01_runtime_callbacks_share_one_evaluator_result() -> None:
    evaluator = _evaluator()
    both_stored = _close_both_doors(evaluator)

    class FakeEnv:
        snapshot = _snapshot(
            2,
            stored=both_stored,
            storage_hidden=True,
            upper_closed=True,
            lower_closed=True,
        )

        def evaluation_snapshot(self) -> WR01EvaluationSnapshot:
            return self.snapshot

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["close_second_door"]
    assert "canonical_reset" not in diagnostics["stage_checks"]
    assert not success_for_runtime(runtime, {})
    assert diagnostics_for_runtime(runtime) == diagnostics
    report = report_stages_for_runtime(runtime)
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert report["stages"]["success"]["passed"] is False


def test_wr01_runtime_registers_public_report_stage_projector() -> None:
    assert wr01_runtime.TASK_RUNTIME_API.report_stage_profile_id == (
        "WR-01/report-v2"
    )
    assert callable(wr01_runtime.TASK_RUNTIME_API.report_stages)
