from __future__ import annotations

import pytest

from bench.evaluation.summary import aggregate_episode_results


def _ts03_report(
    *,
    target: int,
    passed_sugars: int,
    success: bool,
) -> dict[str, object]:
    stages: dict[str, dict[str, object]] = {}
    for index in range(1, 4):
        applicable = index <= target
        passed = index <= passed_sugars if applicable else None
        stages[f"sugar_{index}"] = {
            "applicable": applicable,
            "passed": passed,
            "first_pass_step": index if passed is True else None,
        }
    stages["success"] = {
        "applicable": True,
        "passed": success,
        "first_pass_step": target + 1 if success else None,
    }
    return {
        "schema_version": "membench.report_stages/v1",
        "task_id": "TS03",
        "profile_id": "TS03/report-v2",
        "order": ["sugar_1", "sugar_2", "sugar_3", "success"],
        "stages": stages,
    }


def test_legacy_aggregation_is_preserved_without_expected_seeds():
    summary = aggregate_episode_results(
        [
            {"status": "success", "success_step": 3, "inference_latency_ms": [1.0]},
            {"status": "failure", "failure_reason": "timeout"},
        ]
    )

    assert summary["episodes_total"] == 2
    assert summary["counts"]["success"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["task_success_rate"] == 0.5
    assert summary["overall_success_rate"] == 0.5
    assert summary["planned"] == 2
    assert summary["recorded"] == 2
    assert summary["pending"] == 0
    assert summary["unexpected"] == 0
    assert summary["completion_rate"] == 1.0
    assert summary["publishable"] is False


def test_expected_seeds_define_completion_and_formal_rates():
    summary = aggregate_episode_results(
        [
            {"seed": 1, "status": "success", "success_step": 4},
            {"seed": 1, "status": "failure", "failure_reason": "duplicate"},
            {"seed": 2, "status": "failure", "failure_reason": "timeout"},
            {"seed": 99, "status": "success"},
            {"seed": 99, "status": "failure"},
        ],
        expected_seeds=[1, 2, 3, 3],
    )

    assert summary["planned"] == 3
    assert summary["recorded"] == 2
    assert summary["pending"] == 1
    assert summary["unexpected"] == 1
    assert summary["completion_rate"] == pytest.approx(2 / 3)
    assert summary["episodes_total"] == 2
    assert summary["counts"] == {
        "success": 1,
        "failure": 1,
        "error": 0,
        "cancelled": 0,
    }
    assert summary["task_success_rate"] == 0.5
    assert summary["overall_success_rate"] == pytest.approx(1 / 3)
    assert summary["failure_reasons"] == {"timeout": 1}
    assert summary["duplicate_seeds"] == [1, 99]
    assert summary["publishable"] is False


def test_complete_clean_expected_suite_is_publishable():
    summary = aggregate_episode_results(
        [
            {"seed": 10, "status": "success"},
            {"seed": 11, "status": "failure", "failure_reason": "max_steps"},
        ],
        expected_seeds=[10, 11],
    )

    assert summary["completion_rate"] == 1.0
    assert summary["task_success_rate"] == 0.5
    assert summary["overall_success_rate"] == 0.5
    assert summary["duplicate_seeds"] == []
    assert summary["publishable"] is True


@pytest.mark.parametrize("status", ["error", "cancelled", "unknown"])
def test_non_task_terminal_status_is_not_publishable(status: str):
    summary = aggregate_episode_results(
        [{"seed": 10, "status": status}],
        expected_seeds=[10],
    )

    assert summary["completion_rate"] == 1.0
    assert summary["task_success_rate"] is None
    assert summary["overall_success_rate"] == 0.0
    assert summary["publishable"] is False


def test_expected_seed_values_must_be_integers():
    with pytest.raises(ValueError, match="integer"):
        aggregate_episode_results([], expected_seeds=["1"])


def test_report_stage_denominators_exclude_non_applicable_stages() -> None:
    summary = aggregate_episode_results(
        [
            {
                "seed": 1,
                "task_id": "TS03",
                "status": "success",
                "success": True,
                "report_stages": _ts03_report(
                    target=1,
                    passed_sugars=1,
                    success=True,
                ),
            },
            {
                "seed": 2,
                "task_id": "TS03",
                "status": "failure",
                "success": False,
                "failure_reason": "max_steps",
                "report_stages": _ts03_report(
                    target=2,
                    passed_sugars=1,
                    success=False,
                ),
            },
            {
                "seed": 3,
                "task_id": "TS03",
                "status": "success",
                "success": True,
                "report_stages": _ts03_report(
                    target=3,
                    passed_sugars=3,
                    success=True,
                ),
            },
        ],
        expected_seeds=[1, 2, 3],
    )

    report = summary["report_stages"]
    assert report["complete"] is True
    assert report["eligible_episodes"] == report["reported_episodes"] == 3
    assert report["stages"]["sugar_1"] == {
        "passed": 3,
        "denominator": 3,
        "rate": 1.0,
        "not_applicable": 0,
        "first_pass_step_mean": 1.0,
    }
    assert report["stages"]["sugar_2"]["passed"] == 1
    assert report["stages"]["sugar_2"]["denominator"] == 2
    assert report["stages"]["sugar_2"]["not_applicable"] == 1
    assert report["stages"]["sugar_3"]["passed"] == 1
    assert report["stages"]["sugar_3"]["denominator"] == 1
    assert report["stages"]["sugar_3"]["not_applicable"] == 2
    assert report["stages"]["success"]["passed"] == 2
    assert report["stages"]["success"]["denominator"] == 3
    assert summary["publishable"] is True


def test_partial_report_stage_coverage_is_not_publishable() -> None:
    summary = aggregate_episode_results(
        [
            {
                "seed": 1,
                "task_id": "TS03",
                "status": "success",
                "success": True,
                "report_stages": _ts03_report(
                    target=1,
                    passed_sugars=1,
                    success=True,
                ),
            },
            {
                "seed": 2,
                "task_id": "TS03",
                "status": "failure",
                "success": False,
                "failure_reason": "max_steps",
            },
        ],
        expected_seeds=[1, 2],
    )

    assert summary["report_stages"]["reported_episodes"] == 1
    assert summary["report_stages"]["missing_episodes"] == 1
    assert summary["report_stages"]["complete"] is False
    assert summary["publishable"] is False


def test_incompatible_report_stage_profiles_cannot_be_aggregated() -> None:
    first = _ts03_report(target=1, passed_sugars=1, success=True)
    second = {
        "schema_version": "membench.report_stages/v1",
        "task_id": "WA-05",
        "profile_id": "WA-05/report-v1",
        "order": ["cup_filled", "cup_staged", "success"],
        "stages": {
            stage: {"applicable": True, "passed": True, "first_pass_step": index}
            for index, stage in enumerate(("cup_filled", "cup_staged", "success"), start=1)
        },
    }

    with pytest.raises(ValueError, match="incompatible report-stage profiles"):
        aggregate_episode_results(
            [
                {
                    "seed": 1,
                    "task_id": "TS03",
                    "status": "success",
                    "success": True,
                    "report_stages": first,
                },
                {
                    "seed": 2,
                    "task_id": "WA-05",
                    "status": "success",
                    "success": True,
                    "report_stages": second,
                },
            ],
            expected_seeds=[1, 2],
        )
