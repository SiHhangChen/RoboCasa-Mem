from __future__ import annotations

import pytest

from bench.evaluation.summary import aggregate_episode_results


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
