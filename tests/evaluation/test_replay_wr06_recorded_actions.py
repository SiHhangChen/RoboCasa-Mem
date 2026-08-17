"""Tests for WR-06 recorded-episode discovery and replay reporting."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.eval.replay_wr06_recorded_actions import (
    _discover_episodes,
    _failure_reason,
    _parse_seeds,
)


def _episode(root: Path, branch: str, seed: int, attempt: int = 0) -> Path:
    path = (
        root
        / branch
        / "WR-06"
        / f"seed_{seed:06d}_attempt_{attempt:02d}"
    )
    path.mkdir(parents=True)
    (path / "trajectory.h5").touch()
    return path


def test_parse_seeds_accepts_ranges_and_descending_ranges() -> None:
    assert _parse_seeds("0-2,5;7-6") == {0, 1, 2, 5, 6, 7}


def test_discovery_includes_failed_tree_when_manifest_is_partial(tmp_path) -> None:
    accepted = _episode(tmp_path, "train", seed=0)
    failed = _episode(tmp_path, "debug_failed", seed=1)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "task_id": "WR-06",
                        "seed": 0,
                        "attempt": 0,
                        "success": True,
                        "episode_identity": "recorded-zero",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    episodes = _discover_episodes(tmp_path, set(), source_status="all")

    assert [(item["seed"], item["source_status"]) for item in episodes] == [
        (0, "success"),
        (1, "failed"),
    ]
    assert episodes[0]["episode_dir"] == accepted
    assert episodes[0]["episode_identity"] == "recorded-zero"
    assert episodes[1]["episode_dir"] == failed


def test_discovery_status_filter_selects_failed_copy(tmp_path) -> None:
    _episode(tmp_path, "train", seed=3)
    failed = _episode(tmp_path, "debug_failed", seed=3)

    episodes = _discover_episodes(tmp_path, {3}, source_status="failed")

    assert len(episodes) == 1
    assert episodes[0]["episode_dir"] == failed
    assert episodes[0]["source_success"] is False


def test_failure_reason_distinguishes_regression_and_prefix() -> None:
    assert _failure_reason(success=True, success_step=10, consumed=20, available=20) is None
    assert (
        _failure_reason(success=False, success_step=10, consumed=20, available=20)
        == "terminal_state_not_preserved"
    )
    assert (
        _failure_reason(success=False, success_step=None, consumed=5, available=20)
        == "replay_stopped_early"
    )
    assert (
        _failure_reason(success=False, success_step=None, consumed=20, available=20)
        == "recorded_actions_exhausted"
    )


def test_failure_reason_reports_incomplete_physical_phase() -> None:
    assert (
        _failure_reason(
            success=False,
            success_step=None,
            consumed=20,
            available=20,
            diagnostics={"physical_events": {"phase": "verify_return"}},
        )
        == "terminal_verification_incomplete"
    )
    assert (
        _failure_reason(
            success=False,
            success_step=None,
            consumed=20,
            available=20,
            diagnostics={
                "failure_code": "plate_reentered_after_departure",
                "physical_events": {
                    "phase": "invalid",
                    "violation_reason": "plate_reentered_after_departure",
                },
            },
        )
        == "plate_reentered_after_departure"
    )
