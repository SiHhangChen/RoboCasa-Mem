from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.eval import summarize_wr04_stages as summary


def _make_run(root: Path, name: str, manifest: dict) -> Path:
    run = root / "gr00t" / "WR04" / "90000" / name
    (run / "episodes").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_formal_summary_excludes_legacy_runs(tmp_path, monkeypatch) -> None:
    official = _make_run(
        tmp_path,
        "arbitrary-replay-name",
        {"replay": {"evaluator_profile": summary.OFFICIAL_EVALUATOR_PROFILE}},
    )
    _make_run(tmp_path, "old-replay-env-v2", {})
    monkeypatch.setattr(summary, "EVAL_ROOT", tmp_path)

    assert summary.collect_runs() == [official]
    assert len(summary.collect_runs(include_legacy=True)) == 2


def test_final_success_requires_consistent_terminal_result() -> None:
    stale_success = {
        "status": "failure",
        "success": False,
        "failure_reason": "sink_reentered_after_departure",
        "diagnostics": {
            "success": True,
            "item_removed_from_home": True,
            "item_staged_at_fixed_sink": True,
            "memory_barrier_latched": True,
            "first_return_level_correct": True,
            "same_item_returned": True,
        },
    }

    hits = summary.stage_hits(stale_success)

    assert hits["barrier"] is True
    assert hits["success"] is False


def test_episode_row_exposes_terminal_violation(tmp_path) -> None:
    run = tmp_path / "run-replay-full-horizon-v5"
    result = {
        "seed": 1060,
        "status": "failure",
        "success": False,
        "failure_reason": "sink_reentered_after_departure",
        "env_steps": 1200,
        "diagnostics": {
            "success": False,
            "failure_code": "sink_reentered_after_departure",
            "physical_events": {
                "phase": "invalid",
                "phase_history": [
                    {"phase": "home", "step": 0},
                    {"phase": "to_return", "step": 815},
                    {"phase": "invalid", "step": 1200},
                ],
            },
        },
    }

    row = summary.episode_row(run, result)

    assert row["success"] is False
    assert row["failure_reason"] == "sink_reentered_after_departure"
    assert row["final_phase"] == "invalid"
    assert row["phase_history"].endswith("invalid@1200")


def test_no_v5_runs_clears_stale_formal_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(summary, "EVAL_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["summarize_wr04_stages.py"])
    stage_report = tmp_path / "gr00t_wr04_stage_summary.csv"
    episode_report = tmp_path / "gr00t_wr04_episode_results.csv"
    stage_report.write_text("stale-v4-success\n", encoding="utf-8")
    episode_report.write_text("stale-v4-success\n", encoding="utf-8")

    assert summary.main() == 1

    assert "stale-v4-success" not in stage_report.read_text(encoding="utf-8-sig")
    assert "stale-v4-success" not in episode_report.read_text(encoding="utf-8-sig")
