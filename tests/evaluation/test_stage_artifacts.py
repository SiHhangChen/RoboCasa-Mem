from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import pytest

from bench.evaluation.stage_artifacts import StageArtifactWriter
from bench.evaluation.stage_rollout import (
    FAILURE,
    INVALID_PREFIX,
    SUCCESS,
    StageEvaluationResult,
)


def _result(stage: str, seed: int, status: str, steps: int = 5) -> StageEvaluationResult:
    success = status == SUCCESS
    return StageEvaluationResult(
        task_id="WA-05",
        target_stage=stage,
        seed=seed,
        status=status,
        success=success,
        success_step=steps if success else None,
        expert_prefix_steps=3 if stage != "cup_filled" else 0,
        model_control_steps=steps if status != INVALID_PREFIX else 0,
        total_model_inference_calls=2 if status != INVALID_PREFIX else 0,
        environment_steps=steps + 3 if stage != "cup_filled" else steps,
        handoff_observation_id=3 if stage != "cup_filled" else 0,
        failure_reason=None if success else "test_failure",
    )


def _manifest(created_at: str = "first") -> dict:
    return {
        "schema_version": "membench.stage_run_manifest/v1",
        "run_id": "test-stage-run",
        "created_at": created_at,
        "stage_suite": {"version": "v1", "hash": "sha256:stage"},
        "base_suite": {"version": "base", "hash": "sha256:base"},
        "task_id": "WA-05",
        "targets": ["cup_filled", "cup_staged"],
        "target_sources": {
            "cup_filled": "main_rollout",
            "cup_staged": "expert_prefix",
        },
        "seeds": [1, 2],
        "expert_prefix_root": "/prefixes",
        "main_rollout_run": "/main-run",
        "policy_contract": {"policy": {"name": "test"}},
    }


def test_stage_writer_commits_results_and_builds_combined_summary(tmp_path: Path) -> None:
    writer = StageArtifactWriter(tmp_path / "run")
    writer.write_manifest(_manifest())
    writer.write_manifest(_manifest(created_at="resume-time"))

    writer.write_result(_result("cup_filled", 1, SUCCESS, 7), source="main_rollout")
    writer.write_result(_result("cup_filled", 2, FAILURE, 9), source="main_rollout")
    writer.write_result(_result("cup_staged", 1, SUCCESS, 11), source="expert_prefix")
    writer.write_result(
        _result("cup_staged", 2, INVALID_PREFIX),
        source="expert_prefix",
    )

    summary = writer.write_summary()
    assert summary["complete"] is True
    assert summary["publishable"] is True
    assert summary["macro_stage_sr"] == pytest.approx(0.75)
    assert summary["stages"]["cup_filled"]["stage_sr"] == 0.5
    assert summary["stages"]["cup_filled"]["prefix_validity"] is None
    assert summary["stages"]["cup_staged"]["stage_sr"] == 1.0
    assert summary["stages"]["cup_staged"]["prefix_validity"] == 0.5
    assert summary["stages"]["cup_staged"]["model_control_steps_median"] == 11.0

    result_path = (
        writer.run_dir
        / "stages/cup_staged/episodes/seed_000001/result.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["source"] == "expert_prefix"
    assert (result_path.parent / "actions.npz").is_file()
    assert (result_path.parent / "steps.jsonl").is_file()


def test_stage_summary_marks_expected_missing_seed_pending(tmp_path: Path) -> None:
    writer = StageArtifactWriter(tmp_path / "run")
    writer.write_manifest(_manifest())
    writer.write_result(_result("cup_filled", 1, SUCCESS), source="main_rollout")

    summary = writer.write_summary()
    assert summary["complete"] is False
    assert summary["publishable"] is False
    assert summary["stages"]["cup_filled"]["pending_seeds"] == [2]
    assert summary["stages"]["cup_staged"]["pending_seeds"] == [1, 2]


def test_stage_session_serializes_dataclass_privileged_state(tmp_path: Path) -> None:
    @dataclass
    class PrivateState:
        completed: bool

    writer = StageArtifactWriter(tmp_path / "run")
    session = writer.start_episode(
        target_stage="cup_staged",
        seed=1,
        task=None,
        record_video=False,
        video_cameras=(),
        video_fps=1,
    )
    session.record(
        {
            "privileged_stage_state": PrivateState(completed=True),
            "executed_action": [0.0] * 13,
        }
    )
    session.finish(
        _result("cup_staged", 1, SUCCESS),
        source="expert_prefix",
        request_metrics=[],
    )
    line = (
        writer.run_dir
        / "stages/cup_staged/episodes/seed_000001/steps.jsonl"
    ).read_text(encoding="utf-8")
    assert json.loads(line)["privileged_stage_state"] == {"completed": True}
