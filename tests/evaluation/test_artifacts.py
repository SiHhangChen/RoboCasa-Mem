from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.evaluation.artifacts import ArtifactWriter


def _finish(writer: ArtifactWriter, seed: int, status: str) -> None:
    session = writer.start_episode(
        seed=seed,
        record_video=False,
        video_cameras=(),
        video_fps=20,
    )
    session.append_action(np.zeros(13, dtype=np.float32))
    session.append_request_metric({"client_latency_ms": 1.0})
    session.finish(
        {
            "seed": seed,
            "status": status,
            "success": status == "success",
            "success_step": 1 if status == "success" else None,
            "inference_latency_ms": [1.0],
        }
    )


def _ts01_k1_report(*, success: bool) -> dict[str, object]:
    return {
        "schema_version": "membench.report_stages/v1",
        "task_id": "TS01",
        "profile_id": "TS01/report-v1",
        "order": ["sugar_1", "sugar_2", "sugar_3", "success"],
        "stages": {
            "sugar_1": {
                "applicable": True,
                "passed": True,
                "first_pass_step": 1,
            },
            "sugar_2": {
                "applicable": False,
                "passed": None,
                "first_pass_step": None,
            },
            "sugar_3": {
                "applicable": False,
                "passed": None,
                "first_pass_step": None,
            },
            "success": {
                "applicable": True,
                "passed": success,
                "first_pass_step": 2 if success else None,
            },
        },
    }


def _finish_with_report(writer: ArtifactWriter, seed: int) -> None:
    session = writer.start_episode(
        seed=seed,
        record_video=False,
        video_cameras=(),
        video_fps=20,
    )
    session.append_action(np.zeros(13, dtype=np.float32))
    session.finish(
        {
            "seed": seed,
            "task_id": "TS01",
            "status": "success",
            "success": True,
            "success_step": 2,
            "report_stages": _ts01_k1_report(success=True),
        }
    )


def test_episode_commit_resume_and_summary(tmp_path: Path):
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=(1000, 1001))
    _finish(writer, 1000, "success")
    _finish(writer, 1001, "failure")

    assert writer.completed_seeds() == {1000, 1001}
    assert (writer.episodes_dir / "seed_001000" / "actions.npz").is_file()
    summary = writer.write_summary()
    assert summary["counts"]["success"] == 1
    assert summary["counts"]["failure"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["planned"] == summary["recorded"] == 2
    assert summary["pending"] == 0
    assert summary["publishable"] is True


def test_retry_errors_replaces_episode_without_deleting_first(tmp_path: Path):
    writer = ArtifactWriter(tmp_path / "run")
    _finish(writer, 1000, "error")
    assert writer.completed_seeds(retry_errors=True) == set()

    _finish(writer, 1000, "success")

    result = json.loads(
        (writer.episodes_dir / "seed_001000" / "result.json").read_text()
    )
    assert result["status"] == "success"
    assert not list(writer.episodes_dir.glob(".seed_001000.backup.*"))


def test_resume_rejects_different_checkpoint_identity(tmp_path: Path):
    writer = ArtifactWriter(tmp_path / "run")
    manifest = {
        "run_id": "run-1",
        "suite": {"hash": "sha256:suite"},
        "task": {"id": "WX-01"},
        "seeds": [1000],
        "policy_contract": {
            "checkpoint": {"fingerprint": "sha256-manifest:first"},
            "observation": {"state_schema": "membench.robot_state37.v2"},
            "action": {"schema": "membench.pandaomron_hybrid13.v2"},
        },
        "rollout": {"replan_steps": 50},
    }
    writer.write_run_manifest(manifest)
    writer.write_run_manifest(manifest)
    changed = {
        **manifest,
        "policy_contract": {
            **manifest["policy_contract"],
            "checkpoint": {"fingerprint": "sha256-manifest:second"},
        },
    }

    with pytest.raises(ValueError, match="identity mismatch"):
        writer.write_run_manifest(changed)


def test_resume_rejects_different_camera_contract(tmp_path: Path):
    writer = ArtifactWriter(tmp_path / "run")
    manifest = {
        "run_id": "run-1",
        "suite": {"hash": "sha256:suite", "version": "v1"},
        "task": {"id": "WX-01", "config_sha256": "sha256:task"},
        "seeds": [1],
        "policy_contract": {
            "checkpoint": {"fingerprint": "sha256:checkpoint"},
            "observation": {
                "state_schema": "membench.robot_state37.v2",
                "required_cameras": ["agentview_left", "agentview_right", "eye_in_hand"],
                "image_size": [224, 224],
            },
            "action": {
                "schema": "membench.pandaomron_hybrid13.v2",
                "horizon": 50,
            },
            "control_frequency_hz": 20,
        },
        "rollout": {"replan_steps": 50},
        "artifacts": {"record_video": False},
    }
    writer.write_run_manifest(manifest)
    changed = json.loads(json.dumps(manifest))
    changed["policy_contract"]["observation"]["image_size"] = [256, 256]

    with pytest.raises(ValueError, match="identity mismatch"):
        writer.write_run_manifest(changed)


def test_invalid_committed_result_is_not_resumable_or_publishable(tmp_path: Path):
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=(7,))
    episode_dir = writer.episodes_dir / "seed_000007"
    episode_dir.mkdir()
    (episode_dir / "actions.npz").write_bytes(b"not-used-by-shallow-validation")
    (episode_dir / "request_metrics.jsonl").write_text("", encoding="utf-8")
    (episode_dir / "result.json").write_text(
        json.dumps({"seed": 8, "status": "maybe", "success": False}),
        encoding="utf-8",
    )

    assert writer.completed_seeds() == set()
    summary = writer.write_summary()
    assert summary["pending"] == 1
    assert summary["publishable"] is False
    assert summary["invalid_episode_artifacts"]


def test_requested_video_camera_must_be_present(tmp_path: Path):
    writer = ArtifactWriter(tmp_path / "run")
    session = writer.start_episode(
        seed=1,
        record_video=True,
        video_cameras=("eye_in_hand",),
        video_fps=20,
    )

    with pytest.raises(KeyError, match="eye_in_hand"):
        session.append_frames({"agentview_right": np.zeros((2, 2, 3), dtype=np.uint8)})
    session.abort()


def test_report_stages_persist_and_remain_resumable(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=(7,))
    _finish_with_report(writer, 7)

    assert writer.completed_seeds() == {7}
    result = writer.load_episode_results()[0]
    assert result["report_stages"]["profile_id"] == "TS01/report-v1"
    assert result["report_stages"]["stages"]["sugar_2"]["applicable"] is False
    summary = writer.write_summary()
    assert summary["report_stages"]["complete"] is True
    assert summary["publishable"] is True


def test_invalid_report_stage_artifact_is_not_resumable(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=(7,))
    _finish_with_report(writer, 7)
    result_path = writer.episodes_dir / "seed_000007" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["report_stages"]["stages"]["success"] = {
        "applicable": True,
        "passed": False,
        "first_pass_step": None,
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    assert writer.completed_seeds() == set()
    summary = writer.write_summary()
    assert summary["pending"] == 1
    assert summary["publishable"] is False
    assert "episode success" in next(iter(summary["invalid_episode_artifacts"].values()))
