from __future__ import annotations

import json

import numpy as np

from bench.evaluation.artifacts import ArtifactWriter
from scripts.eval.summarize_run import main, summarize_run


def _write_success(writer: ArtifactWriter, seed: int) -> None:
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
            "status": "success",
            "success": True,
            "success_step": 1,
        }
    )


def test_summary_can_be_rebuilt_and_used_as_a_gate(tmp_path, capsys):
    run_dir = tmp_path / "run"
    writer = ArtifactWriter(run_dir, expected_seeds=(1, 2))
    writer.write_run_manifest(
        {
            "run_id": "test",
            "suite": {"hash": "sha256:test", "version": "test"},
            "task": {"id": "WX-01"},
            "seeds": [1, 2],
            "policy_contract": {},
            "rollout": {},
            "artifacts": {},
        }
    )
    _write_success(writer, 1)

    assert summarize_run(run_dir)["pending"] == 1
    assert main([str(run_dir)]) == 2
    assert json.loads(capsys.readouterr().out)["publishable"] is False

    _write_success(writer, 2)
    assert main([str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["publishable"] is True
