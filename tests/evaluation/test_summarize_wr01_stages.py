from __future__ import annotations

import json

from scripts.eval import summarize_wr01_stages as summary


def _episode(*, success: bool = False) -> dict:
    stages = {
        stage: {
            "observed": True,
            "qualified": success,
        }
        for stage in summary.STAGES
    }
    return {
        "seed": 0,
        "success": success,
        "diagnostics": {
            "success": success,
            "stages": stages,
        },
    }


def test_stage_hits_use_qualified_evaluator_values() -> None:
    episode = _episode()
    episode["diagnostics"]["stages"]["second_can_pick"]["observed"] = True
    episode["diagnostics"]["stages"]["second_can_pick"]["qualified"] = False

    hits = summary.stage_hits(episode)

    assert not hits["second_can_pick"]
    assert not hits["retrieve_item"]


def test_success_requires_final_result_and_final_stage() -> None:
    episode = _episode(success=True)
    assert all(summary.stage_hits(episode).values())

    episode["success"] = False
    assert not summary.stage_hits(episode)["retrieve_item"]


def test_native_replay_summary_loader_reads_results(tmp_path) -> None:
    run = tmp_path / "wr01-run"
    run.mkdir()
    (run / "manifest.json").write_text(
        '{"task_id":"WR-01","replay":{"evaluator_profile":"test"}}',
        encoding="utf-8",
    )
    episode = _episode(success=True)
    (run / "summary.json").write_text(
        '{"results":[' + json.dumps(episode) + ']}',
        encoding="utf-8",
    )

    rows = summary.load_episodes(run)

    assert rows == [episode]
    assert summary.summarize_run(run)["retrieve_item_n"] == 1
