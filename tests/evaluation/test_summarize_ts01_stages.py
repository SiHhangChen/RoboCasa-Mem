from __future__ import annotations

import json

from scripts.eval import summarize_ts01_stages as summary


def _episode(
    *,
    target: int,
    total: int,
    success: bool,
) -> dict:
    stages = {}
    for index, stage in enumerate(summary.STAGES[:3], start=1):
        applicable = index <= target
        stages[stage] = {
            "qualified": bool(success and applicable),
            "effective_success": bool(success),
        }
    stages["button"] = {
        "qualified": success,
        "effective_success": success,
    }
    stages["success"] = {
        "qualified": success,
        "effective_success": success,
    }
    return {
        "seed": 0,
        "status": "success" if success else "failure",
        "success": success,
        "failure_reason": None if success else "max_steps",
        "diagnostics": {
            "success": success,
            "target_sugar_count": target,
            "total_sugar_count": total,
            "stages": stages,
        },
    }


def test_stage_summary_preserves_padding_and_groups_by_cell(tmp_path) -> None:
    run = tmp_path / "ts01-run"
    episode_dir = run / "episodes" / "seed_000001"
    episode_dir.mkdir(parents=True)
    episode = _episode(target=1, total=3, success=True)
    (episode_dir / "result.json").write_text(
        json.dumps(episode),
        encoding="utf-8",
    )

    result = summary.summarize_run(run)

    assert result["overall"]["success_count"] == 1
    assert result["cells"]["k1_n3"]["episodes"] == 1
    assert result["cells"]["k1_n3"]["stages"]["sugar_2"] == {
        "qualified_count": 0,
        "qualified_rate": 0.0,
        "effective_count": 1,
        "effective_rate": 1.0,
    }


def test_final_success_requires_episode_diagnostics_and_stage_agreement() -> None:
    episode = _episode(target=2, total=5, success=True)
    assert summary.episode_success(episode)

    episode["diagnostics"]["stages"]["success"]["qualified"] = False
    assert not summary.episode_success(episode)


def test_invalid_or_missing_k_n_is_not_grouped() -> None:
    episode = _episode(target=1, total=3, success=False)
    del episode["diagnostics"]["total_sugar_count"]

    assert summary.cell_id(episode) is None
    assert summary.aggregate([episode])["episodes"] == 1
