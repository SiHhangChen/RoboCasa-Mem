from __future__ import annotations

import json

import pytest

from scripts.eval import summarize_wa02_stages as summary


def _authoritative_episode(
    *,
    seed: int = 1,
    status: str = "failure",
    qualified_through: int = 2,
) -> dict:
    stages = {
        stage: {
            "observed": index <= qualified_through,
            "qualified": index <= qualified_through,
        }
        for index, stage in enumerate(summary.STAGES)
    }
    return {
        "seed": seed,
        "status": status,
        "success": qualified_through == len(summary.STAGES) - 1,
        "diagnostics": {
            "evaluation": {"profile_id": summary.EVALUATOR_PROFILE},
            "stages": stages,
        },
    }


def _legacy_episode() -> dict:
    return {
        "seed": 2,
        "status": "failure",
        "success": False,
        "episode_metadata": {
            "resolved_episode": {
                "facts": {
                    "hot_cup_id": "cup_B",
                    "cold_cup_id": "cup_A",
                    "fill_temperature_order": ["cold", "hot"],
                    "requested_cup_id": "cup_A",
                    "service_slot_by_cup": {
                        "cup_A": "slot_1",
                        "cup_B": "slot_2",
                    },
                }
            }
        },
        "diagnostics": {
            "fill": {
                "recorded_temperature_by_cup": {
                    "cup_A": "cold",
                    "cup_B": "hot",
                }
            },
            "service": {
                "recorded_slot_by_cup": {
                    "cup_A": "slot_1",
                    "cup_B": "slot_2",
                }
            },
            "selection": {"final_taken_cup_id": "cup_A"},
        },
    }


def _write_run(tmp_path, episodes: list[dict]):
    run = tmp_path / "wa02-test-run"
    (run / "episodes").mkdir(parents=True)
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "wa02-test-run",
                "seeds": [episode["seed"] for episode in episodes],
                "task": {"id": "WA-02"},
                "policy_contract": {
                    "checkpoint": {"uri": "file:///tmp/ckpts/WA02/090000/model"}
                },
            }
        ),
        encoding="utf-8",
    )
    for episode in episodes:
        directory = run / "episodes" / f"seed_{episode['seed']:06d}"
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
    return run


def test_authoritative_stage_hits_use_qualified_values() -> None:
    episode = _authoritative_episode(qualified_through=1)
    episode["diagnostics"]["stages"]["second_cup_ready"]["observed"] = True

    source, hits = summary.stage_values(episode)

    assert source == summary.EVALUATOR_PROFILE
    assert hits["first_cup_fill"]
    assert not hits["first_cup_ready"]
    assert not hits["second_cup_ready"]


def test_legacy_episode_is_partial_and_keeps_pick_stages_unknown() -> None:
    source, hits = summary.stage_values(_legacy_episode())

    assert source == "legacy_partial"
    assert hits["first_cup_pick"] is None
    assert hits["second_cup_pick"] is None
    assert hits["first_cup_fill"] is True
    assert hits["second_cup_fill"] is True
    assert hits["selection_success"] is True


def test_summary_excludes_legacy_and_error_episodes_from_funnel(tmp_path) -> None:
    run = _write_run(
        tmp_path,
        [
            _authoritative_episode(seed=1, qualified_through=2),
            _authoritative_episode(seed=3, status="error", qualified_through=6),
            _legacy_episode(),
        ],
    )

    row = summary.summarize_run(run)

    assert row["n_episodes"] == 3
    assert row["n_ok"] == 2
    assert row["n_funnel"] == 1
    assert row["n_authoritative"] == 2
    assert row["n_legacy_partial"] == 1
    assert row["first_cup_ready_n"] == 1
    assert row["second_cup_pick_n"] == 0
    assert row["first_cup_ready_pct"] == 100.0
    assert row["checkpoint_step"] == "090000"


def test_summary_rejects_non_monotonic_authoritative_stages() -> None:
    episode = _authoritative_episode(qualified_through=0)
    episode["diagnostics"]["stages"]["second_cup_fill"]["qualified"] = True

    with pytest.raises(ValueError, match="non-monotonic WA-02 stages"):
        summary.stage_values(episode)
