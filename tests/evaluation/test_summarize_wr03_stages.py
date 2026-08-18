from __future__ import annotations

from scripts.eval import summarize_wr03_stages as summary


def _episode(*, success: bool = False) -> dict:
    stages = {
        stage: {
            "observed": True,
            "qualified": success,
        }
        for stage in summary.STAGES
        if stage != "success"
    }
    return {
        "status": "success" if success else "failure",
        "success": success,
        "failure_reason": None if success else "incomplete",
        "diagnostics": {
            "success": success,
            "stages": stages,
            "retrieval": {"protocol_violation": False},
            # Deliberately contradictory legacy values: authoritative stages
            # must win whenever the new evaluator payload is present.
            "success_checks": {
                "can_storage_verified": True,
                "both_doors_closed_after_storage": True,
                "memory_barrier_latched": True,
                "correct_retrieval_door": True,
                "no_wrong_retrieval_door": True,
                "same_can_retrieved": True,
            },
        },
    }


def test_wr03_summary_uses_qualified_evaluator_stages() -> None:
    episode = _episode()
    episode["diagnostics"]["stages"]["can_pick"]["qualified"] = True

    hits = summary.stage_hits(episode)

    assert hits["can_pick"]
    assert not hits["can_store"]
    assert not hits["success"]


def test_wr03_summary_requires_consistent_final_success() -> None:
    episode = _episode(success=True)
    assert all(summary.stage_hits(episode).values())

    episode["diagnostics"]["success"] = False
    assert not summary.stage_hits(episode)["success"]


def test_wr03_legacy_results_do_not_fabricate_pick_stage() -> None:
    episode = _episode(success=True)
    del episode["diagnostics"]["stages"]

    hits = summary.stage_hits(episode)

    assert not hits["can_pick"]
    assert hits["can_store"]
    assert hits["retrieve_can"]
