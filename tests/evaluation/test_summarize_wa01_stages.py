from __future__ import annotations

from scripts.eval import summarize_gr00t_stages, summarize_pi05_wa01


def _authoritative_episode() -> dict:
    qualified = {
        "sugar": True,
        "one_ready": True,
        "ready": False,
        "selection": False,
        "correct": False,
        "success": False,
    }
    return {
        "success": False,
        "diagnostics": {
            "stages": {
                name: {
                    "observed": True,
                    "qualified": value,
                }
                for name, value in qualified.items()
            },
            # These legacy fields deliberately claim all later stages.  The
            # new summaries must ignore them when authoritative stages exist.
            "physical_events": {
                "sugar_delivery": {"coffee_id": "coffee_A"},
                "cups_ready_latched": True,
                "selection": {"coffee_id": "coffee_A"},
            },
            "coffee_A": {"on_dining_counter": True, "upright": True},
            "coffee_B": {"on_dining_counter": True, "upright": True},
            "selection": {"selected_coffee_id": "coffee_A", "correct": True},
        },
    }


def test_wa01_stage_summaries_use_qualified_not_observed_or_legacy_fields() -> None:
    episode = _authoritative_episode()

    assert summarize_gr00t_stages.stage_hits(episode) == {
        "sugar": True,
        "one_ready": True,
        "ready": False,
        "selection": False,
        "correct": False,
        "success": False,
    }
    assert summarize_pi05_wa01._episode_metrics(episode) == (
        True,
        True,
        False,
        False,
        False,
        False,
    )


def test_wa01_stage_summaries_keep_legacy_result_compatibility() -> None:
    episode = {
        "success": True,
        "status": "success",
        "diagnostics": {
            "physical_events": {
                "sugar_delivery": {"coffee_id": "coffee_A"},
                "cups_ready_latched": True,
                "selection": {"coffee_id": "coffee_A"},
            },
            "coffee_A": {"on_dining_counter": True, "upright": True},
            "coffee_B": {"on_dining_counter": True, "upright": True},
            "sugar": {"consumed": True},
            "selection": {"selected_coffee_id": "coffee_A", "correct": True},
        },
    }

    assert all(summarize_gr00t_stages.stage_hits(episode).values())
    assert all(summarize_pi05_wa01._episode_metrics(episode))
