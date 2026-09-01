from __future__ import annotations

from pathlib import Path

from scripts.annotate.apply_subtask_labels import load_config, match_subtask
from scripts.annotate.planner_to_stage_boundaries_wa03 import build_stage_names


def test_build_stage_names_matches_current_wa03_plan() -> None:
    names = build_stage_names(["coffee_A", "coffee_B", "coffee_C"])

    assert len(names) == 39
    assert names[34] == "return_after_place_third_coffee_C_dining_to_regular_pose"
    assert names[35] == "verify_all_cups_ready_for_selection"
    assert names[-1] == "select_sugared_coffee"


def test_current_stage_names_match_wa03_subtask_config() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wa03_subtasks.json")
    names = build_stage_names(["coffee_A", "coffee_B", "coffee_C"])

    matched = [match_subtask(name, config["stage_prefixes"]) for name in names]

    assert matched == [
        *(["add_sugar"] * 7),
        *(["place_first"] * 8),
        *(["place_second"] * 10),
        *(["place_third"] * 11),
        *(["identify"] * 3),
    ]


def test_build_stage_names_keeps_legacy_38_stage_support() -> None:
    names = build_stage_names(
        ["coffee_A", "coffee_B", "coffee_C"],
        include_post_third_return=False,
    )

    assert len(names) == 38
    assert names[34] == "verify_all_cups_ready_for_selection"
    assert not any(name.startswith("return_after_place_third_") for name in names)
