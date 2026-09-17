from __future__ import annotations

from pathlib import Path

from scripts.annotate.apply_subtask_labels import (
    label_episode,
    load_config,
    match_subtask,
)
from scripts.annotate.planner_to_stage_boundaries_ts03 import (
    build_stage_names,
    pick_close_start_offset,
)


def test_ts03_stage_names_repeat_sugar_subtask_then_button() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "ts03_subtasks.json")
    names = build_stage_names(["sugar2", "sugar0"])

    matched = [match_subtask(name, config["stage_prefixes"]) for name in names]

    assert len(names) == 14
    assert matched == [*(["add_sugar"] * 12), *(["press_button"] * 2)]


def test_ts03_keeps_two_labels_and_one_keyframe_per_sugar_cycle() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "ts03_subtasks.json")
    names = build_stage_names(["sugar2", "sugar0"])
    record = {
        "task_id": "TS03",
        "seed": 3,
        "keyframes": [11, 23, 24],
        "stages": [
            {"name": name, "end_frame": (index + 1) * 2}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=28)

    assert labeled["subtask_order"] == [
        "add a sugar cube to the hot chocolate",
        "press the completion button",
    ]
    assert labeled["subtask_indices"] == [0] * 24 + [1] * 4
    assert labeled["keyframes"] == [11, 23, 24]


def test_ts03_pick_close_offset_uses_executed_pick_sequence() -> None:
    details = {
        "open_gripper": {"steps": 10},
        "preshape_gripper": {"steps": 5},
        "close_gripper": {"steps": 20},
        "stages": [
            {"stage": "pregrasp", "execution": {"executed_steps": 48}},
            {"stage": "grasp", "execution": {"executed_steps": 37}},
            {"stage": "lift", "execution": {"executed_steps": 38}},
        ],
    }
    assert pick_close_start_offset(details, path="pick") == 100
