from __future__ import annotations

from pathlib import Path

from scripts.annotate.apply_subtask_labels import label_episode, load_config
from scripts.annotate.planner_to_stage_boundaries_wr05 import (
    build_stage_names,
    build_task_instruction,
    count_navigation_steps,
    count_place_steps,
)


def test_wr05_navigation_adds_station_torso_steps() -> None:
    steps, fields = count_navigation_steps(
        {"steps": 68, "station_torso": {"steps": 22}}
    )

    assert steps == 90
    assert fields["navigation_steps"] == 68
    assert fields["station_torso_steps"] == 22


def test_wr05_place_ignores_duplicated_early_stop_evidence() -> None:
    steps, fields = count_place_steps(
        {
            "release_gripper": {"steps": 35, "release_settle_steps": 10},
            "stages": [
                {"execution": {"executed_steps": 127}},
                {
                    "execution": {"executed_steps": 15},
                    "accepted_trajectory_early_stop_for_release": {
                        "early_stop": {"executed_steps": 15}
                    },
                },
                {"execution": {"executed_steps": 43}},
            ],
        }
    )

    assert steps == 230
    assert fields["executed_steps"] == 185
    assert fields["steps"] == 35
    assert fields["release_settle_steps"] == 10


def test_wr05_labels_eight_executable_subtasks_with_nine_keyframes() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wr05_subtasks.json")
    home = {"ketchup": "shelf_upper", "mustard": "shelf_lower"}
    plate = {"ketchup": "plate_left", "mustard": "plate_right"}
    names = build_stage_names(home, plate)
    record = {
        "task_id": "WR-05",
        "seed": 0,
        "ketchup_original_level": "upper",
        "mustard_original_level": "lower",
        "ketchup_plate_side": "left",
        "mustard_plate_side": "right",
        "stages": [
            {"name": name, "end_frame": index + 1}
            for index, name in enumerate(names)
        ],
        "keyframes": [0, 4, 7, 10, 13, 16, 19, 22, 25],
    }

    labeled = label_episode(record, config, recorded_total=26)
    labels = [labeled["subtask_order"][index] for index in labeled["subtask_indices"]]

    assert labels == [
        *( ["pick up the ketchup bottle from the upper shelf"] * 5 ),
        *( ["place the ketchup bottle beside the serving plate"] * 3 ),
        *( ["pick up the mustard bottle from the lower shelf"] * 3 ),
        *( ["place the mustard bottle beside the serving plate"] * 3 ),
        *( ["pick up the ketchup bottle from beside the serving plate"] * 3 ),
        *( ["return the ketchup bottle upright to the upper shelf"] * 3 ),
        *( ["pick up the mustard bottle from beside the serving plate"] * 3 ),
        *( ["return the mustard bottle upright to the lower shelf"] * 3 ),
    ]
    assert labeled["keyframes"] == [0, 4, 7, 10, 13, 16, 19, 22, 25]


def test_wr05_instruction_hides_plate_sides_and_home_assignment() -> None:
    instruction = build_task_instruction()

    assert "First move the ketchup beside the serving plate" in instruction
    assert "then move the mustard beside it" in instruction
    assert "return the ketchup and then the mustard" in instruction
    assert "original shelf levels" in instruction
    assert "shelf_upper" not in instruction
    assert "shelf_lower" not in instruction
    assert "left side" not in instruction
    assert "right side" not in instruction
