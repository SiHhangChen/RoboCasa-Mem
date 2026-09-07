from __future__ import annotations

from pathlib import Path

from scripts.annotate.apply_subtask_labels import label_episode, load_config
from scripts.annotate.planner_to_stage_boundaries_wr06 import (
    build_stage_names,
    build_task_instruction,
    count_navigation_steps,
    count_place_steps,
)


def test_wr06_navigation_adds_station_torso_steps() -> None:
    steps, fields = count_navigation_steps(
        {"steps": 62, "station_torso": {"steps": 29}}
    )

    assert steps == 91
    assert fields["navigation_steps"] == 62
    assert fields["station_torso_steps"] == 29


def test_wr06_place_ignores_duplicated_early_stop_evidence() -> None:
    steps, fields = count_place_steps(
        {
            "release_gripper": {"steps": 35, "release_settle_steps": 10},
            "stages": [
                {"execution": {"executed_steps": 127}},
                {
                    "execution": {
                        "executed_steps": 15,
                        "early_stop": {"executed_steps": 15},
                    },
                    "accepted_trajectory_early_stop_for_release": {
                        "early_stop": {"executed_steps": 15}
                    },
                },
                {"execution": {"executed_steps": 43}},
                {"execution": {"executed_steps": 21}},
            ],
        }
    )

    assert steps == 251
    assert fields["executed_steps"] == 206
    assert fields["steps"] == 35
    assert fields["release_settle_steps"] == 10


def test_wr06_labels_five_memory_aligned_subtasks() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wr06_subtasks.json")
    names = build_stage_names("shelf_upper", "plate_right")
    record = {
        "task_id": "WR-06",
        "seed": 3,
        "original_level": "upper",
        "plate_side": "right",
        "stages": [
            {"name": name, "end_frame": index + 1}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=15)
    labels = labeled["subtask_order"]

    assert [labels[index] for index in labeled["subtask_indices"]] == [
        *(["inspect the shelf levels to locate the ketchup bottle"] * 2),
        *(["pick up the ketchup bottle from the upper shelf"] * 3),
        *(["place the ketchup bottle beside the serving plate"] * 4),
        *(["pick up the ketchup bottle from beside the serving plate"] * 3),
        *(["return the ketchup bottle upright to the upper shelf"] * 3),
    ]
    assert labeled["keyframes"] == [1, 4, 8, 11, 14]


def test_wr06_instruction_hides_demo_side_and_original_level() -> None:
    instruction = build_task_instruction()

    assert "beside the serving plate" in instruction
    assert "whether the ketchup bottle starts on the lower or upper shelf" in instruction
    assert "original shelf level" in instruction
    assert "shelf_upper" not in instruction
    assert "left side" not in instruction
    assert "right side" not in instruction
