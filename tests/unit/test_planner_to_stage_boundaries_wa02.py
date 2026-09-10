from __future__ import annotations

from pathlib import Path

from scripts.annotate.apply_subtask_labels import label_episode, load_config
from scripts.annotate.planner_to_stage_boundaries_wa02 import (
    build_stage_names,
    build_task_instruction,
    count_navigation_steps,
    count_sink_exit_lift_steps,
)


def test_wa02_navigation_uses_terminal_aggregate_once() -> None:
    steps, fields = count_navigation_steps(
        {
            "segments": [
                {"pre_rotate": {"steps": 31}, "post_translate": {"steps": 4}},
                {
                    "primitive": "settle_pose",
                    "steps": 94,
                    "pre_translate": {"steps": 74},
                    "rotate": {"steps": 4},
                    "post_translate": {"steps": 16},
                },
            ],
            "station_torso": {"steps": 30},
        }
    )

    assert steps == 159
    assert fields == {
        "executed_steps": 0,
        "release_settle_steps": 0,
        "steps": 159,
        "base_translation_steps": 0,
    }


def test_wa02_sink_exit_counts_base_and_attempts_without_final_duplicate() -> None:
    steps, fields = count_sink_exit_lift_steps(
        {
            "base_translation_steps": 90,
            "movement_attempts": [
                {"execution": {"executed_steps": 9}},
                {"execution": {"executed_steps": 8}},
            ],
            "movement": {"execution": {"executed_steps": 8}},
        }
    )

    assert steps == 107
    assert fields == {
        "executed_steps": 17,
        "release_settle_steps": 0,
        "steps": 0,
        "base_translation_steps": 90,
    }


def test_wa02_plan_produces_seven_task_semantic_labels() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wa02_subtasks.json")
    names = build_stage_names(
        "cup_B",
        "cup_A",
        {"cup_A": "slot_2", "cup_B": "slot_1"},
        "cup_A",
    )
    record = {
        "task_id": "WA-02",
        "seed": 7,
        "requested_service_side": "left",
        "keyframes": [2, 6, 10, 13, 17, 21, 22],
        "stages": [
            {"name": name, "end_frame": index + 1}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=26)
    labels = labeled["subtask_order"]

    assert [labels[index] for index in labeled["subtask_indices"]] == [
        *(["pick up the first cup"] * 3),
        *(["fill the held cup with hot water"] * 4),
        *(["place the held cup on the dining table to the left of the sink"] * 4),
        *(["return to the starting counter and pick up the remaining cup"] * 3),
        *(["fill the held cup with cold water"] * 4),
        *(["place the held cup on the dining table to the left of the sink"] * 4),
        *(["pick up the cup from the left side of the dining table"] * 4),
    ]
    assert labeled["keyframes"] == [2, 6, 10, 13, 17, 21, 22]


def test_wa02_cold_first_plan_keeps_temperature_labels_correct() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wa02_subtasks.json")
    names = build_stage_names(
        "cup_B",
        "cup_A",
        {"cup_A": "slot_2", "cup_B": "slot_1"},
        "cup_A",
        ("cold", "hot"),
    )
    record = {
        "task_id": "WA-02",
        "seed": 8,
        "requested_service_side": "left",
        "keyframes": [2, 6, 10, 13, 17, 21, 22],
        "stages": [
            {"name": name, "end_frame": index + 1}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=26)

    assert [
        labeled["subtask_order"][index]
        for index in labeled["subtask_indices"]
    ] == [
        *(["pick up the first cup"] * 3),
        *(["fill the held cup with cold water"] * 4),
        *(["place the held cup on the dining table to the left of the sink"] * 4),
        *(["return to the starting counter and pick up the remaining cup"] * 3),
        *(["fill the held cup with hot water"] * 4),
        *(["place the held cup on the dining table to the left of the sink"] * 4),
        *(["pick up the cup from the left side of the dining table"] * 4),
    ]


def test_wa02_instruction_exposes_only_requested_temperature() -> None:
    instruction = build_task_instruction("cold")

    assert instruction == (
        "Fill two identical cups with hot and cold water, stage both cups, "
        "then take the cold-water cup."
    )
    assert "first cup" not in instruction
    assert "remaining cup" not in instruction
    assert "left side" not in instruction
    assert "right side" not in instruction
    assert "cup_A" not in instruction
    assert "cup_B" not in instruction
