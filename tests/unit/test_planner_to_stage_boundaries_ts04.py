from __future__ import annotations

from pathlib import Path

from scripts.annotate.apply_subtask_labels import label_episode, load_config
from scripts.annotate.planner_to_stage_boundaries_ts04 import (
    build_search_instruction,
    build_stage_names,
    count_stage_steps,
)


def test_ts04_drawer_trace_is_authoritative_but_button_uses_step_fields() -> None:
    drawer_steps, drawer_fields = count_stage_steps(
        {
            "actuation": {
                "primitive": "physical_drawer_pull",
                "trace_summary": {"sample_count": 395},
                "open_gripper": {"steps": 25},
            }
        }
    )
    button_steps, button_fields = count_stage_steps(
        {
            "actuation": {
                "primitive": "physical_completion_button_press",
                "trace_summary": {"sample_count": 289},
                "stages": {
                    "hold": {"steps": 12},
                    "prepress": {"execution": {"executed_steps": 195}},
                    "press": {"execution": {"executed_steps": 37}},
                    "retreat": {"execution": {"executed_steps": 36}},
                },
            }
        }
    )

    assert drawer_steps == 395
    assert drawer_fields["trace_samples"] == 395
    assert button_steps == 280
    assert button_fields == {
        "executed_steps": 268,
        "release_settle_steps": 0,
        "steps": 12,
        "trace_samples": 0,
    }


def test_ts04_stage_names_repeat_button_for_search_length() -> None:
    order = ["cabinet_slot_1", "cabinet_slot_0"]

    assert build_stage_names(order, 1)[-2:] == [
        "navigate_completion_button",
        "press_completion_button_1_of_1",
    ]
    assert build_stage_names(order, 2)[-3:] == [
        "navigate_completion_button",
        "press_completion_button_1_of_2",
        "press_completion_button_2_of_2",
    ]


def test_ts04_labels_two_drawers_and_repeated_button_subtask() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "ts04_subtasks.json")
    names = build_stage_names(["cabinet_slot_1", "cabinet_slot_0"], 2)
    record = {
        "task_id": "TS04",
        "seed": 3,
        "keyframes": [9, 19, 20],
        "stages": [
        {"name": name, "end_frame": (index + 1) * 2}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=24)

    assert labeled["subtask_order"] == [
        "search the left drawer for the red cube, then close it",
        "search the right drawer for the red cube, then close it",
        "press and release the completion button",
    ]
    assert labeled["subtask_indices"] == [1] * 10 + [0] * 10 + [2] * 4
    assert labeled["keyframes"] == [9, 19, 20]


def test_ts04_search_instruction_exposes_order_without_target() -> None:
    instruction = build_search_instruction(["cabinet_slot_1", "cabinet_slot_0"])

    assert "right, then left" in instruction
    assert "target drawer" not in instruction
    assert "contains the cube" in instruction
