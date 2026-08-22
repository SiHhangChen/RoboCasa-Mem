from __future__ import annotations

from pathlib import Path

import pytest

from scripts.annotate.apply_subtask_labels import label_episode, load_config, match_subtask
from scripts.annotate.planner_to_stage_boundaries_ts02 import (
    build_search_instruction,
    build_stage_names,
    count_stage_steps,
)


def test_ts02_drawer_trace_count_is_authoritative_but_button_uses_step_fields() -> None:
    drawer_steps, drawer_fields = count_stage_steps(
        {
            "actuation": {
                "primitive": "physical_drawer_pull",
                "trace_summary": {"sample_count": 395},
                "open_gripper": {"steps": 25},
                "close_gripper": {"steps": 25},
            }
        }
    )
    button_steps, button_fields = count_stage_steps(
        {
            "actuation": {
                "primitive": "physical_completion_button_press",
                "trace_summary": {"sample_count": 282},
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


@pytest.mark.parametrize("search_count", [1, 2, 3])
def test_ts02_stage_names_use_one_repeatable_button_label(
    search_count: int,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "ts02_subtasks.json")
    order = ["cabinet_slot_2", "cabinet_slot_0", "cabinet_slot_1"]
    names = build_stage_names(order, search_count)
    record = {
        "task_id": "TS02",
        "seed": 0,
        "search_count": search_count,
        "keyframes": [
            *[4 * drawer_number - 1 for drawer_number in range(1, search_count + 1)],
            4 * search_count,
        ],
        "stages": [
            {"name": name, "end_frame": index + 1}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=len(names))
    expected_drawer_indices = [2, 0, 1][:search_count]
    expected = []
    for drawer_index in expected_drawer_indices:
        expected.extend([drawer_index] * 4)
    # Navigation plus all K press-and-release stages stay in one high-level
    # subtask. The environment decides whether another press is required.
    expected.extend([3] * (search_count + 1))

    assert labeled["subtask_indices"] == expected
    assert labeled["keyframes"] == [
        *[4 * drawer_number - 1 for drawer_number in range(1, search_count + 1)],
        4 * search_count,
    ]


def test_ts02_search_instruction_exposes_order_without_target() -> None:
    instruction = build_search_instruction(
        ["cabinet_slot_1", "cabinet_slot_2", "cabinet_slot_0"]
    )

    assert "center, right, then left" in instruction
    assert "target drawer" not in instruction
    assert "contains the cube" in instruction
