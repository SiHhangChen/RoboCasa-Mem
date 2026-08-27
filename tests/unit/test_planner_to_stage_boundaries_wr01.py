from __future__ import annotations

from pathlib import Path

import pytest

from scripts.annotate.apply_subtask_labels import label_episode, load_config, match_subtask
from scripts.annotate.planner_to_stage_boundaries_wr01 import (
    build_stage_names,
    count_recorded_steps,
)


def test_wr01_step_count_does_not_double_count_nested_aggregates() -> None:
    total, fields = count_recorded_steps(
        {
            "segments": [
                {"steps": 48, "pre_translate": {"steps": 48}},
                {"pre_rotate": {"steps": 6}, "post_translate": {"steps": 4}},
            ],
            "execution": {"executed_steps": 9, "early_stop": {"executed_steps": 9}},
        }
    )
    assert total == 67
    assert fields == {"executed_steps": 9, "release_settle_steps": 0, "steps": 58}


@pytest.mark.parametrize(
    ("assignment", "target_level"),
    [
        ({"lower": "red_can", "upper": "blue_can"}, "lower"),
        ({"lower": "red_can", "upper": "blue_can"}, "upper"),
        ({"lower": "blue_can", "upper": "red_can"}, "lower"),
        ({"lower": "blue_can", "upper": "red_can"}, "upper"),
    ],
)
def test_wr01_stage_names_match_eight_semantic_groups(assignment: dict[str, str], target_level: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wr01_subtasks.json")
    names = build_stage_names(assignment, target_level)
    matched = [match_subtask(name, config["stage_prefixes"]) for name in names]
    assert len(names) == 27
    assert matched == [
        *(["pick_first"] * 3),
        *(["store_lower"] * 4),
        *(["pick_remaining"] * 3),
        *(["store_upper"] * 4),
        *(["close_upper"] * 3),
        *(["close_lower"] * 4),
        *(["open_remembered"] * 4),
        *(["retrieve"] * 2),
    ]


@pytest.mark.parametrize(("target_level", "open_index"), [("upper", 6), ("lower", 7)])
def test_wr01_only_retrieval_door_label_depends_on_hidden_relation(target_level: str, open_index: int) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wr01_subtasks.json")
    assignment = {"lower": "blue_can", "upper": "red_can"}
    names = build_stage_names(assignment, target_level)
    record = {
        "task_id": "WR-01",
        "seed": 0,
        "target_level": target_level,
        "stages": [{"name": name, "end_frame": index + 1} for index, name in enumerate(names)],
    }
    labeled = label_episode(record, config, recorded_total=len(names))
    assert labeled["subtask_indices"] == [
        *([0] * 3),
        *([1] * 4),
        *([2] * 3),
        *([3] * 4),
        *([4] * 3),
        *([5] * 4),
        *([open_index] * 4),
        *([8] * 2),
    ]
    assert labeled["keyframes"] == [2, 6, 9, 13, 16, 20, 21, 26]
    assert labeled["subtask_order"][6] == "open the upper cabinet door"
    assert labeled["subtask_order"][7] == "open the lower cabinet door"
