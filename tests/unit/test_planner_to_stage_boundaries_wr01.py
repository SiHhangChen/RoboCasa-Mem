from __future__ import annotations

from pathlib import Path

import pytest

from scripts.annotate.apply_subtask_labels import label_episode, load_config, match_subtask
from scripts.annotate.planner_to_stage_boundaries_wr01 import (
    _validate_stage_primitive,
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


@pytest.mark.parametrize("index", [0, 7])
def test_wr01_accepts_station_navigation_without_primitive(index: int) -> None:
    _validate_stage_primitive(
        seed=0,
        index=index,
        details={"steps": 12},
        expected="manual_base_path_navigate",
    )


def test_wr01_rejects_other_missing_primitives() -> None:
    with pytest.raises(ValueError, match="stage 1 primitive mismatch"):
        _validate_stage_primitive(
            seed=0,
            index=1,
            details={"steps": 12},
            expected="manual_base_path_navigate",
        )


@pytest.mark.parametrize(
    ("assignment", "target_level"),
    [
        ({"lower": "red_can", "upper": "blue_can"}, "lower"),
        ({"lower": "red_can", "upper": "blue_can"}, "upper"),
        ({"lower": "blue_can", "upper": "red_can"}, "lower"),
        ({"lower": "blue_can", "upper": "red_can"}, "upper"),
    ],
)
def test_wr01_stage_names_match_current_semantic_groups(assignment: dict[str, str], target_level: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wr01_subtasks.json")
    names = build_stage_names(assignment, target_level)
    matched = [match_subtask(name, config["stage_prefixes"]) for name in names]
    assert len(names) == 20
    assert matched == [
        "observe",
        *(["close_first"] * 3),
        *(["close_second"] * 3),
        *(["handoff"] * 3),
        *(["open_remembered"] * 4),
        *(["retrieve"] * 3),
        *(["place"] * 3),
    ]


@pytest.mark.parametrize("target_level", ["upper", "lower"])
def test_wr01_labels_do_not_expose_the_hidden_relation(target_level: str) -> None:
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
        0,
        *([1] * 3),
        *([2] * 3),
        *([3] * 3),
        *([4] * 4),
        *([5] * 3),
        *([6] * 3),
    ]
    assert labeled["keyframes"] == [0, 3, 6, 9, 10, 16, 19]
    assert labeled["subtask_order"][4] == "open the remembered cabinet door"
