from __future__ import annotations

from pathlib import Path

import pytest

from scripts.annotate.apply_subtask_labels import label_episode, load_config, match_subtask
from scripts.annotate.planner_to_stage_boundaries_wr03 import (
    build_stage_names,
    count_recorded_steps,
)


def test_wr03_step_count_does_not_double_count_navigation_aggregate() -> None:
    total, fields = count_recorded_steps(
        {
            "segments": [
                {
                    "steps": 48,
                    "pre_translate": {"steps": 48},
                    "rotate": {"steps": 0},
                },
                {
                    "pre_rotate": {"steps": 6},
                    "post_translate": {"steps": 4},
                },
            ]
        }
    )

    assert total == 58
    assert fields == {"executed_steps": 0, "release_settle_steps": 0, "steps": 58}


@pytest.mark.parametrize("level", ["lower", "upper"])
def test_wr03_stage_names_match_subtask_config(level: str) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wr03_subtasks.json")
    names = build_stage_names(level)

    matched = [match_subtask(name, config["stage_prefixes"]) for name in names]

    assert len(names) == 20
    assert matched == [
        *(["store_can"] * 7),
        *(["close_upper"] * 3),
        *(["close_lower"] * 4),
        *(["open_remembered"] * 4),
        *(["retrieve_can"] * 2),
    ]


@pytest.mark.parametrize(
    ("level", "open_index"),
    [("upper", 3), ("lower", 4)],
)
def test_wr03_only_retrieval_door_label_depends_on_memory(
    level: str, open_index: int
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wr03_subtasks.json")
    names = build_stage_names(level)
    record = {
        "task_id": "WR-03",
        "seed": 0,
        "stored_level": level,
        "stages": [
            {"name": name, "end_frame": index + 1}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=len(names))

    assert labeled["subtask_indices"] == [
        *([0] * 7),
        *([1] * 3),
        *([2] * 4),
        *([open_index] * 4),
        *([5] * 2),
    ]
    assert labeled["keyframes"] == [6, 9, 13, 14, 19]
