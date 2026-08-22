from __future__ import annotations

import json
from pathlib import Path

from scripts.annotate.apply_subtask_labels import (
    label_episode,
    load_config,
    load_seed_registry,
    match_subtask,
)
from scripts.annotate.planner_to_stage_boundaries_ts01 import build_stage_names


def test_ts01_stage_names_repeat_one_sugar_subtask_then_button() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "ts01_subtasks.json")
    names = build_stage_names(["sugar2", "sugar0", "sugar4"])

    matched = [match_subtask(name, config["stage_prefixes"]) for name in names]

    assert len(names) == 20
    assert matched == [*(['add_sugar'] * 18), *(['press_button'] * 2)]


def test_ts01_keeps_two_labels_and_explicit_cycle_keyframes() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "ts01_subtasks.json")
    names = build_stage_names(["sugar2", "sugar0", "sugar4"])
    # Give every reconstructed planner stage two frames. A complete sugar
    # cycle is six stages, so its final frames are 11, 23, and 35. The button
    # subtask begins at frame 36.
    record = {
        "task_id": "TS01",
        "seed": 7,
        "keyframes": [11, 23, 35, 36],
        "stages": [
            {"name": name, "end_frame": (index + 1) * 2}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=40)

    assert labeled["subtask_order"] == [
        "add a sugar cube to the hot chocolate",
        "press the completion button",
    ]
    assert labeled["subtask_indices"] == [0] * 36 + [1] * 4
    assert labeled["keyframes"] == [11, 23, 35, 36]


def test_seed_registry_maps_non_contiguous_source_seeds(tmp_path: Path) -> None:
    registry = tmp_path / "seeds.jsonl"
    registry.write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 10, "seed": 12}),
                json.dumps({"episode_index": 199, "seed": 202}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_seed_registry(registry) == {12: 10, 202: 199}
