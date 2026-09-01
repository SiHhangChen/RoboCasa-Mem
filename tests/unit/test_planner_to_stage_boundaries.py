from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.annotate.apply_subtask_labels import label_episode, load_config, match_subtask
from scripts.annotate.planner_to_stage_boundaries import (
    EXPERT_PLAN_VERSION,
    NAVIGATE_STAGE_INDICES,
    build_boundary_record,
    build_stage_names,
    collect_boundary_records,
    count_recorded_steps,
)


def test_count_recorded_steps_uses_exact_whitelist() -> None:
    total, fields = count_recorded_steps(
        {
            "steps": 4,
            "settle_steps": 100,
            "preshape_gripper_steps": 200,
            "nested": [
                {"execution": {"executed_steps": 7, "settle_steps": 300}},
                {"release": {"release_settle_steps": 5}},
            ],
        }
    )

    assert total == 16
    assert fields == {"executed_steps": 7, "release_settle_steps": 5, "steps": 4}


@pytest.mark.parametrize("value", [True, -1, 1.5, None])
def test_count_recorded_steps_rejects_invalid_step_values(value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer|must be non-negative"):
        count_recorded_steps({"steps": value})


def test_build_stage_names_matches_wa01_v2_template() -> None:
    names = build_stage_names("coffee_B", "coffee_A", "coffee_B")

    assert len(names) == 29
    assert names[0] == "navigate_pick_sugar"
    assert names[7] == "verify_first_pick_reachability_coffee_B"
    assert names[15] == "navigate_pick_second_delivery_coffee_A"
    assert names[24] == "return_after_place_second_coffee_A_dining_to_regular_pose"
    assert names[25] == "verify_cups_ready_for_selection"
    assert names[-1] == "select_sugared_coffee"


def test_reconstructed_names_match_current_subtask_config() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wa01_subtasks.json")
    names = build_stage_names("coffee_B", "coffee_A", "coffee_B")

    matched = [match_subtask(name, config["stage_prefixes"]) for name in names]

    assert matched == [
        *(["add_sugar"] * 7),
        *(["place_first"] * 8),
        *(["place_second"] * 10),
        *(["identify"] * 4),
    ]


@pytest.mark.parametrize(
    ("slot", "expected_index"),
    [("left", 3), ("right", 4)],
)
def test_final_stage_maps_to_fixed_dining_slot_label(slot: str, expected_index: int) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs" / "memer" / "wa01_subtasks.json")
    names = build_stage_names("coffee_B", "coffee_A", "coffee_B")
    record = {
        "task_id": "WA-01",
        "seed": 0,
        "sugared_coffee_slot": slot,
        "stages": [
            {"name": name, "end_frame": index + 1}
            for index, name in enumerate(names)
        ],
    }

    labeled = label_episode(record, config, recorded_total=len(names))

    assert labeled["subtask_indices"][-4:] == [expected_index] * 4
    assert labeled["keyframes"] == [6, 14, 24, 25]


def _write_episode(
    root: Path,
    *,
    seed: int,
    include_release_settle: bool = True,
    stage_count: int = 29,
) -> Path:
    episode_dir = root / f"seed_{seed:06d}_attempt_00"
    episode_dir.mkdir(parents=True)
    first, second = ("coffee_B", "coffee_A") if seed % 2 == 0 else ("coffee_A", "coffee_B")
    final_slot_assignment = {
        "coffee_A": "left" if seed % 2 == 0 else "right",
        "coffee_B": "right" if seed % 2 == 0 else "left",
    }
    first_slot = final_slot_assignment[first]
    second_slot = final_slot_assignment[second]

    stages = []
    base_total = 0
    for index in range(stage_count):
        details: dict[str, object] = {"execution": {"executed_steps": index + 1}}
        base_total += index + 1
        if include_release_settle and index in {4, 12, 22}:
            settle = {4: 5, 12: 10, 22: 10}[index]
            details["release_gripper"] = {"release_settle_steps": settle}
        stages.append(
            {
                "stage_index": index,
                "stage_kind": "navigate" if index in NAVIGATE_STAGE_INDICES else "operate",
                "success": True,
                "failure_code": None,
                "details": details,
            }
        )

    episode = {
        "accepted": True,
        "planner_pass": True,
        "step_count": base_total + 25,
        "metadata": {
            "task_id": "WA-01",
            "seed": seed,
            "expert_plan": {
                "version": EXPERT_PLAN_VERSION,
                "stage_count": stage_count,
                "metadata": {
                    "delivery_order": [first, second],
                    "sugared_coffee_id": first,
                    "final_slot_assignment": final_slot_assignment,
                },
            },
        },
    }
    planner = {
        "task_id": "WA-01",
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "pass": True,
        "episode_result": {
            "success": True,
            "stage_count": stage_count,
            "stages": [True] * stage_count,
        },
        "rollout": {"stages": stages},
        "final_diagnostics": {
            "sugar": {"sugared_coffee_id": first},
            "dining_placement": {
                "occupancy": {
                    first_slot: [first],
                    second_slot: [second],
                }
            },
        },
    }
    (episode_dir / "episode.json").write_text(json.dumps(episode), encoding="utf-8")
    (episode_dir / "planner.json").write_text(json.dumps(planner), encoding="utf-8")
    return episode_dir


def test_build_boundary_record_recovers_exact_recorded_total(tmp_path: Path) -> None:
    episode_dir = _write_episode(tmp_path, seed=0)

    record = build_boundary_record(episode_dir)

    assert record["seed"] == 0
    assert record["total_frames"] == record["recorded_total_frames"] == sum(range(1, 30)) + 25
    assert record["step_field_totals"] == {
        "executed_steps": sum(range(1, 30)),
        "release_settle_steps": 25,
        "steps": 0,
    }
    assert record["stages"][4]["steps"] == 10
    assert record["stages"][12]["steps"] == 23
    assert record["stages"][22]["steps"] == 33
    assert record["stages"][-1]["end_frame"] == record["total_frames"]


def test_build_boundary_record_accepts_legacy_28_stage_episode(tmp_path: Path) -> None:
    episode_dir = _write_episode(tmp_path, seed=0, stage_count=28)

    record = build_boundary_record(episode_dir)
    names = [stage["name"] for stage in record["stages"]]

    assert len(names) == 28
    assert names[24] == "verify_cups_ready_for_selection"
    assert not any(name.startswith("return_after_place_second_") for name in names)


def test_build_boundary_record_rejects_unaccounted_recorded_frames(tmp_path: Path) -> None:
    episode_dir = _write_episode(tmp_path, seed=0, include_release_settle=False)

    with pytest.raises(ValueError, match="recovered 435 stage frames != recorded 460"):
        build_boundary_record(episode_dir)


def test_collect_boundary_records_requires_contiguous_complete_seeds(tmp_path: Path) -> None:
    _write_episode(tmp_path, seed=0)
    _write_episode(tmp_path, seed=2)

    with pytest.raises(ValueError, match="expected contiguous seeds"):
        collect_boundary_records(tmp_path, expected_episodes=2)
