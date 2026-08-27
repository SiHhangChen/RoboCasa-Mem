#!/usr/bin/env python
"""Recover exact WA-03 stage boundaries from recorded planner summaries.

The native recorder persists action counts and the expert-plan metadata, but
not the original stage names.  This bridge reconstructs the 38 canonical
stages, verifies the recorded semantic diagnostics, and emits the boundary
contract consumed by ``apply_subtask_labels.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "WA-03"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 38
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})
COFFEE_IDS = frozenset({"coffee_A", "coffee_B", "coffee_C"})
SLOTS = frozenset({"left", "center", "right"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def count_recorded_steps(value: Any, *, path: str = "details") -> tuple[int, dict[str, int]]:
    """Count only recorder-owned action-step fields, including nested details."""
    total = 0
    fields = {key: 0 for key in sorted(STEP_KEYS)}

    def visit(node: Any, node_path: str) -> None:
        nonlocal total
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key)
                child_path = f"{node_path}.{key}"
                if key in STEP_KEYS:
                    _require(
                        isinstance(child, int) and not isinstance(child, bool) and child >= 0,
                        f"{child_path} must be a non-negative integer, got {child!r}",
                    )
                    total += int(child)
                    fields[key] += int(child)
                visit(child, child_path)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, child in enumerate(node):
                visit(child, f"{node_path}[{index}]")

    visit(value, path)
    return total, fields


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WA-03 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def _expert_metadata(episode: Mapping[str, Any], *, seed: int) -> Mapping[str, Any]:
    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing episode metadata")
    assert isinstance(metadata, Mapping)
    _require(metadata.get("task_id") == TASK_ID, f"seed {seed}: episode task mismatch")
    _require(metadata.get("seed") == seed, f"seed {seed}: episode seed mismatch")
    expert = metadata.get("expert_plan")
    _require(isinstance(expert, Mapping), f"seed {seed}: missing expert_plan metadata")
    assert isinstance(expert, Mapping)
    _require(expert.get("version") == EXPERT_PLAN_VERSION, f"seed {seed}: expert version mismatch")
    _require(expert.get("stage_count") == EXPECTED_STAGE_COUNT, f"seed {seed}: expert stage count mismatch")
    value = expert.get("metadata")
    _require(isinstance(value, Mapping), f"seed {seed}: missing expert_plan.metadata")
    assert isinstance(value, Mapping)
    return value


def build_stage_names(delivery_order: Sequence[str]) -> list[str]:
    order = [str(value) for value in delivery_order]
    _require(len(order) == 3 and set(order) == COFFEE_IDS, "delivery_order must contain coffee_A/B/C exactly once")
    first, second, third = order
    names = [
        "navigate_pick_sugar",
        "pick_sugar",
        "return_picked_sugar_to_regular_pose",
        "navigate_place_sugar",
        "place_sugar",
        "record_sugar_delivery",
        "return_after_place_sugar_to_regular_pose",
    ]
    for ordinal, coffee in (("first", first), ("second", second), ("third", third)):
        names.extend(
            [
                *([f"navigate_pick_{ordinal}_delivery_{coffee}"] if ordinal != "first" else []),
                f"verify_{ordinal}_pick_reachability_{coffee}",
                f"pick_{ordinal}_delivery_{coffee}",
                *([f"raise_{ordinal}_delivery_{coffee}_for_transport"] if ordinal != "first" else []),
                f"return_{ordinal}_delivery_{coffee}_to_regular_pose_for_transport",
                f"verify_{ordinal}_{coffee}_held_for_transport",
                f"navigate_deliver_{ordinal}_{coffee}_dining",
                f"place_{ordinal}_{coffee}_dining",
                f"verify_{ordinal}_{coffee}_on_dining",
                *([f"return_after_place_{ordinal}_{coffee}_dining_to_regular_pose"] if ordinal != "third" else []),
            ]
        )
        if ordinal == "third":
            names.append("verify_all_cups_ready_for_selection")
    names.extend([
        "pick_sugared_coffee_topdown",
        "verify_sugared_coffee_held_after_topdown_pick",
        "select_sugared_coffee",
    ])
    _require(len(names) == EXPECTED_STAGE_COUNT, f"reconstructed {len(names)} stages, expected {EXPECTED_STAGE_COUNT}")
    return names


def _sugared_slot(episode: Mapping[str, Any], planner: Mapping[str, Any], *, seed: int) -> str:
    expert = _expert_metadata(episode, seed=seed)
    sugared = str(expert.get("sugared_coffee_id", ""))
    _require(sugared in COFFEE_IDS, f"seed {seed}: invalid sugared coffee {sugared!r}")
    assignment = expert.get("final_slot_assignment")
    _require(isinstance(assignment, Mapping), f"seed {seed}: missing final_slot_assignment")
    assert isinstance(assignment, Mapping)
    assignment = {str(key): str(value) for key, value in assignment.items()}
    _require(set(assignment) == COFFEE_IDS and set(assignment.values()) == SLOTS, f"seed {seed}: invalid final slot assignment")
    planned = assignment[sugared]

    diagnostics = planner.get("final_diagnostics")
    _require(isinstance(diagnostics, Mapping), f"seed {seed}: missing final diagnostics")
    assert isinstance(diagnostics, Mapping)
    _require(diagnostics.get("success") is True, f"seed {seed}: final diagnostics failed")
    sugar = diagnostics.get("sugar")
    _require(isinstance(sugar, Mapping), f"seed {seed}: missing sugar diagnostics")
    assert isinstance(sugar, Mapping)
    _require(sugar.get("sugared_coffee_id") == sugared, f"seed {seed}: observed sugar cup mismatch")
    placement = diagnostics.get("dining_placement")
    _require(isinstance(placement, Mapping), f"seed {seed}: missing dining placement diagnostics")
    assert isinstance(placement, Mapping)
    occupancy = placement.get("occupancy")
    _require(isinstance(occupancy, Mapping), f"seed {seed}: missing dining occupancy")
    assert isinstance(occupancy, Mapping)
    observed = [slot for slot in SLOTS if sugared in occupancy.get(slot, [])]
    _require(len(observed) == 1, f"seed {seed}: sugared cup must occupy exactly one slot, found {observed}")
    _require(observed[0] == planned, f"seed {seed}: observed slot {observed[0]!r} != planned {planned!r}")
    selection = diagnostics.get("selection")
    _require(isinstance(selection, Mapping), f"seed {seed}: missing selection diagnostics")
    assert isinstance(selection, Mapping)
    _require(selection.get("correct") is True and selection.get("selected_coffee_id") == sugared, f"seed {seed}: invalid final selection")
    return planned


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = _episode_seed(episode_dir)
    episode = _read_json(episode_dir / "episode.json")
    planner = _read_json(episode_dir / "planner.json")
    _require(episode.get("accepted") is True, f"seed {seed}: episode is not accepted")
    _require(episode.get("planner_pass") is True, f"seed {seed}: planner did not pass")
    _require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task mismatch")
    _require(planner.get("expert_plan_version") == EXPERT_PLAN_VERSION, f"seed {seed}: planner version mismatch")
    _require(planner.get("pass") is True, f"seed {seed}: planner pass is false")
    expert = _expert_metadata(episode, seed=seed)
    order = expert.get("delivery_order")
    _require(isinstance(order, list), f"seed {seed}: delivery_order must be a list")
    assert isinstance(order, list)
    sugared = str(expert.get("sugared_coffee_id", ""))
    _require(str(expert.get("sugar_target_mode", "")) == "unmarked", f"seed {seed}: expected unmarked WA-03 data")
    _require(sugared == str(order[0]), f"seed {seed}: unmarked expert must sugar delivery_order[0]")
    stages = planner.get("rollout", {}).get("stages")
    _require(isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT, f"seed {seed}: expected {EXPECTED_STAGE_COUNT} planner stages")
    assert isinstance(stages, list)
    result = planner.get("episode_result")
    _require(isinstance(result, Mapping) and result.get("success") is True and result.get("stage_count") == EXPECTED_STAGE_COUNT, f"seed {seed}: invalid episode result")
    assert isinstance(result, Mapping)
    names = build_stage_names(order)
    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    fields = {key: 0 for key in sorted(STEP_KEYS)}
    for index, (stage, name) in enumerate(zip(stages, names)):
        _require(isinstance(stage, Mapping), f"seed {seed}: invalid stage {index}")
        assert isinstance(stage, Mapping)
        _require(stage.get("stage_index") == index and stage.get("success") is True, f"seed {seed}: invalid stage {index} result")
        details = stage.get("details")
        _require(isinstance(details, Mapping), f"seed {seed}: stage {index} missing details")
        assert isinstance(details, Mapping)
        steps, counts = count_recorded_steps(details, path=f"seed[{seed}].stage[{index}].details")
        start = cumulative
        cumulative += steps
        for key, value in counts.items():
            fields[key] += value
        output_stages.append({
            "index": index,
            "name": name,
            "kind": str(stage.get("stage_kind", "")),
            "primitive": str(details.get("primitive", "")),
            "steps": steps,
            "step_fields": counts,
            "start_frame": start,
            "end_frame": cumulative,
        })
    recorded = episode.get("step_count")
    _require(isinstance(recorded, int) and not isinstance(recorded, bool) and recorded > 0, f"seed {seed}: invalid step_count")
    _require(cumulative == recorded, f"seed {seed}: recovered {cumulative} != recorded {recorded}; fields={fields}")
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "delivery_order": [str(value) for value in order],
        "sugared_coffee_id": sugared,
        "sugared_coffee_slot": _sugared_slot(episode, planner, seed=seed),
        "total_frames": cumulative,
        "recorded_total_frames": recorded,
        "step_field_totals": fields,
        "source_episode_dir": episode_dir.name,
        "stages": output_stages,
    }


def collect_boundary_records(episodes_root: Path, *, expected_episodes: int) -> list[dict[str, Any]]:
    episode_dirs = sorted(path for path in episodes_root.glob("seed_*_attempt_00") if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name))
    if expected_episodes > 0:
        _require(len(episode_dirs) == expected_episodes, f"expected {expected_episodes} episodes, found {len(episode_dirs)}")
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    _require(len(seeds) == len(set(seeds)), "duplicate seeds discovered")
    if expected_episodes > 0:
        _require(seeds == list(range(expected_episodes)), f"expected contiguous seeds 0..{expected_episodes - 1}, got {seeds}")
    return records


def write_boundary_records(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=200)
    args = parser.parse_args()
    if args.expected_episodes < 0:
        parser.error("--expected-episodes must be non-negative")
    records = collect_boundary_records(args.episodes_root.resolve(), expected_episodes=args.expected_episodes)
    write_boundary_records(records, args.output)
    print(f"wrote {len(records)} WA-03 boundary records ({sum(int(r['total_frames']) for r in records)} frames) to {args.output}")


if __name__ == "__main__":
    main()
