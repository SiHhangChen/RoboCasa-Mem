#!/usr/bin/env python
"""Recover exact WR01 stage boundaries from recorded planner summaries.

WR01 stores one colored can in each of two fixed cabinet levels, closes both
doors, and then retrieves the requested can.  The seed-selected storage
assignment is an expert demonstration outcome, not policy-visible input.  This
bridge preserves that assignment only to validate the rollout and to label the
remembered upper/lower retrieval-door decision.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "WR-01"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 20
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})
NAVIGATE_STAGE_INDICES = frozenset({0, 1, 4, 7, 10, 14, 17})
OPTIONAL_NAVIGATION_PRIMITIVE_INDICES = frozenset({0, 7})
ITEM_IDS = frozenset({"red_can", "blue_can"})
LEVELS = ("lower", "upper")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def count_recorded_steps(
    value: Any, *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Count recorder-defined action fields without nested aggregate duplication."""
    total = 0
    fields = {key: 0 for key in sorted(STEP_KEYS)}

    def visit(node: Any, node_path: str) -> None:
        nonlocal total
        if isinstance(node, Mapping):
            has_aggregate = "steps" in node or "executed_steps" in node
            for raw_key, child in node.items():
                key = str(raw_key)
                child_path = f"{node_path}.{key}"
                if key in STEP_KEYS:
                    if isinstance(child, bool) or not isinstance(child, int) or child < 0:
                        raise ValueError(
                            f"{child_path} must be a non-negative integer, got {child!r}"
                        )
                    total += child
                    fields[key] += child
                elif not has_aggregate:
                    visit(child, child_path)
            return
        if isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            for index, child in enumerate(node):
                visit(child, f"{node_path}[{index}]")

    visit(value, path)
    return total, fields


def _validate_assignment(value: Any, *, path: str) -> dict[str, str]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    assert isinstance(value, Mapping)
    assignment = {level: str(value.get(level, "")) for level in LEVELS}
    _require(
        set(assignment.values()) == ITEM_IDS,
        f"{path} must assign the red and blue cans once: {assignment}",
    )
    return assignment


def build_stage_names(
    item_by_level: Mapping[str, str],
    target_level: str,
    close_order: Sequence[str] = ("lower", "upper"),
) -> list[str]:
    _validate_assignment(item_by_level, path="item_by_level")
    _require(target_level in LEVELS, f"invalid WR01 target level: {target_level!r}")
    order = tuple(str(value) for value in close_order)
    _require(len(order) == 2 and set(order) == set(LEVELS), "invalid WR01 close order")
    names = ["observe_open_cabinets"]
    for index, level in enumerate(order, start=1):
        names.extend([f"navigate_to_door_{index}", f"close_door_{index}", f"return_after_door_{index}"])
    names.extend([
        "navigate_to_neutral_reset",
        "canonical_memory_reset",
        "latch_memory_handoff",
        "navigate_to_remembered_cabinet",
        "open_remembered_target_cabinet",
        "verify_remembered_target_cabinet",
        "return_after_opening_target_cabinet",
        "navigate_to_target_can",
        "pick_requested_colored_can",
        "return_with_requested_can",
        "navigate_to_initial_table",
        "place_requested_can_on_table",
        "verify_final_table_placement",
    ])
    _require(len(names) == EXPECTED_STAGE_COUNT, "WR01 stage reconstruction drifted")
    return names


def _expected_primitives() -> list[str]:
    return [
        "manual_base_path_navigate", "manual_base_path_navigate", "close_storage_door",
        "return_arm_to_regular_pose", "manual_base_path_navigate", "close_storage_door",
        "return_arm_to_regular_pose", "manual_base_path_navigate", "canonical_memory_reset",
        "latch_storage_hidden", "manual_base_path_navigate", "open_storage_door",
        "verify_retrieval_door", "return_arm_to_regular_pose", "manual_base_path_navigate",
        "retrieve_item_from_cabinet", "return_arm_to_regular_pose", "manual_base_path_navigate",
        "place_on_table", "verify_final_placement",
    ]


def _validate_stage_primitive(
    *, seed: int, index: int, details: Mapping[str, Any], expected: str
) -> None:
    actual = details.get("primitive")
    if actual is None and index in OPTIONAL_NAVIGATION_PRIMITIVE_INDICES:
        return
    _require(actual == expected, f"seed {seed}: stage {index} primitive mismatch")


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WR01 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = _episode_seed(episode_dir)
    episode = _read_json(episode_dir / "episode.json")
    planner = _read_json(episode_dir / "planner.json")
    _require(episode.get("accepted") is True, f"seed {seed}: episode is not accepted")
    _require(episode.get("planner_pass") is True, f"seed {seed}: planner did not pass")
    _require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task mismatch")
    _require(planner.get("expert_plan_version") == EXPERT_PLAN_VERSION, f"seed {seed}: planner version mismatch")
    _require(planner.get("pass") is True, f"seed {seed}: planner pass is false")

    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing episode metadata")
    assert isinstance(metadata, Mapping)
    _require(metadata.get("task_id") == TASK_ID, f"seed {seed}: episode task mismatch")
    _require(metadata.get("seed") == seed, f"seed {seed}: episode seed mismatch")
    resolved = metadata.get("resolved_episode")
    expert = metadata.get("expert_plan")
    _require(isinstance(resolved, Mapping), f"seed {seed}: missing resolved episode")
    _require(isinstance(expert, Mapping), f"seed {seed}: missing expert plan")
    assert isinstance(resolved, Mapping) and isinstance(expert, Mapping)
    facts = resolved.get("facts")
    expert_meta = expert.get("metadata")
    _require(isinstance(facts, Mapping), f"seed {seed}: missing resolved facts")
    _require(isinstance(expert_meta, Mapping), f"seed {seed}: missing expert metadata")
    assert isinstance(facts, Mapping) and isinstance(expert_meta, Mapping)
    assignment = _validate_assignment(facts.get("item_by_level"), path=f"seed {seed} item_by_level")
    target_item = str(facts.get("target_item_id", ""))
    target_level = str(facts.get("target_level", ""))
    _require(target_item in ITEM_IDS, f"seed {seed}: invalid target item")
    _require(target_level in LEVELS, f"seed {seed}: invalid target level")
    _require(assignment[target_level] == target_item, f"seed {seed}: target relation mismatch")
    close_order = tuple(str(value) for value in facts.get("close_order", ("lower", "upper")))
    _require(len(close_order) == 2 and set(close_order) == set(LEVELS), f"seed {seed}: invalid close order")
    _require(tuple(expert_meta.get("close_order", close_order)) == close_order, f"seed {seed}: expert close order mismatch")
    _require(expert_meta.get("target_item_id") == target_item and expert_meta.get("target_level") == target_level, f"seed {seed}: expert target mismatch")

    diagnostics = planner.get("final_diagnostics")
    _require(isinstance(diagnostics, Mapping), f"seed {seed}: missing diagnostics")
    assert isinstance(diagnostics, Mapping)
    storage = diagnostics.get("storage")
    retrieval = diagnostics.get("retrieval")
    _require(isinstance(storage, Mapping), f"seed {seed}: missing storage diagnostics")
    _require(isinstance(retrieval, Mapping), f"seed {seed}: missing retrieval diagnostics")
    assert isinstance(storage, Mapping) and isinstance(retrieval, Mapping)
    expected_by_item = {item: level for level, item in assignment.items()}
    _require(
        diagnostics.get("success") is True
        and diagnostics.get("terminal") is True
        and storage.get("storage_hidden") is True
        and storage.get("stored_levels_by_item") == expected_by_item
        and retrieval.get("opened_level") == target_level
        and retrieval.get("retrieved_item_id") == target_item
        and retrieval.get("wrong_level_opened") is False,
        f"seed {seed}: WR01 memory protocol validation failed",
    )

    result = planner.get("episode_result")
    rollout = planner.get("rollout")
    _require(isinstance(result, Mapping), f"seed {seed}: missing episode result")
    _require(isinstance(rollout, Mapping), f"seed {seed}: missing rollout")
    assert isinstance(result, Mapping) and isinstance(rollout, Mapping)
    _require(result.get("success") is True and result.get("stage_count") == EXPECTED_STAGE_COUNT, f"seed {seed}: invalid episode result")
    stages = rollout.get("stages")
    _require(isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT, f"seed {seed}: expected {EXPECTED_STAGE_COUNT} stages")
    assert isinstance(stages, list)
    recorded_total = episode.get("step_count")
    _require(isinstance(recorded_total, int) and not isinstance(recorded_total, bool) and recorded_total > 0, f"seed {seed}: invalid frame count")
    assert isinstance(recorded_total, int)

    stage_names = build_stage_names(assignment, target_level, close_order)
    primitives = _expected_primitives()
    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    field_totals = {key: 0 for key in sorted(STEP_KEYS)}
    for index, (stage, name, primitive) in enumerate(zip(stages, stage_names, primitives)):
        _require(isinstance(stage, Mapping), f"seed {seed}: invalid stage {index}")
        assert isinstance(stage, Mapping)
        expected_kind = "navigate" if index in NAVIGATE_STAGE_INDICES else "operate"
        _require(stage.get("stage_index") == index and stage.get("stage_kind") == expected_kind and stage.get("success") is True, f"seed {seed}: stage {index} contract mismatch")
        details = stage.get("details")
        _require(isinstance(details, Mapping), f"seed {seed}: stage {index} missing details")
        assert isinstance(details, Mapping)
        _validate_stage_primitive(
            seed=seed,
            index=index,
            details=details,
            expected=primitive,
        )
        steps, fields = count_recorded_steps(details, path=f"seed[{seed}].stage[{index}].details")
        start = cumulative
        cumulative += steps
        for key, value in fields.items():
            field_totals[key] += value
        output_stages.append({"index": index, "name": name, "kind": expected_kind, "primitive": primitive, "steps": steps, "step_fields": fields, "start_frame": start, "end_frame": cumulative})
    _require(cumulative == recorded_total, f"seed {seed}: recovered {cumulative} frames != recorded {recorded_total}; fields={field_totals}")

    instruction = str(metadata.get("policy_prompt", ""))
    _require(instruction, f"seed {seed}: missing policy prompt")
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "item_by_level": assignment,
        "close_order": list(close_order),
        "target_item_id": target_item,
        "target_level": target_level,
        "high_level_instruction": instruction,
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "stages": output_stages,
    }


def collect_boundary_records(episodes_root: Path, *, expected_episodes: int) -> list[dict[str, Any]]:
    episode_dirs = sorted(path for path in episodes_root.glob("seed_*_attempt_00") if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name))
    if expected_episodes > 0:
        _require(len(episode_dirs) == expected_episodes, f"expected {expected_episodes} WR01 episodes, found {len(episode_dirs)}")
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    _require(len(seeds) == len(set(seeds)), "duplicate WR01 seeds discovered")
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
    tasks = {str(record["high_level_instruction"]) for record in records}
    print(f"wrote {len(records)} WR01 boundary records ({sum(int(record['total_frames']) for record in records)} frames, {len(tasks)} task plans) to {args.output}")


if __name__ == "__main__":
    main()
