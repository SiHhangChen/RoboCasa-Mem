#!/usr/bin/env python
"""Recover exact WX-01 frame boundaries from recorded planner summaries.

WX-01 records six physical expert stages, while MemER exposes four semantic
stages.  This bridge counts the recorder's action-step fields with aggregate
awareness, verifies that the six stages exactly cover every recorded frame,
and emits the boundary contract consumed by ``apply_subtask_labels.py``.

The successful WX-01 collection contains 200 episodes whose source seeds are
not contiguous (failed source seeds were omitted before LeRobot conversion),
so this script validates episode count and uniqueness without requiring seeds
to be ``0..N-1``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "WX-01"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 6
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = ("executed_steps", "release_settle_steps", "steps")
HIGH_LEVEL_INSTRUCTION = (
    "Start with the sponge already grasped. Wipe the dirty dining-table spot, "
    "then place the cup onto the cleaned spot."
)
STAGE_NAMES = (
    "navigate_to_wipe_area_holding_sponge",
    "wipe_remembered_blocking_origin",
    "place_cleaning_tool_after_wipe",
    "return_after_post_wipe_sponge_place",
    "pick_displaced_blocking_object",
    "return_blocking_object_to_remembered_origin",
)
STAGE_PRIMITIVES = (
    "manual_base_path_navigate",
    "wx01_wipe_remembered_area",
    "wx01_place_at_accessible_target",
    "return_arm_to_regular_pose",
    "pick",
    "wx01_place_at_accessible_target",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def non_negative_int(value: Any, path: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{path} must be a non-negative integer, got {value!r}",
    )
    return int(value)


def episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WX-01 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def count_recorded_steps(
    value: Any, *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Count recorder step fields without double-counting nested diagnostics.

    Navigation and manipulation summaries sometimes expose an aggregate
    ``steps`` or ``executed_steps`` value together with nested diagnostic
    mappings that repeat the same count.  Once an aggregate is present on a
    mapping, descendants are diagnostics rather than additional actions.
    """

    total = 0
    fields = {key: 0 for key in STEP_KEYS}

    def visit(node: Any, node_path: str) -> None:
        nonlocal total
        if isinstance(node, Mapping):
            has_aggregate = "steps" in node or "executed_steps" in node
            for raw_key, child in node.items():
                key = str(raw_key)
                child_path = f"{node_path}.{key}"
                if key in STEP_KEYS:
                    amount = non_negative_int(child, child_path)
                    total += amount
                    fields[key] += amount
                if not has_aggregate and key not in STEP_KEYS:
                    visit(child, child_path)
            return
        if isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            for index, child in enumerate(node):
                visit(child, f"{node_path}[{index}]")

    visit(value, path)
    return total, fields


def validate_episode_contract(
    episode: Mapping[str, Any], planner: Mapping[str, Any], seed: int
) -> tuple[int, list[Mapping[str, Any]]]:
    require(episode.get("accepted") is True, f"seed {seed}: episode not accepted")
    require(episode.get("planner_pass") is True, f"seed {seed}: planner did not pass")
    recorded_total = non_negative_int(
        episode.get("step_count"), f"seed[{seed}].step_count"
    )
    require(recorded_total > 0, f"seed {seed}: empty episode")

    metadata = episode.get("metadata")
    require(isinstance(metadata, Mapping), f"seed {seed}: missing episode metadata")
    require(metadata.get("task_id") == TASK_ID, f"seed {seed}: task mismatch")
    require(metadata.get("seed") == seed, f"seed {seed}: metadata seed mismatch")
    expert = metadata.get("expert_plan")
    require(isinstance(expert, Mapping), f"seed {seed}: missing expert plan")
    require(
        expert.get("version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: episode expert-plan version mismatch",
    )
    require(
        expert.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: episode expert-plan stage count mismatch",
    )

    require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task mismatch")
    require(
        planner.get("expert_plan_version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: planner expert-plan version mismatch",
    )
    require(planner.get("pass") is True, f"seed {seed}: planner pass is false")
    result = planner.get("episode_result")
    require(isinstance(result, Mapping), f"seed {seed}: missing episode_result")
    require(
        result.get("success") is True
        and result.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: invalid episode_result",
    )
    rollout = planner.get("rollout")
    require(isinstance(rollout, Mapping), f"seed {seed}: missing rollout")
    stages = rollout.get("stages")
    require(
        isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expected {EXPECTED_STAGE_COUNT} rollout stages",
    )
    return recorded_total, stages


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = episode_seed(episode_dir)
    episode = read_json(episode_dir / "episode.json")
    planner = read_json(episode_dir / "planner.json")
    recorded_total, stages = validate_episode_contract(episode, planner, seed)

    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    field_totals = {key: 0 for key in STEP_KEYS}
    for index, (stage, expected_name, expected_primitive) in enumerate(
        zip(stages, STAGE_NAMES, STAGE_PRIMITIVES, strict=True)
    ):
        require(isinstance(stage, Mapping), f"seed {seed}: invalid stage {index}")
        require(
            stage.get("stage_index") == index and stage.get("success") is True,
            f"seed {seed}: stage {index} failed or has wrong index",
        )
        expected_kind = "navigate" if index == 0 else "operate"
        require(
            stage.get("stage_kind") == expected_kind,
            f"seed {seed}: stage {index} kind mismatch",
        )
        details = stage.get("details")
        require(isinstance(details, Mapping), f"seed {seed}: stage {index} missing details")
        require(
            details.get("primitive") == expected_primitive,
            f"seed {seed}: stage {index} primitive mismatch: "
            f"{details.get('primitive')!r} != {expected_primitive!r}",
        )
        if index in {3}:
            require(
                details.get("stage") == expected_name,
                f"seed {seed}: stage {index} diagnostic name mismatch",
            )

        steps, fields = count_recorded_steps(
            details, path=f"seed[{seed}].stage[{index}].details"
        )
        start_frame = cumulative
        cumulative += steps
        for key, amount in fields.items():
            field_totals[key] += amount
        output_stages.append(
            {
                "index": index,
                "name": expected_name,
                "kind": expected_kind,
                "primitive": expected_primitive,
                "steps": steps,
                "step_fields": fields,
                "start_frame": start_frame,
                "end_frame": cumulative,
            }
        )

    require(
        cumulative == recorded_total,
        f"seed {seed}: recovered {cumulative} stage frames != recorded "
        f"{recorded_total}; field totals={field_totals}",
    )

    # The semantic labels merge [0,1], [2,3], [4], and [5].  Store their last
    # frames explicitly so the LeRobot copy and SFT export share one boundary
    # contract instead of independently inferring it.
    keyframes = [
        output_stages[1]["end_frame"] - 1,
        output_stages[3]["end_frame"] - 1,
        output_stages[4]["end_frame"] - 1,
        output_stages[5]["end_frame"] - 1,
    ]
    require(
        all(0 <= int(frame) < recorded_total for frame in keyframes),
        f"seed {seed}: keyframe outside episode: {keyframes}",
    )

    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "high_level_instruction": HIGH_LEVEL_INSTRUCTION,
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "keyframes": keyframes,
        "stages": output_stages,
    }


def collect_boundary_records(
    episodes_root: Path, expected_episodes: int
) -> list[dict[str, Any]]:
    episode_dirs = sorted(
        path
        for path in episodes_root.glob("seed_*_attempt_00")
        if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name)
    )
    if expected_episodes > 0:
        require(
            len(episode_dirs) == expected_episodes,
            f"expected {expected_episodes} episodes, found {len(episode_dirs)}",
        )
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    require(len(seeds) == len(set(seeds)), "duplicate WX-01 seeds discovered")
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
    records = collect_boundary_records(
        args.episodes_root.resolve(), args.expected_episodes
    )
    write_boundary_records(records, args.output)
    print(
        f"wrote {len(records)} WX-01 boundary records "
        f"({sum(int(record['total_frames']) for record in records)} frames) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
