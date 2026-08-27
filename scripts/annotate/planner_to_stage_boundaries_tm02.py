#!/usr/bin/env python
"""Recover exact TM02 frame boundaries from recorded planner summaries.

TM02 has two legal routes. Both turn on the faucet, open the cabinet, retrieve
and place the can, and then complete faucet-off/cabinet-close in opposite
orders. This bridge reconstructs the 19 canonical stage names, validates the
recorded physical operations, and emits the boundary format consumed by
``apply_subtask_labels.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "TM02"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 19
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = ("executed_steps", "release_settle_steps", "steps")
FIELD_KEYS = (*STEP_KEYS, "sample_count")
ROUTES = frozenset({"L1", "L2"})
HIGH_LEVEL_INSTRUCTION = (
    "You have two tasks to complete. Task A: press the blue water button "
    "beside the sink to turn on the faucet and fill the pot already in the "
    "sink, then press the same blue button to turn off the faucet after the "
    "pot has filled for a while. Task B: go to the cabinet, open the sliding "
    "door, take out the can, place it in the marked area, and close the "
    "cabinet door. After both tasks are complete, press the red completion "
    "button beside the sink to indicate completion."
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


def count_recorded_steps(
    value: Any, *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Count recorder action fields without double-counting diagnostics."""
    total = 0
    fields = {key: 0 for key in FIELD_KEYS}

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


def count_stage_steps(
    details: Mapping[str, Any], *, seed: int, index: int
) -> tuple[int, dict[str, int]]:
    """Count one stage, handling close-door compatibility aliases.

    A close-door result retains every physical attempt and also mirrors the
    final attempt into top-level ``commands``/``stages`` fields. Only the
    attempt list represents executed actions; counting the aliases would add
    the final attempt twice.
    """
    if details.get("primitive") != "close_storage_door":
        return count_recorded_steps(
            details, path=f"seed[{seed}].stage[{index}].details"
        )
    attempts = details.get("physical_attempts")
    require(
        isinstance(attempts, list) and attempts,
        f"seed {seed}: close-door stage {index} lacks physical_attempts",
    )
    totals = {key: 0 for key in FIELD_KEYS}
    step_total = 0
    for attempt_index, attempt in enumerate(attempts):
        require(
            isinstance(attempt, Mapping),
            f"seed {seed}: invalid close-door attempt {attempt_index}",
        )
        amount, fields = count_recorded_steps(
            attempt,
            path=(
                f"seed[{seed}].stage[{index}].details"
                f".physical_attempts[{attempt_index}]"
            ),
        )
        step_total += amount
        for key, value in fields.items():
            totals[key] += value
    return step_total, totals


def episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected TM02 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def expert_metadata(episode: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    metadata = episode.get("metadata")
    require(isinstance(metadata, Mapping), f"seed {seed}: missing metadata")
    assert isinstance(metadata, Mapping)
    require(metadata.get("task_id") == TASK_ID, f"seed {seed}: task mismatch")
    require(metadata.get("seed") == seed, f"seed {seed}: seed mismatch")
    expert = metadata.get("expert_plan")
    require(isinstance(expert, Mapping), f"seed {seed}: missing expert_plan")
    assert isinstance(expert, Mapping)
    require(
        expert.get("version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: expert-plan version mismatch",
    )
    require(
        expert.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expert stage-count mismatch",
    )
    value = expert.get("metadata")
    require(isinstance(value, Mapping), f"seed {seed}: missing expert metadata")
    assert isinstance(value, Mapping)
    return value


def build_stage_names(route: str) -> list[str]:
    require(route in ROUTES, f"invalid TM02 route: {route!r}")
    a1 = [
        "navigate_sink_A1",
        "press_blue_water_button_on",
        "return_arm_after_A1",
    ]
    b1 = [
        "navigate_cabinet_door_B1",
        "open_cabinet_door_B1",
        "return_arm_after_cabinet_open",
        "navigate_can_retrieval_B1",
        "pick_can_from_cabinet_B1",
        "navigate_can_place_region_B1",
        "place_and_release_can_B1",
        "return_arm_after_B1",
    ]
    a2 = [
        "navigate_sink_A2",
        "press_blue_water_button_off",
        "return_arm_after_A2",
    ]
    b2 = [
        "navigate_cabinet_door_B2",
        "close_cabinet_door_B2",
        "return_arm_after_B2",
    ]
    tail = [*a2, *b2] if route == "L1" else [*b2, *a2]
    names = [
        *a1,
        *b1,
        *tail,
        "navigate_red_completion_button",
        "press_red_completion_button",
    ]
    require(len(names) == EXPECTED_STAGE_COUNT, "internal TM02 stage-count error")
    return names


def expected_stage_contract(name: str) -> tuple[str, str]:
    if name.startswith("navigate_"):
        return "navigate", "manual_base_path_navigate"
    if name.startswith("press_blue_water_button_"):
        return "operate", "operate_water_button"
    if name.startswith("return_arm_"):
        return "operate", "return_arm_to_regular_pose"
    if name == "open_cabinet_door_B1":
        return "operate", "open_storage_door"
    if name == "pick_can_from_cabinet_B1":
        return "operate", "retrieve_item_from_cabinet"
    if name == "place_and_release_can_B1":
        return "operate", "place_can_in_marked_region"
    if name == "close_cabinet_door_B2":
        return "operate", "close_storage_door"
    if name == "press_red_completion_button":
        return "operate", "physical_tm02_completion_button_press"
    raise ValueError(f"unknown TM02 stage name {name!r}")


def validate_stage_details(
    details: Mapping[str, Any], *, name: str, seed: int, index: int
) -> None:
    if name == "press_blue_water_button_on":
        require(details.get("mode") == "on", f"seed {seed}: stage {index} mode mismatch")
        require(
            details.get("temperature") == "cold",
            f"seed {seed}: stage {index} temperature mismatch",
        )
    elif name == "press_blue_water_button_off":
        require(details.get("mode") == "off", f"seed {seed}: stage {index} mode mismatch")
        require(
            details.get("temperature") == "cold",
            f"seed {seed}: stage {index} temperature mismatch",
        )
    elif name == "press_red_completion_button":
        require(
            details.get("button") == "completion_button",
            f"seed {seed}: stage {index} button mismatch",
        )


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = episode_seed(episode_dir)
    episode = read_json(episode_dir / "episode.json")
    planner = read_json(episode_dir / "planner.json")
    require(episode.get("accepted") is True, f"seed {seed}: episode not accepted")
    require(episode.get("planner_pass") is True, f"seed {seed}: planner failed")
    recorded_total = non_negative_int(
        episode.get("step_count"), f"seed[{seed}].step_count"
    )
    require(recorded_total > 0, f"seed {seed}: empty episode")
    expert = expert_metadata(episode, seed)
    route = str(expert.get("route", ""))
    names = build_stage_names(route)

    require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task mismatch")
    require(
        planner.get("expert_plan_version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: planner version mismatch",
    )
    require(planner.get("pass") is True, f"seed {seed}: planner pass is false")
    result = planner.get("episode_result")
    require(
        isinstance(result, Mapping)
        and result.get("success") is True
        and result.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: invalid episode result",
    )
    stages = planner.get("rollout", {}).get("stages")
    require(
        isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expected {EXPECTED_STAGE_COUNT} rollout stages",
    )
    assert isinstance(stages, list)

    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    field_totals = {key: 0 for key in FIELD_KEYS}
    for index, (stage, name) in enumerate(zip(stages, names, strict=True)):
        require(isinstance(stage, Mapping), f"seed {seed}: invalid stage {index}")
        assert isinstance(stage, Mapping)
        require(
            stage.get("stage_index") == index and stage.get("success") is True,
            f"seed {seed}: stage {index} failed or has wrong index",
        )
        details = stage.get("details")
        require(isinstance(details, Mapping), f"seed {seed}: stage {index} missing details")
        assert isinstance(details, Mapping)
        kind, primitive = expected_stage_contract(name)
        require(stage.get("stage_kind") == kind, f"seed {seed}: stage {index} kind mismatch")
        require(details.get("primitive") == primitive, f"seed {seed}: stage {index} primitive mismatch")
        validate_stage_details(details, name=name, seed=seed, index=index)
        steps, fields = count_stage_steps(details, seed=seed, index=index)
        start = cumulative
        cumulative += steps
        for key, amount in fields.items():
            field_totals[key] += amount
        output_stages.append(
            {
                "index": index,
                "name": name,
                "kind": kind,
                "primitive": primitive,
                "steps": steps,
                "step_fields": fields,
                "start_frame": start,
                "end_frame": cumulative,
            }
        )
    require(
        cumulative == recorded_total,
        f"seed {seed}: recovered {cumulative} frames != recorded {recorded_total}; "
        f"field totals={field_totals}",
    )
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "high_level_instruction": HIGH_LEVEL_INSTRUCTION,
        "route": route,
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "stages": output_stages,
    }


def collect_boundary_records(
    episodes_roots: Sequence[Path], expected_episodes: int
) -> list[dict[str, Any]]:
    episode_dirs = sorted(
        (
            path
            for root in episodes_roots
            for path in root.glob("seed_*_attempt_00")
            if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name)
        ),
        key=episode_seed,
    )
    if expected_episodes > 0:
        require(
            len(episode_dirs) == expected_episodes,
            f"expected {expected_episodes} episodes, found {len(episode_dirs)}",
        )
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    require(len(seeds) == len(set(seeds)), "duplicate TM02 seeds discovered")
    if expected_episodes > 0:
        require(
            seeds == list(range(expected_episodes)),
            f"expected contiguous seeds 0..{expected_episodes - 1}",
        )
    return records


def write_boundary_records(
    records: Sequence[Mapping[str, Any]], output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=200)
    args = parser.parse_args()
    if args.expected_episodes < 0:
        parser.error("--expected-episodes must be non-negative")
    records = collect_boundary_records(
        [root.resolve() for root in args.episodes_root], args.expected_episodes
    )
    write_boundary_records(records, args.output)
    route_counts = {
        route: sum(record["route"] == route for record in records)
        for route in sorted(ROUTES)
    }
    print(
        f"wrote {len(records)} TM02 boundary records "
        f"({sum(int(record['total_frames']) for record in records)} frames; "
        f"routes={route_counts}) to {args.output}"
    )


if __name__ == "__main__":
    main()
