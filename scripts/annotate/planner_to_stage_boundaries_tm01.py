#!/usr/bin/env python
"""Recover exact TM01 frame boundaries from recorded planner summaries.

TM01 has two legal expert routes and two drawer orders. This bridge rebuilds
the 18 recorded stage names from the frozen episode metadata, verifies every
physical planner stage, and emits the boundary contract consumed by
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


TASK_ID = "TM01"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 18
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = ("executed_steps", "release_settle_steps", "steps")
FIELD_KEYS = (*STEP_KEYS, "sample_count")
DRAWERS = frozenset({"drawer_left", "drawer_right"})
ROUTES = frozenset({"L1", "L2"})
HIGH_LEVEL_INSTRUCTION = (
    "You have two tasks to complete. Task A: use the stove's control knob to "
    "turn on the heating function and heat the water, then turn off the heating "
    "after the water has heated for a while. Task B: go to the other side of "
    "the workspace and inspect both drawers, opening each drawer, looking "
    "inside, and closing it. After both tasks are complete, press either green "
    "button at the two ends to indicate completion."
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
    """Count recorder action fields without double-counting nested diagnostics."""
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


def episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected TM01 episode directory: {episode_dir.name!r}")
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


def drawer_unit(drawer: str, suffix: str) -> list[str]:
    return [
        f"navigate_{drawer}_{suffix}",
        f"open_{drawer}_{suffix}",
        f"inspect_{drawer}_{suffix}",
        f"close_{drawer}_{suffix}",
        f"return_arm_after_{suffix}",
    ]


def build_stage_names(
    route: str, drawer_order: Sequence[str]
) -> tuple[list[str], str]:
    drawers = [str(value) for value in drawer_order]
    require(
        len(drawers) == 2 and set(drawers) == DRAWERS,
        f"invalid drawer order: {drawers}",
    )
    require(route in ROUTES, f"invalid route: {route!r}")
    a1 = ["navigate_stove_A1", "turn_native_stove_knob_on", "return_arm_after_A1"]
    a2 = ["navigate_stove_A2", "turn_native_stove_knob_off", "return_arm_after_A2"]
    b1 = drawer_unit(drawers[0], "B1")
    b2 = drawer_unit(drawers[1], "B2")
    side = "drawer_side" if route == "L1" else "stove_side"
    tail = [*a2, *b2] if route == "L1" else [*b2, *a2]
    names = [
        *a1,
        *b1,
        *tail,
        f"navigate_{side}_completion_button",
        f"press_{side}_completion_button",
    ]
    require(len(names) == EXPECTED_STAGE_COUNT, "internal TM01 stage-count error")
    return names, side


def expected_stage_contract(name: str) -> tuple[str, str, str | None]:
    if name.startswith("navigate_"):
        return "navigate", "manual_base_path_navigate", None
    if name == "turn_native_stove_knob_on":
        return "operate", "pinch_and_rotate_native_knob", name
    if name == "turn_native_stove_knob_off":
        return "operate", "pinch_and_rotate_native_knob", name
    if name.startswith("return_arm_after_A"):
        return "operate", "return_arm_to_regular_pose", None
    if name.startswith("open_drawer_"):
        return "operate", "physical_drawer_pull", "open_drawer_physically"
    if name.startswith("inspect_drawer_"):
        return "operate", "physical_drawer_inspection_dwell", "inspect_drawer"
    if name.startswith("close_drawer_"):
        return "operate", "physical_drawer_push_close", "close_drawer_physically"
    if name.startswith("return_arm_after_B"):
        return "operate", "drawer_post_close_retreat_and_return", None
    if name.startswith("press_"):
        return "operate", "physical_tm01_completion_button_press", None
    raise ValueError(f"unknown TM01 stage name {name!r}")


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
    drawer_order = expert.get("drawer_order")
    require(isinstance(drawer_order, list), f"seed {seed}: missing drawer_order")
    assert isinstance(drawer_order, list)
    names, completion_side = build_stage_names(route, drawer_order)

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
        kind, primitive, operation = expected_stage_contract(name)
        require(stage.get("stage_kind") == kind, f"seed {seed}: stage {index} kind mismatch")
        require(details.get("primitive") == primitive, f"seed {seed}: stage {index} primitive mismatch")
        if operation is not None:
            require(details.get("operation") == operation, f"seed {seed}: stage {index} operation mismatch")
        if name.startswith("press_"):
            require(
                details.get("button") == f"completion_button_{completion_side}",
                f"seed {seed}: completion button mismatch",
            )
        if primitive in {"physical_drawer_pull", "physical_drawer_push_close"}:
            trace = details.get("trace_summary")
            require(
                isinstance(trace, Mapping),
                f"seed {seed}: stage {index} missing drawer trace summary",
            )
            assert isinstance(trace, Mapping)
            steps = non_negative_int(
                trace.get("sample_count"),
                f"seed[{seed}].stage[{index}].details.trace_summary.sample_count",
            )
            fields = {key: 0 for key in FIELD_KEYS}
            fields["sample_count"] = steps
        else:
            steps, base_fields = count_recorded_steps(
                details, path=f"seed[{seed}].stage[{index}].details"
            )
            fields = {key: int(base_fields.get(key, 0)) for key in FIELD_KEYS}
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
        "drawer_order": [str(value) for value in drawer_order],
        "first_drawer": str(drawer_order[0]),
        "remaining_drawer": str(drawer_order[1]),
        "completion_button_side": completion_side,
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
    require(len(seeds) == len(set(seeds)), "duplicate TM01 seeds discovered")
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
    print(
        f"wrote {len(records)} TM01 boundary records "
        f"({sum(int(record['total_frames']) for record in records)} frames) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
