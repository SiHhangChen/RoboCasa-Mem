#!/usr/bin/env python
"""Recover exact TS02 stage boundaries and search-memory events.

TS02 samples a drawer search order independently from the hidden red-cube
location. This bridge reconstructs the variable-length expert plan and keeps
that sampled order only in the recorded stages. The policy-visible instruction
remains the original target-independent search task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "TS02"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
FIXTURE_DISPLAY_NAMES = {
    "cabinet_slot_0": "left",
    "cabinet_slot_1": "center",
    "cabinet_slot_2": "right",
}
TASK_INSTRUCTION = (
    "Search the three drawers for the red cube. Close every drawer after "
    "looking inside. Once you find the cube and close its drawer, press the "
    "completion button once for each drawer you inspected."
)
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})
DRAWER_TRACE_PRIMITIVES = frozenset(
    {"physical_drawer_pull", "physical_drawer_push_close"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def count_recorded_steps(
    value: Any, *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Count recorder-owned action steps without double-counting aggregates."""
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
                    if (
                        isinstance(child, bool)
                        or not isinstance(child, int)
                        or child < 0
                    ):
                        raise ValueError(
                            f"{child_path} must be a non-negative integer"
                        )
                    total += child
                    fields[key] += child
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
    details: Mapping[str, Any], *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Recover one TS02 stage's exact recorded action count.

    Physical drawer operations store the authoritative executed-frame count in
    ``actuation.trace_summary.sample_count``.  Button operations also carry a
    trace sample count, but it includes observer-only samples, so their exact
    count continues to come from ``steps``/``executed_steps`` fields.
    """
    actuation = details.get("actuation")
    if isinstance(actuation, Mapping) and actuation.get("primitive") in DRAWER_TRACE_PRIMITIVES:
        trace = actuation.get("trace_summary")
        _require(isinstance(trace, Mapping), f"{path}: missing drawer trace summary")
        assert isinstance(trace, Mapping)
        sample_count = trace.get("sample_count")
        _require(
            isinstance(sample_count, int)
            and not isinstance(sample_count, bool)
            and sample_count >= 0,
            f"{path}.actuation.trace_summary.sample_count is invalid",
        )
        return int(sample_count), {
            "executed_steps": 0,
            "release_settle_steps": 0,
            "steps": 0,
            "trace_samples": int(sample_count),
        }
    steps, fields = count_recorded_steps(details, path=path)
    return steps, {**fields, "trace_samples": 0}


def build_stage_names(
    fixture_order: Sequence[str], search_count: int
) -> list[str]:
    _require(search_count in {1, 2, 3}, f"invalid TS02 search count {search_count}")
    full_order = tuple(str(value) for value in fixture_order)
    _require(
        len(full_order) == 3
        and set(full_order) == set(FIXTURE_DISPLAY_NAMES),
        f"invalid TS02 fixture order {full_order!r}",
    )
    names: list[str] = []
    for fixture in full_order[:search_count]:
        names.extend(
            [
                f"navigate_{fixture}",
                f"open_{fixture}",
                f"inspect_{fixture}",
                f"close_{fixture}",
                f"return_after_{fixture}",
            ]
        )
    names.append("navigate_completion_button")
    for press_index in range(1, search_count + 1):
        names.append(
            f"press_completion_button_{press_index}_of_{search_count}"
        )
    return names


def build_search_instruction() -> str:
    return TASK_INSTRUCTION


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected TS02 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = _episode_seed(episode_dir)
    episode = _read_json(episode_dir / "episode.json")
    planner = _read_json(episode_dir / "planner.json")
    _require(episode.get("accepted") is True, f"seed {seed}: episode is not accepted")
    _require(episode.get("planner_pass") is True, f"seed {seed}: planner did not pass")
    _require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task mismatch")
    _require(
        planner.get("expert_plan_version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: planner version mismatch",
    )
    _require(planner.get("pass") is True, f"seed {seed}: planner pass is false")

    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing episode metadata")
    assert isinstance(metadata, Mapping)
    _require(metadata.get("task_id") == TASK_ID, f"seed {seed}: episode task mismatch")
    _require(metadata.get("seed") == seed, f"seed {seed}: episode seed mismatch")
    expert = metadata.get("expert_plan")
    resolved = metadata.get("resolved_episode")
    _require(isinstance(expert, Mapping), f"seed {seed}: missing expert plan")
    _require(isinstance(resolved, Mapping), f"seed {seed}: missing resolved episode")
    assert isinstance(expert, Mapping)
    assert isinstance(resolved, Mapping)
    _require(expert.get("version") == EXPERT_PLAN_VERSION, f"seed {seed}: expert version mismatch")
    expert_meta = expert.get("metadata")
    facts = resolved.get("facts")
    _require(isinstance(expert_meta, Mapping), f"seed {seed}: missing expert metadata")
    _require(isinstance(facts, Mapping), f"seed {seed}: missing resolved facts")
    assert isinstance(expert_meta, Mapping)
    assert isinstance(facts, Mapping)

    full_order = tuple(str(value) for value in expert_meta.get("full_fixture_order", []))
    _require(
        list(full_order) == list(facts.get("demo_fixture_order", [])),
        f"seed {seed}: expert and resolved search orders differ",
    )
    target_fixture = str(expert_meta.get("target_fixture", ""))
    _require(target_fixture == facts.get("target_fixture"), f"seed {seed}: target mismatch")
    search_count = int(expert_meta.get("search_count", 0))
    _require(
        search_count == full_order.index(target_fixture) + 1,
        f"seed {seed}: search count is inconsistent with target position",
    )
    fixture_order = tuple(str(value) for value in expert_meta.get("fixture_order", []))
    _require(
        fixture_order == full_order[:search_count],
        f"seed {seed}: inspected fixture prefix mismatch",
    )

    diagnostics = planner.get("final_diagnostics")
    _require(isinstance(diagnostics, Mapping), f"seed {seed}: missing final diagnostics")
    assert isinstance(diagnostics, Mapping)
    _require(diagnostics.get("success") is True, f"seed {seed}: final success is false")
    _require(diagnostics.get("target_found") is True, f"seed {seed}: target not found")
    _require(diagnostics.get("target_fixture") == target_fixture, f"seed {seed}: observed target mismatch")
    _require(
        list(diagnostics.get("completed_fixture_history", [])) == list(fixture_order),
        f"seed {seed}: completed fixture history mismatch",
    )
    _require(
        diagnostics.get("required_button_presses") == search_count
        and diagnostics.get("completion_button_press_count") == search_count,
        f"seed {seed}: completion-button count mismatch",
    )

    stages = planner.get("rollout", {}).get("stages")
    stage_names = build_stage_names(full_order, search_count)
    _require(
        isinstance(stages, list) and len(stages) == len(stage_names),
        f"seed {seed}: expected {len(stage_names)} rollout stages",
    )
    assert isinstance(stages, list)
    _require(
        expert.get("stage_count") == len(stage_names),
        f"seed {seed}: expert stage count mismatch",
    )
    episode_result = planner.get("episode_result")
    _require(isinstance(episode_result, Mapping), f"seed {seed}: missing episode result")
    assert isinstance(episode_result, Mapping)
    _require(
        episode_result.get("success") is True
        and episode_result.get("stage_count") == len(stage_names),
        f"seed {seed}: planner episode result mismatch",
    )

    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    keyframes: list[int] = []
    field_totals = {
        "executed_steps": 0,
        "release_settle_steps": 0,
        "steps": 0,
        "trace_samples": 0,
    }
    for index, (stage, name) in enumerate(zip(stages, stage_names)):
        _require(isinstance(stage, Mapping), f"seed {seed}: invalid stage {index}")
        assert isinstance(stage, Mapping)
        _require(stage.get("stage_index") == index, f"seed {seed}: stage index mismatch")
        _require(stage.get("success") is True, f"seed {seed}: stage {index} failed")
        expected_kind = "navigate" if name.startswith("navigate_") else "operate"
        _require(stage.get("stage_kind") == expected_kind, f"seed {seed}: stage kind mismatch at {index}")
        details = stage.get("details")
        _require(isinstance(details, Mapping), f"seed {seed}: stage {index} lacks details")
        assert isinstance(details, Mapping)
        steps, fields = count_stage_steps(
            details, path=f"seed[{seed}].stage[{index}].details"
        )
        start_frame = cumulative
        cumulative += steps
        for key, value in fields.items():
            field_totals[key] += value
        output_stages.append(
            {
                "index": index,
                "name": name,
                "kind": expected_kind,
                "steps": steps,
                "step_fields": fields,
                "start_frame": start_frame,
                "end_frame": cumulative,
            }
        )
        if name.startswith("return_after_cabinet_slot_"):
            _require(steps > 0, f"seed {seed}: return stage {name!r} is empty")
            keyframes.append(cumulative - 1)
        if name == "navigate_completion_button":
            keyframes.append(start_frame)

    total_frames = episode.get("step_count")
    _require(
        isinstance(total_frames, int)
        and not isinstance(total_frames, bool)
        and total_frames > 0,
        f"seed {seed}: invalid recorded frame count",
    )
    _require(
        cumulative == total_frames,
        f"seed {seed}: recovered {cumulative} != recorded {total_frames}; fields={field_totals}",
    )
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "full_fixture_order": list(full_order),
        "fixture_order": list(fixture_order),
        "target_fixture": target_fixture,
        "search_count": search_count,
        "high_level_instruction": build_search_instruction(),
        "total_frames": cumulative,
        "recorded_total_frames": total_frames,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "keyframes": sorted(set(keyframes)),
        "stages": output_stages,
    }


def collect_boundary_records(
    episodes_root: Path, *, expected_episodes: int
) -> list[dict[str, Any]]:
    episode_dirs = [
        path
        for path in episodes_root.glob("seed_*_attempt_00")
        if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name)
    ]
    if not episode_dirs:
        episode_dirs = [
            path
            for path in episodes_root.glob("*/train/*/seed_*_attempt_00")
            if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name)
        ]
    episode_dirs = sorted(episode_dirs, key=_episode_seed)
    if expected_episodes > 0:
        _require(
            len(episode_dirs) == expected_episodes,
            f"expected {expected_episodes} episodes, found {len(episode_dirs)}",
        )
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    _require(len(seeds) == len(set(seeds)), "duplicate seeds discovered")
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
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=200)
    args = parser.parse_args()
    if args.expected_episodes < 0:
        parser.error("--expected-episodes must be non-negative")
    records = collect_boundary_records(
        args.episodes_root.resolve(), expected_episodes=args.expected_episodes
    )
    write_boundary_records(records, args.output)
    print(
        f"wrote {len(records)} TS02 boundary records "
        f"({sum(int(r['total_frames']) for r in records)} frames) to {args.output}"
    )


if __name__ == "__main__":
    main()
