#!/usr/bin/env python
"""Recover TS01 stage boundaries and per-sugar memory events.

TS01 intentionally exposes only two subtask labels: adding sugar and pressing
the completion button. The repeated sugar cycles remain one semantic label,
while the final frame of every completed pick-place-return cycle is retained
as an explicit keyframe event so MemER can count progress for K=2/3 episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "TS01"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def count_recorded_steps(value: Any, *, path: str = "details") -> tuple[int, dict[str, int]]:
    total = 0
    fields = {key: 0 for key in sorted(STEP_KEYS)}

    def visit(node: Any, node_path: str) -> None:
        nonlocal total
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key)
                if key in STEP_KEYS:
                    if isinstance(child, bool) or not isinstance(child, int) or child < 0:
                        raise ValueError(f"{node_path}.{key} must be a non-negative integer")
                    total += child
                    fields[key] += child
                visit(child, f"{node_path}.{key}")
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, child in enumerate(node):
                visit(child, f"{node_path}[{index}]")

    visit(value, path)
    return total, fields


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected TS01 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def _resolved_facts(episode: Mapping[str, Any], *, seed: int) -> Mapping[str, Any]:
    resolved = episode.get("metadata", {}).get("resolved_episode")
    _require(isinstance(resolved, Mapping), f"seed {seed}: missing resolved_episode metadata")
    facts = resolved.get("facts")
    _require(isinstance(facts, Mapping), f"seed {seed}: missing resolved_episode.facts")
    return facts


def build_stage_names(target_sugar_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for raw_sugar in target_sugar_names:
        sugar = str(raw_sugar)
        _require(bool(sugar), "target sugar name must not be empty")
        names.extend(
            [
                f"navigate_{sugar}",
                f"pick_{sugar}",
                f"retract_{sugar}_after_pick",
                f"navigate_{sugar}_to_saucepan",
                f"place_{sugar}_saucepan",
                f"return_after_{sugar}_place",
            ]
        )
    names.extend(["navigate_completion_button", "press_completion_button"])
    return names


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = _episode_seed(episode_dir)
    episode = _read_json(episode_dir / "episode.json")
    planner = _read_json(episode_dir / "planner.json")
    _require(episode.get("accepted") is True, f"seed {seed}: episode is not accepted")
    _require(episode.get("planner_pass") is True, f"seed {seed}: planner did not pass")
    _require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task is not TS01")
    _require(planner.get("expert_plan_version") == EXPERT_PLAN_VERSION, f"seed {seed}: planner version mismatch")
    _require(planner.get("pass") is True, f"seed {seed}: planner pass is false")

    stages = planner.get("rollout", {}).get("stages")
    _require(isinstance(stages, list) and stages, f"seed {seed}: missing planner rollout stages")
    facts = _resolved_facts(episode, seed=seed)
    target_count = int(facts["target_sugar_count"])
    target_names = facts.get("target_sugar_names")
    _require(
        isinstance(target_names, list) and len(target_names) == target_count,
        f"seed {seed}: invalid target_sugar_names",
    )
    stage_names = build_stage_names(target_names)
    _require(
        len(stage_names) == len(stages),
        f"seed {seed}: reconstructed {len(stage_names)} names for {len(stages)} stages",
    )
    total_frames = int(episode["step_count"])
    output_stages: list[dict[str, Any]] = []
    keyframes: list[int] = []
    cumulative = 0
    field_totals = {key: 0 for key in sorted(STEP_KEYS)}
    placed_sugars = 0

    for index, (stage, name) in enumerate(zip(stages, stage_names)):
        _require(isinstance(stage, Mapping), f"seed {seed}: stage {index} is invalid")
        _require(stage.get("stage_index") == index, f"seed {seed}: stage index mismatch at {index}")
        _require(stage.get("success") is True, f"seed {seed}: stage {index} failed")
        details = stage.get("details")
        _require(isinstance(details, Mapping), f"seed {seed}: stage {index} has no details")
        steps, fields = count_recorded_steps(details, path=f"seed[{seed}].stage[{index}].details")
        start_frame = cumulative
        cumulative += steps
        for key, value in fields.items():
            field_totals[key] += value
        output_stages.append(
            {
                "index": index,
                "name": name,
                "kind": str(stage.get("stage_kind", "")),
                "steps": steps,
                "step_fields": fields,
                "start_frame": start_frame,
                "end_frame": cumulative,
            }
        )
        if name.startswith("return_after_") and name.endswith("_place"):
            placed_sugars += 1
            _require(steps > 0, f"seed {seed}: completed sugar cycle {name!r} is empty")
            keyframes.append(cumulative - 1)
        if name == "navigate_completion_button":
            keyframes.append(start_frame)

    _require(cumulative == total_frames, f"seed {seed}: recovered {cumulative} != recorded {total_frames}")
    _require(placed_sugars == target_count, f"seed {seed}: found {placed_sugars} placements, expected K={target_count}")
    _require(output_stages[-1]["name"] == "press_completion_button", f"seed {seed}: final stage is not button press")
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "target_sugar_count": target_count,
        "total_sugar_count": int(facts["total_sugar_count"]),
        "total_frames": cumulative,
        "recorded_total_frames": total_frames,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "keyframes": sorted(set(keyframes)),
        "stages": output_stages,
    }


def collect_boundary_records(episodes_root: Path, *, expected_episodes: int) -> list[dict[str, Any]]:
    episode_dirs = sorted(
        path for path in episodes_root.glob("seed_*_attempt_00")
        if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name)
    )
    if expected_episodes > 0:
        _require(len(episode_dirs) == expected_episodes, f"expected {expected_episodes} episodes, found {len(episode_dirs)}")
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    _require(len(seeds) == len(set(seeds)), "duplicate seeds discovered")
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
    print(f"wrote {len(records)} TS01 boundary records ({sum(int(r['total_frames']) for r in records)} frames) to {args.output}")


if __name__ == "__main__":
    main()
