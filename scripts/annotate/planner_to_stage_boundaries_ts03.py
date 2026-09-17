#!/usr/bin/env python
"""Recover TS03 stage boundaries and per-sugar memory events.

TS03 varies the requested sugar count K and the visible sugar count N across
the full K<=N<=3 table.  The high-level policy still uses two semantic labels: one
repeatable sugar-delivery subtask and one completion-button subtask.  This
bridge reconstructs every expert stage and retains the end of each completed
sugar cycle as an explicit keyframe so MemER can count progress toward K.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "TS03"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
VALID_PROFILES = frozenset({(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)})
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})


def pick_close_start_offset(details: Mapping[str, Any], *, path: str) -> int:
    """Return the pick-local frame where the closing gripper command begins."""
    open_gripper = details.get("open_gripper")
    preshape = details.get("preshape_gripper")
    close_gripper = details.get("close_gripper")
    stages = details.get("stages")
    _require(isinstance(open_gripper, Mapping), f"{path}: missing open_gripper")
    _require(isinstance(preshape, Mapping), f"{path}: missing preshape_gripper")
    _require(isinstance(close_gripper, Mapping), f"{path}: missing close_gripper")
    _require(isinstance(stages, list) and len(stages) == 3, f"{path}: invalid pick stages")
    assert isinstance(open_gripper, Mapping)
    assert isinstance(preshape, Mapping)
    assert isinstance(close_gripper, Mapping)
    assert isinstance(stages, list)
    _require(
        [stage.get("stage") for stage in stages if isinstance(stage, Mapping)]
        == ["pregrasp", "grasp", "lift"],
        f"{path}: unexpected pick-stage order",
    )

    def steps(mapping: Mapping[str, Any], key: str, field_path: str) -> int:
        value = mapping.get(key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{field_path} must be a non-negative integer",
        )
        return int(value)

    pregrasp = stages[0]
    grasp = stages[1]
    _require(isinstance(pregrasp, Mapping) and isinstance(grasp, Mapping), f"{path}: invalid pick stages")
    assert isinstance(pregrasp, Mapping) and isinstance(grasp, Mapping)
    pregrasp_execution = pregrasp.get("execution")
    grasp_execution = grasp.get("execution")
    _require(isinstance(pregrasp_execution, Mapping), f"{path}: missing pregrasp execution")
    _require(isinstance(grasp_execution, Mapping), f"{path}: missing grasp execution")
    assert isinstance(pregrasp_execution, Mapping) and isinstance(grasp_execution, Mapping)
    close_steps = steps(close_gripper, "steps", f"{path}.close_gripper.steps")
    _require(close_steps >= 7, f"{path}: close command is too short for evidence")
    return sum(
        (
            steps(open_gripper, "steps", f"{path}.open_gripper.steps"),
            steps(
                pregrasp_execution,
                "executed_steps",
                f"{path}.stages[0].execution.executed_steps",
            ),
            steps(preshape, "steps", f"{path}.preshape_gripper.steps"),
            steps(
                grasp_execution,
                "executed_steps",
                f"{path}.stages[1].execution.executed_steps",
            ),
        )
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
    """Count the action-step fields persisted by the TS03 recorder."""
    total = 0
    fields = {key: 0 for key in sorted(STEP_KEYS)}

    def visit(node: Any, node_path: str) -> None:
        nonlocal total
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key)
                if key in STEP_KEYS:
                    if (
                        isinstance(child, bool)
                        or not isinstance(child, int)
                        or child < 0
                    ):
                        raise ValueError(
                            f"{node_path}.{key} must be a non-negative integer"
                        )
                    total += child
                    fields[key] += child
                visit(child, f"{node_path}.{key}")
        elif isinstance(node, Sequence) and not isinstance(
            node, (str, bytes, bytearray)
        ):
            for index, child in enumerate(node):
                visit(child, f"{node_path}[{index}]")

    visit(value, path)
    return total, fields


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


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected TS03 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def _resolved_facts(episode: Mapping[str, Any], *, seed: int) -> Mapping[str, Any]:
    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing episode metadata")
    assert isinstance(metadata, Mapping)
    _require(metadata.get("task_id") == TASK_ID, f"seed {seed}: episode task mismatch")
    _require(metadata.get("seed") == seed, f"seed {seed}: episode seed mismatch")
    resolved = metadata.get("resolved_episode")
    _require(isinstance(resolved, Mapping), f"seed {seed}: missing resolved_episode")
    assert isinstance(resolved, Mapping)
    facts = resolved.get("facts")
    _require(isinstance(facts, Mapping), f"seed {seed}: missing resolved facts")
    assert isinstance(facts, Mapping)
    return facts


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

    facts = _resolved_facts(episode, seed=seed)
    target_count = int(facts["target_sugar_count"])
    total_count = int(facts["total_sugar_count"])
    _require(
        (target_count, total_count) in VALID_PROFILES,
        f"seed {seed}: unsupported K/N profile {(target_count, total_count)}",
    )
    sugar_names = tuple(str(value) for value in facts.get("sugar_names", ()))
    target_names = tuple(
        str(value) for value in facts.get("target_sugar_names", ())
    )
    distractor_names = tuple(
        str(value) for value in facts.get("distractor_sugar_names", ())
    )
    _require(
        len(sugar_names) == total_count and len(set(sugar_names)) == total_count,
        f"seed {seed}: invalid sugar_names",
    )
    _require(
        len(target_names) == target_count
        and len(set(target_names)) == target_count
        and set(target_names).issubset(sugar_names),
        f"seed {seed}: invalid target_sugar_names",
    )
    _require(
        set(distractor_names) == set(sugar_names) - set(target_names),
        f"seed {seed}: distractor sugar set mismatch",
    )

    metadata = episode["metadata"]
    assert isinstance(metadata, Mapping)
    expert = metadata.get("expert_plan")
    _require(isinstance(expert, Mapping), f"seed {seed}: missing expert plan")
    assert isinstance(expert, Mapping)
    _require(
        expert.get("version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: episode expert version mismatch",
    )
    expert_metadata = expert.get("metadata")
    _require(
        isinstance(expert_metadata, Mapping),
        f"seed {seed}: missing expert plan metadata",
    )
    assert isinstance(expert_metadata, Mapping)
    _require(
        int(expert_metadata.get("target_sugar_count", -1)) == target_count
        and int(expert_metadata.get("total_sugar_count", -1)) == total_count,
        f"seed {seed}: expert and resolved K/N profiles differ",
    )

    stage_names = build_stage_names(target_names)
    stages = planner.get("rollout", {}).get("stages")
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
    _require(
        isinstance(episode_result, Mapping),
        f"seed {seed}: missing planner episode result",
    )
    assert isinstance(episode_result, Mapping)
    _require(
        episode_result.get("success") is True
        and episode_result.get("stage_count") == len(stage_names),
        f"seed {seed}: planner episode result mismatch",
    )

    recorded_total = int(episode["step_count"])
    output_stages: list[dict[str, Any]] = []
    keyframes: list[int] = []
    cumulative = 0
    field_totals = {key: 0 for key in sorted(STEP_KEYS)}
    placed_sugars = 0
    sugar_close_frames: list[int] = []
    for index, (stage, name) in enumerate(zip(stages, stage_names)):
        _require(isinstance(stage, Mapping), f"seed {seed}: stage {index} is invalid")
        assert isinstance(stage, Mapping)
        _require(
            stage.get("stage_index") == index,
            f"seed {seed}: stage index mismatch at {index}",
        )
        _require(stage.get("success") is True, f"seed {seed}: stage {index} failed")
        expected_kind = "navigate" if (
            name.startswith("navigate_")
        ) else "operate"
        _require(
            stage.get("stage_kind") == expected_kind,
            f"seed {seed}: stage {index} kind mismatch",
        )
        details = stage.get("details")
        _require(isinstance(details, Mapping), f"seed {seed}: stage {index} has no details")
        assert isinstance(details, Mapping)
        steps, fields = count_recorded_steps(
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
                "kind": str(stage.get("stage_kind", "")),
                "steps": steps,
                "step_fields": fields,
                "start_frame": start_frame,
                "end_frame": cumulative,
            }
        )
        if name.startswith("pick_"):
            close_frame = start_frame + pick_close_start_offset(
                details, path=f"seed[{seed}].stage[{index}].details"
            )
            _require(
                start_frame <= close_frame < cumulative,
                f"seed {seed}: close frame lies outside pick stage {index}",
            )
            sugar_close_frames.append(close_frame)
        if name.startswith("return_after_") and name.endswith("_place"):
            placed_sugars += 1
            _require(steps > 0, f"seed {seed}: completed sugar cycle is empty")
            keyframes.append(cumulative - 1)
        if name == "navigate_completion_button":
            keyframes.append(start_frame)

    _require(
        cumulative == recorded_total,
        f"seed {seed}: recovered {cumulative} != recorded {recorded_total}",
    )
    _require(
        placed_sugars == target_count,
        f"seed {seed}: found {placed_sugars} placements, expected K={target_count}",
    )
    _require(
        len(sugar_close_frames) == target_count,
        f"seed {seed}: found {len(sugar_close_frames)} closes, expected K={target_count}",
    )
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "profile_id": f"k{target_count}_n{total_count}",
        "target_sugar_count": target_count,
        "total_sugar_count": total_count,
        "sugar_close_frames": sugar_close_frames,
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "keyframes": sorted(set(keyframes)),
        "stages": output_stages,
    }


def collect_boundary_records(
    episodes_root: Path, *, expected_episodes: int
) -> list[dict[str, Any]]:
    episode_dirs = sorted(
        path
        for path in episodes_root.glob("seed_*_attempt_00")
        if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name)
    )
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
    total_frames = sum(int(record["total_frames"]) for record in records)
    print(
        f"wrote {len(records)} TS03 boundary records "
        f"({total_frames} frames) to {args.output}"
    )


if __name__ == "__main__":
    main()
