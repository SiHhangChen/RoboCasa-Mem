#!/usr/bin/env python
"""Recover exact WR08 boundaries and demonstrated carton order."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.annotate.planner_to_stage_boundaries_wa02 import count_stage_steps


TASK_ID = "WR-08"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 17
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
ITEM_IDS = frozenset({"boxed_food_a", "boxed_food_b"})
SEMANTIC_ONLY_INDICES = frozenset({4, 7, 12, 15})
HOLDING_NAVIGATION_INDICES = frozenset({5, 13})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object in {path}")
    return value


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WR08 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def _holding_navigation_steps(
    details: Mapping[str, Any], *, path: str
) -> tuple[int, dict[str, int]]:
    navigation = details.get("holding_navigation")
    _require(isinstance(navigation, Mapping), f"{path}: missing holding_navigation")
    assert isinstance(navigation, Mapping)
    phases = navigation.get("phases")
    _require(
        isinstance(phases, list) and len(phases) == 2,
        f"{path}: expected rotate and translate phases",
    )
    assert isinstance(phases, list)
    _require(
        [phase.get("phase") for phase in phases if isinstance(phase, Mapping)]
        == ["rotate", "translate"],
        f"{path}: invalid holding-navigation phase order",
    )
    counts: list[int] = []
    for index, phase in enumerate(phases):
        _require(isinstance(phase, Mapping), f"{path}.phases[{index}] is invalid")
        assert isinstance(phase, Mapping)
        steps = phase.get("steps")
        _require(
            isinstance(steps, int) and not isinstance(steps, bool) and steps >= 0,
            f"{path}.phases[{index}].steps is invalid",
        )
        counts.append(int(steps))
    _require(
        details.get("steps") == counts[1],
        f"{path}: outer steps must repeat the translate phase only",
    )
    return sum(counts), {"rotate_steps": counts[0], "translate_steps": counts[1]}


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = _episode_seed(episode_dir)
    episode = _read_json(episode_dir / "episode.json")
    planner = _read_json(episode_dir / "planner.json")
    _require(episode.get("accepted") is True, f"seed {seed}: episode not accepted")
    _require(episode.get("planner_pass") is True, f"seed {seed}: planner did not pass")
    _require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task mismatch")
    _require(
        planner.get("expert_plan_version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: planner version mismatch",
    )
    _require(planner.get("pass") is True, f"seed {seed}: planner pass is false")

    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing metadata")
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

    order = tuple(str(value) for value in facts.get("transport_order", ()))
    _require(len(order) == 2 and set(order) == ITEM_IDS, f"seed {seed}: invalid order")
    source_colors_raw = facts.get("source_color_by_item")
    _require(isinstance(source_colors_raw, Mapping), f"seed {seed}: missing source colors")
    assert isinstance(source_colors_raw, Mapping)
    source_colors = {item: str(source_colors_raw.get(item, "")) for item in ITEM_IDS}
    _require(all(source_colors.values()), f"seed {seed}: invalid source colors")
    _require(len(set(source_colors.values())) == 2, f"seed {seed}: colors are not distinct")
    _require(
        tuple(expert_meta.get("transport_order", ())) == order
        and expert_meta.get("source_color_by_item") == source_colors,
        f"seed {seed}: expert metadata differs from resolved facts",
    )

    rollout = planner.get("rollout")
    result = planner.get("episode_result")
    _require(isinstance(rollout, Mapping), f"seed {seed}: missing rollout")
    _require(isinstance(result, Mapping), f"seed {seed}: missing episode result")
    assert isinstance(rollout, Mapping) and isinstance(result, Mapping)
    _require(
        result.get("success") is True
        and result.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: invalid episode result",
    )
    stages = rollout.get("stages")
    _require(
        isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expected {EXPECTED_STAGE_COUNT} stages",
    )
    assert isinstance(stages, list)
    expected_items = {
        2: order[0], 4: order[0], 6: order[0], 7: order[0],
        10: order[1], 12: order[1], 14: order[1], 15: order[1],
    }
    expected_primitives = {
        2: "pick", 3: "return_arm_to_regular_pose", 4: "latch_source_color_memory",
        5: "move_straight", 6: "place_at_accessible", 7: "verify_target_plate",
        8: "return_arm_to_regular_pose", 10: "pick", 11: "return_arm_to_regular_pose",
        12: "latch_source_color_memory", 13: "move_straight",
        14: "place_at_accessible", 15: "verify_target_plate",
        16: "return_arm_to_regular_pose",
    }
    recorded_total = episode.get("step_count")
    _require(
        isinstance(recorded_total, int)
        and not isinstance(recorded_total, bool)
        and recorded_total > 0,
        f"seed {seed}: invalid frame count",
    )
    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    for index, stage in enumerate(stages):
        _require(isinstance(stage, Mapping), f"seed {seed}: invalid stage {index}")
        assert isinstance(stage, Mapping)
        _require(
            stage.get("stage_index") == index and stage.get("success") is True,
            f"seed {seed}: stage {index} failed or has wrong index",
        )
        details = stage.get("details")
        _require(isinstance(details, Mapping), f"seed {seed}: stage {index} lacks details")
        assert isinstance(details, Mapping)
        if index in expected_primitives:
            _require(
                details.get("primitive") == expected_primitives[index],
                f"seed {seed}: stage {index} primitive mismatch",
            )
        if index in expected_items:
            item = details.get("object_name", details.get("item"))
            _require(item == expected_items[index], f"seed {seed}: stage {index} item mismatch")
        if index in HOLDING_NAVIGATION_INDICES:
            steps, fields = _holding_navigation_steps(
                details, path=f"seed[{seed}].stage[{index}]"
            )
        else:
            steps, fields = count_stage_steps(
                details, path=f"seed[{seed}].stage[{index}]"
            )
        if index in SEMANTIC_ONLY_INDICES:
            _require(steps == 0, f"seed {seed}: semantic stage {index} recorded actions")
        start = cumulative
        cumulative += steps
        output_stages.append(
            {
                "index": index,
                "kind": str(stage.get("stage_kind")),
                "steps": steps,
                "step_fields": fields,
                "start_frame": start,
                "end_frame": cumulative,
            }
        )
    _require(
        cumulative == recorded_total,
        f"seed {seed}: recovered {cumulative} != recorded {recorded_total}",
    )
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "transport_order": list(order),
        "source_color_by_item": source_colors,
        "ordered_source_colors": [source_colors[item] for item in order],
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "source_episode_dir": episode_dir.name,
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
    _require(
        len(episode_dirs) == expected_episodes,
        f"expected {expected_episodes} WR08 episodes, found {len(episode_dirs)}",
    )
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    _require(len(set(seeds)) == len(seeds), "duplicate WR08 seeds")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=200)
    args = parser.parse_args()
    records = collect_boundary_records(
        args.episodes_root.resolve(), expected_episodes=args.expected_episodes
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(f"wrote {len(records)} WR08 boundary records to {args.output}")


if __name__ == "__main__":
    main()
