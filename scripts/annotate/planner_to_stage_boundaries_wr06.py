#!/usr/bin/env python
"""Recover exact WR06 boundaries and demonstration metadata.

WR06 requires remembering the ketchup bottle's visible initial shelf level.
The seed-selected plate side remains recorded for trajectory verification but
is not exposed in the task instruction or semantic subtask labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "WR-06"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 15
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})
NAVIGATE_STAGE_INDICES = frozenset({0, 1, 5, 9, 12})
LEVEL_DISPLAY = {"shelf_lower": "lower", "shelf_upper": "upper"}
PLATE_DISPLAY = {"plate_left": "left", "plate_right": "right"}
TASK_INSTRUCTION = (
    "Remember whether the ketchup bottle starts on the lower or upper shelf. "
    "Move it beside the serving plate, then return it to its original shelf level."
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _non_negative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return int(value)


def count_recorded_steps(
    value: Any, *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Count ordinary WR06 action fields without nested aggregate duplication."""
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
                    count = _non_negative_int(child, child_path)
                    total += count
                    fields[key] += count
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


def count_navigation_steps(
    details: Mapping[str, Any], *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Navigation's top-level count excludes the station torso adjustment."""
    base_steps = _non_negative_int(details.get("steps"), f"{path}.steps")
    torso = details.get("station_torso")
    _require(isinstance(torso, Mapping), f"{path}.station_torso must be an object")
    assert isinstance(torso, Mapping)
    torso_steps = _non_negative_int(
        torso.get("steps"), f"{path}.station_torso.steps"
    )
    return base_steps + torso_steps, {
        "executed_steps": 0,
        "release_settle_steps": 0,
        "steps": base_steps + torso_steps,
        "navigation_steps": base_steps,
        "station_torso_steps": torso_steps,
    }


def count_place_steps(
    details: Mapping[str, Any], *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Count placement motions once, excluding duplicated early-stop evidence."""
    release = details.get("release_gripper")
    stages = details.get("stages")
    _require(isinstance(release, Mapping), f"{path}.release_gripper is missing")
    _require(isinstance(stages, list), f"{path}.stages must be a list")
    assert isinstance(release, Mapping) and isinstance(stages, list)
    release_steps = _non_negative_int(
        release.get("steps"), f"{path}.release_gripper.steps"
    )
    settle_steps = _non_negative_int(
        release.get("release_settle_steps"),
        f"{path}.release_gripper.release_settle_steps",
    )
    executed_steps = 0
    for index, stage in enumerate(stages):
        _require(
            isinstance(stage, Mapping), f"{path}.stages[{index}] must be an object"
        )
        assert isinstance(stage, Mapping)
        execution = stage.get("execution")
        _require(
            isinstance(execution, Mapping),
            f"{path}.stages[{index}].execution must be an object",
        )
        assert isinstance(execution, Mapping)
        stage_steps = _non_negative_int(
            execution.get("executed_steps"),
            f"{path}.stages[{index}].execution.executed_steps",
        )
        executed_steps += stage_steps

        accepted = stage.get("accepted_trajectory_early_stop_for_release")
        if accepted is not None:
            _require(
                isinstance(accepted, Mapping),
                f"{path}.stages[{index}].accepted early stop is invalid",
            )
            assert isinstance(accepted, Mapping)
            early_stop = accepted.get("early_stop")
            _require(
                isinstance(early_stop, Mapping),
                f"{path}.stages[{index}].accepted early stop evidence is missing",
            )
            assert isinstance(early_stop, Mapping)
            repeated_steps = _non_negative_int(
                early_stop.get("executed_steps"),
                f"{path}.stages[{index}].accepted early stop executed_steps",
            )
            _require(
                repeated_steps == stage_steps,
                f"{path}.stages[{index}] early-stop evidence differs from execution",
            )

    total = release_steps + settle_steps + executed_steps
    return total, {
        "executed_steps": executed_steps,
        "release_settle_steps": settle_steps,
        "steps": release_steps,
        "navigation_steps": 0,
        "station_torso_steps": 0,
    }


def count_stage_steps(
    stage: Mapping[str, Any], *, path: str = "stage"
) -> tuple[int, dict[str, int]]:
    details = stage.get("details")
    _require(isinstance(details, Mapping), f"{path}.details must be an object")
    assert isinstance(details, Mapping)
    if stage.get("stage_kind") == "navigate":
        return count_navigation_steps(details, path=f"{path}.details")
    if details.get("primitive") == "place_at_accessible":
        return count_place_steps(details, path=f"{path}.details")
    steps, fields = count_recorded_steps(details, path=f"{path}.details")
    return steps, {
        **fields,
        "navigation_steps": 0,
        "station_torso_steps": 0,
    }


def build_stage_names(original_level: str, plate_region: str) -> list[str]:
    _require(original_level in LEVEL_DISPLAY, f"invalid WR06 level {original_level!r}")
    _require(plate_region in PLATE_DISPLAY, f"invalid WR06 plate region {plate_region!r}")
    return [
        "observe_shelf_levels",
        "navigate_to_original_level",
        "pick_item_from_original_level",
        "stow_item_after_shelf_pick",
        "normalize_transport_posture",
        "navigate_item_to_plate_stage",
        f"place_item_at_{plate_region}",
        "return_arm_after_plate_stage",
        "latch_original_level_memory_barrier",
        "navigate_to_staged_item",
        "pick_item_from_plate_stage",
        "stow_item_for_return",
        "navigate_to_remembered_level",
        "return_item_to_original_level",
        "return_arm_after_home_place",
    ]


def build_task_instruction() -> str:
    return TASK_INSTRUCTION


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WR06 episode directory: {episode_dir.name!r}")
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
    _require(expert.get("version") == EXPERT_PLAN_VERSION, f"seed {seed}: expert version mismatch")
    _require(expert.get("stage_count") == EXPECTED_STAGE_COUNT, f"seed {seed}: expert stage count mismatch")
    original_level = str(facts.get("demo_original_level", ""))
    plate_region = str(facts.get("plate_stage_region", ""))
    _require(original_level in LEVEL_DISPLAY, f"seed {seed}: invalid original level")
    _require(plate_region in PLATE_DISPLAY, f"seed {seed}: invalid plate region")
    _require(
        expert_meta.get("demo_original_level") == original_level
        and expert_meta.get("plate_stage_region") == plate_region,
        f"seed {seed}: expert facts differ from resolved facts",
    )

    diagnostics = planner.get("final_diagnostics")
    _require(isinstance(diagnostics, Mapping), f"seed {seed}: missing diagnostics")
    assert isinstance(diagnostics, Mapping)
    _require(diagnostics.get("success") is True, f"seed {seed}: final success is false")
    expected_lower = original_level == "shelf_lower"
    _require(
        diagnostics.get("initial_level_physically_verified") is True
        and diagnostics.get("item_staged_at_fixed_plate") is True
        and diagnostics.get("memory_barrier_latched") is True
        and diagnostics.get("first_return_level_correct") is True
        and diagnostics.get("item_in_original_level") is True
        and diagnostics.get("item_not_in_other_level") is True
        and diagnostics.get("wrong_return_level_attempted") is False
        and diagnostics.get("no_protocol_violation") is True,
        f"seed {seed}: memory/return protocol validation failed",
    )
    _require(
        diagnostics.get("current_item_in_lower") is expected_lower
        and diagnostics.get("current_item_in_upper") is (not expected_lower),
        f"seed {seed}: final physical shelf level mismatch",
    )

    episode_result = planner.get("episode_result")
    stages = planner.get("rollout", {}).get("stages")
    _require(isinstance(episode_result, Mapping), f"seed {seed}: missing episode result")
    _require(
        isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expected {EXPECTED_STAGE_COUNT} stages",
    )
    assert isinstance(episode_result, Mapping) and isinstance(stages, list)
    _require(
        episode_result.get("success") is True
        and episode_result.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: planner episode result mismatch",
    )

    names = build_stage_names(original_level, plate_region)
    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    field_totals = {
        "executed_steps": 0,
        "release_settle_steps": 0,
        "steps": 0,
        "navigation_steps": 0,
        "station_torso_steps": 0,
    }
    for index, (stage, name) in enumerate(zip(stages, names)):
        _require(isinstance(stage, Mapping), f"seed {seed}: stage {index} is invalid")
        assert isinstance(stage, Mapping)
        _require(stage.get("stage_index") == index, f"seed {seed}: stage index mismatch")
        _require(stage.get("success") is True, f"seed {seed}: stage {index} failed")
        expected_kind = "navigate" if index in NAVIGATE_STAGE_INDICES else "operate"
        _require(
            stage.get("stage_kind") == expected_kind,
            f"seed {seed}: stage {index} kind mismatch",
        )
        steps, fields = count_stage_steps(
            stage, path=f"seed[{seed}].stage[{index}]"
        )
        start_frame = cumulative
        cumulative += steps
        for key, value in fields.items():
            field_totals[key] += value
        details = stage["details"]
        assert isinstance(details, Mapping)
        output_stages.append(
            {
                "index": index,
                "name": name,
                "kind": expected_kind,
                "primitive": str(details.get("primitive", "")),
                "steps": steps,
                "step_fields": fields,
                "start_frame": start_frame,
                "end_frame": cumulative,
            }
        )

    recorded_total = _non_negative_int(
        episode.get("step_count"), f"seed[{seed}].step_count"
    )
    _require(recorded_total > 0, f"seed {seed}: episode is empty")
    _require(
        cumulative == recorded_total,
        f"seed {seed}: recovered {cumulative} != recorded {recorded_total}; fields={field_totals}",
    )
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "original_level": LEVEL_DISPLAY[original_level],
        "original_level_id": original_level,
        "plate_side": PLATE_DISPLAY[plate_region],
        "plate_region": plate_region,
        "high_level_instruction": build_task_instruction(),
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": field_totals,
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
    if expected_episodes > 0:
        _require(
            len(episode_dirs) == expected_episodes,
            f"expected {expected_episodes} episodes, found {len(episode_dirs)}",
        )
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    _require(len(seeds) == len(set(seeds)), "duplicate WR06 seeds discovered")
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
        f"wrote {len(records)} WR06 boundary records "
        f"({sum(int(record['total_frames']) for record in records)} frames, "
        f"{len({record['high_level_instruction'] for record in records})} task plans) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
