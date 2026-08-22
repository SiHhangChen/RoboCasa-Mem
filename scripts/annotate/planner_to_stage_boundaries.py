#!/usr/bin/env python
"""Recover exact WA-01 stage boundaries from recorded planner summaries.

The native recorder stores action counts inside each stage's ``details`` tree,
but it does not persist the original expert-plan stage names.  This bridge
reconstructs the canonical WA-01 v2 names from episode metadata and emits the
``boundaries.jsonl`` contract consumed by ``apply_subtask_labels.py``.

No simulator replay is required.  The bridge is deliberately strict: every
episode must contain the canonical 28 successful stages and the recovered
stage counts must sum exactly to ``episode.json.step_count``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "WA-01"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 28
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})
NAVIGATE_STAGE_INDICES = frozenset({0, 3, 11, 15, 21})
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def count_recorded_steps(value: Any, *, path: str = "details") -> tuple[int, dict[str, int]]:
    """Count action steps using only the recorder's exact step-field whitelist."""
    total = 0
    fields = {key: 0 for key in sorted(STEP_KEYS)}

    def visit(node: Any, node_path: str) -> None:
        nonlocal total
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key)
                child_path = f"{node_path}.{key}"
                if key in STEP_KEYS:
                    if isinstance(child, bool) or not isinstance(child, int):
                        raise ValueError(
                            f"{child_path} must be a non-negative integer, got {child!r}"
                        )
                    if child < 0:
                        raise ValueError(f"{child_path} must be non-negative, got {child}")
                    total += child
                    fields[key] += child
                visit(child, child_path)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, child in enumerate(node):
                visit(child, f"{node_path}[{index}]")

    visit(value, path)
    return total, fields


def build_stage_names(first: str, second: str, sugared: str) -> list[str]:
    """Rebuild the canonical names created by WA-01 expert plan v2."""
    _require(first and second and first != second, "delivery_order must contain two distinct cups")
    _require(sugared in {first, second}, f"sugared coffee {sugared!r} is not in delivery_order")
    return [
        "navigate_pick_sugar",
        "pick_sugar",
        "return_picked_sugar_to_regular_pose",
        "navigate_place_sugar",
        "place_sugar",
        "record_sugar_delivery",
        "return_after_place_sugar_to_regular_pose",
        f"verify_first_pick_reachability_{first}",
        f"pick_first_delivery_{first}",
        f"return_first_delivery_{first}_to_regular_pose_for_transport",
        f"verify_first_{first}_held_for_transport",
        f"navigate_deliver_first_{first}_dining",
        f"place_first_{first}_dining",
        f"verify_first_{first}_on_dining",
        f"return_after_place_first_{first}_dining_to_regular_pose",
        f"navigate_pick_second_delivery_{second}",
        f"verify_second_pick_reachability_{second}",
        f"pick_second_delivery_{second}",
        f"raise_second_delivery_{second}_for_transport",
        f"return_second_delivery_{second}_to_regular_pose_for_transport",
        f"verify_second_{second}_held_for_transport",
        f"navigate_deliver_second_{second}_dining",
        f"place_second_{second}_dining",
        f"verify_second_{second}_on_dining",
        "verify_cups_ready_for_selection",
        "pick_sugared_coffee_topdown",
        "verify_sugared_coffee_held_after_topdown_pick",
        "select_sugared_coffee",
    ]


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WA-01 episode directory name: {episode_dir.name!r}")
    return int(match.group(1))


def _expert_metadata(episode: Mapping[str, Any], *, seed: int) -> Mapping[str, Any]:
    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing episode metadata")
    assert isinstance(metadata, Mapping)
    expert_plan = metadata.get("expert_plan")
    _require(isinstance(expert_plan, Mapping), f"seed {seed}: missing expert_plan metadata")
    assert isinstance(expert_plan, Mapping)
    _require(
        expert_plan.get("version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: episode expert plan version is {expert_plan.get('version')!r}",
    )
    value = expert_plan.get("metadata")
    _require(isinstance(value, Mapping), f"seed {seed}: missing expert_plan.metadata")
    assert isinstance(value, Mapping)
    return value


def _resolve_stage_names(episode: Mapping[str, Any], *, seed: int) -> list[str]:
    expert = _expert_metadata(episode, seed=seed)
    delivery_order = expert.get("delivery_order")
    _require(
        isinstance(delivery_order, list) and len(delivery_order) == 2,
        f"seed {seed}: delivery_order must contain exactly two cups",
    )
    assert isinstance(delivery_order, list)
    first, second = (str(value) for value in delivery_order)
    sugared = expert.get("sugared_coffee_id")
    _require(isinstance(sugared, str) and sugared, f"seed {seed}: missing sugared_coffee_id")
    assert isinstance(sugared, str)
    return build_stage_names(first, second, sugared)


def _resolve_sugared_coffee_slot(
    episode: Mapping[str, Any], planner: Mapping[str, Any], *, seed: int
) -> str:
    """Recover the fixed dining placement zone of the actually sugared cup."""
    expert = _expert_metadata(episode, seed=seed)
    final_assignment = expert.get("final_slot_assignment")
    _require(isinstance(final_assignment, Mapping), f"seed {seed}: missing final_slot_assignment")
    sugared_id = expert.get("sugared_coffee_id")
    _require(isinstance(sugared_id, str) and sugared_id, f"seed {seed}: missing sugared_coffee_id")
    planned_slot = final_assignment.get(sugared_id)
    _require(
        planned_slot in {"left", "right"},
        f"seed {seed}: invalid planned slot {planned_slot!r} for {sugared_id!r}",
    )

    diagnostics = planner.get("final_diagnostics")
    _require(isinstance(diagnostics, Mapping), f"seed {seed}: missing final_diagnostics")
    assert isinstance(diagnostics, Mapping)
    sugar = diagnostics.get("sugar")
    _require(isinstance(sugar, Mapping), f"seed {seed}: missing final_diagnostics.sugar")
    assert isinstance(sugar, Mapping)
    _require(
        sugar.get("sugared_coffee_id") == sugared_id,
        f"seed {seed}: planner sugared cup {sugar.get('sugared_coffee_id')!r} "
        f"!= expert cup {sugared_id!r}",
    )
    placement = diagnostics.get("dining_placement")
    _require(isinstance(placement, Mapping), f"seed {seed}: missing dining_placement")
    assert isinstance(placement, Mapping)
    occupancy = placement.get("occupancy")
    _require(isinstance(occupancy, Mapping), f"seed {seed}: missing dining occupancy")
    assert isinstance(occupancy, Mapping)
    observed_slots = [
        slot
        for slot in ("left", "right")
        if isinstance(occupancy.get(slot), Sequence)
        and not isinstance(occupancy.get(slot), (str, bytes, bytearray))
        and sugared_id in occupancy.get(slot, [])
    ]
    _require(
        len(observed_slots) == 1,
        f"seed {seed}: sugared cup {sugared_id!r} must occupy exactly one slot, found {observed_slots}",
    )
    observed_slot = observed_slots[0]
    _require(
        observed_slot == planned_slot,
        f"seed {seed}: observed sugared slot {observed_slot!r} != planned {planned_slot!r}",
    )
    return observed_slot


def _validate_episode_contract(
    episode: Mapping[str, Any], planner: Mapping[str, Any], *, seed: int
) -> tuple[int, list[Mapping[str, Any]]]:
    _require(episode.get("accepted") is True, f"seed {seed}: episode is not accepted")
    _require(episode.get("planner_pass") is True, f"seed {seed}: episode planner did not pass")
    step_count = episode.get("step_count")
    _require(
        isinstance(step_count, int) and not isinstance(step_count, bool) and step_count > 0,
        f"seed {seed}: invalid episode step_count {step_count!r}",
    )
    assert isinstance(step_count, int)

    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing episode metadata")
    assert isinstance(metadata, Mapping)
    _require(metadata.get("task_id") == TASK_ID, f"seed {seed}: episode task is not {TASK_ID}")
    _require(metadata.get("seed") == seed, f"seed {seed}: episode metadata seed mismatch")

    _require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task is not {TASK_ID}")
    _require(
        planner.get("expert_plan_version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: planner version is {planner.get('expert_plan_version')!r}",
    )
    _require(planner.get("pass") is True, f"seed {seed}: planner pass is false")
    episode_result = planner.get("episode_result")
    _require(isinstance(episode_result, Mapping), f"seed {seed}: missing planner episode_result")
    assert isinstance(episode_result, Mapping)
    _require(episode_result.get("success") is True, f"seed {seed}: planner result failed")
    _require(
        episode_result.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: planner stage_count is {episode_result.get('stage_count')!r}",
    )

    rollout = planner.get("rollout")
    _require(isinstance(rollout, Mapping), f"seed {seed}: missing planner rollout")
    assert isinstance(rollout, Mapping)
    stages = rollout.get("stages")
    _require(
        isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expected {EXPECTED_STAGE_COUNT} rollout stages",
    )
    assert isinstance(stages, list)
    for index, stage in enumerate(stages):
        _require(isinstance(stage, Mapping), f"seed {seed}: stage {index} is not an object")
        assert isinstance(stage, Mapping)
        _require(stage.get("stage_index") == index, f"seed {seed}: stage index mismatch at {index}")
        _require(stage.get("success") is True, f"seed {seed}: stage {index} failed")
        expected_kind = "navigate" if index in NAVIGATE_STAGE_INDICES else "operate"
        _require(
            stage.get("stage_kind") == expected_kind,
            f"seed {seed}: stage {index} kind is {stage.get('stage_kind')!r}, "
            f"expected {expected_kind!r}",
        )
        _require(
            isinstance(stage.get("details"), Mapping),
            f"seed {seed}: stage {index} has no details object",
        )
    return step_count, stages


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = _episode_seed(episode_dir)
    episode_path = episode_dir / "episode.json"
    planner_path = episode_dir / "planner.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"seed {seed}: missing {episode_path}")
    if not planner_path.is_file():
        raise FileNotFoundError(f"seed {seed}: missing {planner_path}")

    episode = _read_json(episode_path)
    planner = _read_json(planner_path)
    recorded_total, planner_stages = _validate_episode_contract(episode, planner, seed=seed)
    stage_names = _resolve_stage_names(episode, seed=seed)
    sugared_coffee_slot = _resolve_sugared_coffee_slot(episode, planner, seed=seed)
    _require(
        len(stage_names) == len(planner_stages),
        f"seed {seed}: reconstructed stage-name count mismatch",
    )

    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    field_totals = {key: 0 for key in sorted(STEP_KEYS)}
    for index, (stage, name) in enumerate(zip(planner_stages, stage_names)):
        stage_steps, stage_fields = count_recorded_steps(
            stage["details"], path=f"seed[{seed}].stage[{index}].details"
        )
        cumulative += stage_steps
        for key, value in stage_fields.items():
            field_totals[key] += value
        output_stages.append(
            {
                "index": index,
                "name": name,
                "kind": str(stage["stage_kind"]),
                "steps": stage_steps,
                "step_fields": stage_fields,
                "end_frame": cumulative,
            }
        )

    _require(
        cumulative == recorded_total,
        f"seed {seed}: recovered {cumulative} stage frames != recorded {recorded_total}; "
        f"field totals={field_totals}",
    )
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "sugared_coffee_slot": sugared_coffee_slot,
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
            f"expected {expected_episodes} WA-01 episodes under {episodes_root}, "
            f"found {len(episode_dirs)}",
        )
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    _require(len(seeds) == len(set(seeds)), f"duplicate seeds discovered under {episodes_root}")
    if expected_episodes > 0:
        _require(
            seeds == list(range(expected_episodes)),
            f"expected contiguous seeds 0..{expected_episodes - 1}, got {seeds}",
        )
    return records


def write_boundary_records(records: Sequence[Mapping[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes-root",
        type=Path,
        required=True,
        help="Native WA-01 episode root containing seed_XXXXXX_attempt_00 directories.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output boundaries.jsonl path.")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=200,
        help="Required episode count; use 0 only for an intentionally partial diagnostic set.",
    )
    args = parser.parse_args()
    if args.expected_episodes < 0:
        parser.error("--expected-episodes must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    records = collect_boundary_records(
        args.episodes_root.resolve(), expected_episodes=args.expected_episodes
    )
    write_boundary_records(records, args.output)
    total_frames = sum(int(record["total_frames"]) for record in records)
    print(
        f"wrote {len(records)} WA-01 boundary records ({total_frames} frames) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
