#!/usr/bin/env python
"""Recover exact WA02 boundaries and task-semantic episode instructions.

WA02's recorded expert happens to choose one starting cup and one service slot
per seed.  Those choices are demonstration details, not policy-visible task
requirements: the first held cup becomes the hot-water cup, the remaining cup
becomes the cold-water cup, and either free dining-table position may be used.
Only the requested final temperature belongs in the episode instruction.  The
final left/right pickup label is recovered from what physically happened in
the demonstration, preserving the actual memory challenge.

The recorder-owned step counts are not uniform across WA02 primitives:

* navigation stores an aggregate ``steps`` value for terminal settle segments;
* sink-exit lift stores Base actions in ``base_translation_steps`` and arm
  actions in ``movement_attempts``; ``movement`` repeats the final attempt.

The primitive-specific counting below is therefore intentionally strict and
must reproduce every episode's recorded frame count exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "WA-02"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 26
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})
START_SIDE_BY_CUP = {"cup_A": "left", "cup_B": "right"}
START_RELATION_BY_CUP = {"cup_A": "nearer the sink", "cup_B": "farther from the sink"}
SERVICE_SIDE_BY_SLOT = {"slot_1": "right", "slot_2": "left"}
NAVIGATE_STAGE_INDICES = frozenset({0, 3, 7, 11, 14, 18, 23})


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
    """Count ordinary action fields, honoring a node's aggregate count."""
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
    """Count manual-path segments plus the task-local station torso motion."""
    segments = details.get("segments")
    _require(isinstance(segments, list), f"{path}.segments must be a list")
    assert isinstance(segments, list)
    total = 0
    fields = {key: 0 for key in sorted(STEP_KEYS)}
    for index, segment in enumerate(segments):
        _require(
            isinstance(segment, Mapping),
            f"{path}.segments[{index}] must be an object",
        )
        steps, counts = count_recorded_steps(
            segment, path=f"{path}.segments[{index}]"
        )
        total += steps
        for key, count in counts.items():
            fields[key] += count

    torso = details.get("station_torso")
    _require(isinstance(torso, Mapping), f"{path}.station_torso must be an object")
    assert isinstance(torso, Mapping)
    torso_steps = _non_negative_int(
        torso.get("steps"), f"{path}.station_torso.steps"
    )
    total += torso_steps
    fields["steps"] += torso_steps
    return total, {**fields, "base_translation_steps": 0}


def count_sink_exit_lift_steps(
    details: Mapping[str, Any], *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Count Base translation and each executed arm correction exactly once."""
    base_steps = _non_negative_int(
        details.get("base_translation_steps"),
        f"{path}.base_translation_steps",
    )
    attempts = details.get("movement_attempts")
    _require(
        isinstance(attempts, list) and bool(attempts),
        f"{path}.movement_attempts must be a non-empty list",
    )
    assert isinstance(attempts, list)
    total = base_steps
    fields = {key: 0 for key in sorted(STEP_KEYS)}
    last_steps = None
    for index, attempt in enumerate(attempts):
        _require(
            isinstance(attempt, Mapping),
            f"{path}.movement_attempts[{index}] must be an object",
        )
        steps, counts = count_recorded_steps(
            attempt, path=f"{path}.movement_attempts[{index}]"
        )
        total += steps
        last_steps = steps
        for key, count in counts.items():
            fields[key] += count

    movement = details.get("movement")
    _require(isinstance(movement, Mapping), f"{path}.movement must be an object")
    assert isinstance(movement, Mapping)
    repeated_steps, _ = count_recorded_steps(movement, path=f"{path}.movement")
    _require(
        repeated_steps == last_steps,
        f"{path}.movement is not the final movement_attempt",
    )
    return total, {**fields, "base_translation_steps": base_steps}


def count_stage_steps(
    details: Mapping[str, Any], *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    primitive = str(details.get("primitive", ""))
    if primitive == "manual_base_path_navigate":
        return count_navigation_steps(details, path=path)
    if primitive == "lift_held_cup_after_sink_exit":
        return count_sink_exit_lift_steps(details, path=path)
    steps, fields = count_recorded_steps(details, path=path)
    return steps, {**fields, "base_translation_steps": 0}


def build_stage_names(
    hot_cup: str,
    cold_cup: str,
    service_slot_by_cup: Mapping[str, str],
    requested_cup: str,
) -> list[str]:
    _require(
        {hot_cup, cold_cup} == set(START_SIDE_BY_CUP),
        "hot and cold bindings must be a permutation of cup_A/cup_B",
    )
    slots = {str(key): str(value) for key, value in service_slot_by_cup.items()}
    _require(
        set(slots) == set(START_SIDE_BY_CUP)
        and set(slots.values()) == set(SERVICE_SIDE_BY_SLOT),
        "service slot assignment must be a cup/slot permutation",
    )
    _require(requested_cup in {hot_cup, cold_cup}, "requested cup is invalid")
    names: list[str] = []
    for ordinal, (temperature, cup) in enumerate(
        (("hot", hot_cup), ("cold", cold_cup)), start=1
    ):
        label = "first" if ordinal == 1 else "second"
        slot = slots[cup]
        names.extend(
            [
                f"navigate_pick_{label}_{cup}",
                f"pick_{label}_{cup}",
                f"return_to_regular_after_pick_{label}_{cup}",
                f"navigate_fill_{label}_{temperature}_{cup}",
                f"sensor_fill_{label}_{temperature}_{cup}",
                f"return_after_fill_{label}_{temperature}_{cup}",
                f"lift_after_sink_exit_{label}_{temperature}_{cup}",
                f"navigate_service_{label}_{cup}_{slot}",
                f"place_{label}_{cup}_{slot}",
                f"verify_and_record_{label}_{cup}_{slot}",
                f"return_after_service_{label}_{cup}",
            ]
        )
    names.extend(
        [
            "verify_both_cups_staged_before_final_pick",
            f"navigate_final_pick_{requested_cup}",
            f"physical_final_pick_{requested_cup}",
            f"record_final_take_{requested_cup}",
        ]
    )
    return names


def build_task_instruction(requested_temperature: str) -> str:
    _require(
        requested_temperature in {"hot", "cold"},
        f"invalid requested temperature {requested_temperature!r}",
    )
    return (
        "Pick up the first cup from the starting counter and fill it with hot "
        "water. Place it on the dining table to the left of the sink. Return "
        "to the starting counter, pick up the remaining cup, fill it with cold "
        "water, and place it on the same dining table. After both cups are "
        f"placed, pick up the {requested_temperature}-water cup."
    )


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WA02 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def _resolved_facts(episode: Mapping[str, Any], *, seed: int) -> Mapping[str, Any]:
    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing metadata")
    assert isinstance(metadata, Mapping)
    _require(metadata.get("task_id") == TASK_ID, f"seed {seed}: task mismatch")
    _require(metadata.get("seed") == seed, f"seed {seed}: seed mismatch")
    resolved = metadata.get("resolved_episode")
    _require(isinstance(resolved, Mapping), f"seed {seed}: missing resolved episode")
    assert isinstance(resolved, Mapping)
    facts = resolved.get("facts")
    _require(isinstance(facts, Mapping), f"seed {seed}: missing resolved facts")
    assert isinstance(facts, Mapping)
    return facts


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

    facts = _resolved_facts(episode, seed=seed)
    hot_cup = str(facts.get("hot_cup_id", ""))
    cold_cup = str(facts.get("cold_cup_id", ""))
    requested_cup = str(facts.get("requested_cup_id", ""))
    requested_temperature = str(facts.get("requested_temperature", ""))
    fill_order = list(facts.get("fill_temperature_order", ()))
    slots_raw = facts.get("service_slot_by_cup")
    _require(isinstance(slots_raw, Mapping), f"seed {seed}: missing slot mapping")
    assert isinstance(slots_raw, Mapping)
    slots = {str(key): str(value) for key, value in slots_raw.items()}
    _require(fill_order == ["hot", "cold"], f"seed {seed}: fill order is not hot/cold")
    _require(
        requested_temperature in {"hot", "cold"},
        f"seed {seed}: requested temperature is invalid",
    )
    expected_requested = hot_cup if requested_temperature == "hot" else cold_cup
    _require(
        requested_cup == expected_requested,
        f"seed {seed}: requested cup/temperature mismatch",
    )

    metadata = episode["metadata"]
    assert isinstance(metadata, Mapping)
    expert = metadata.get("expert_plan")
    _require(isinstance(expert, Mapping), f"seed {seed}: missing expert plan")
    assert isinstance(expert, Mapping)
    _require(expert.get("version") == EXPERT_PLAN_VERSION, f"seed {seed}: expert version mismatch")
    _require(expert.get("stage_count") == EXPECTED_STAGE_COUNT, f"seed {seed}: expert stage count mismatch")
    expert_meta = expert.get("metadata")
    _require(isinstance(expert_meta, Mapping), f"seed {seed}: missing expert metadata")
    assert isinstance(expert_meta, Mapping)
    _require(
        list(expert_meta.get("fill_temperature_order", ())) == fill_order
        and expert_meta.get("requested_cup_id") == requested_cup
        and dict(expert_meta.get("service_slot_by_cup", {})) == slots,
        f"seed {seed}: expert metadata differs from resolved facts",
    )

    diagnostics = planner.get("final_diagnostics")
    _require(isinstance(diagnostics, Mapping), f"seed {seed}: missing diagnostics")
    assert isinstance(diagnostics, Mapping)
    _require(diagnostics.get("success") is True, f"seed {seed}: final success is false")
    oracle = diagnostics.get("oracle")
    fill = diagnostics.get("fill")
    service = diagnostics.get("service")
    selection = diagnostics.get("selection")
    _require(isinstance(oracle, Mapping), f"seed {seed}: missing oracle diagnostics")
    _require(isinstance(fill, Mapping), f"seed {seed}: missing fill diagnostics")
    _require(isinstance(service, Mapping), f"seed {seed}: missing service diagnostics")
    _require(isinstance(selection, Mapping), f"seed {seed}: missing selection diagnostics")
    assert isinstance(oracle, Mapping) and isinstance(fill, Mapping)
    assert isinstance(service, Mapping) and isinstance(selection, Mapping)
    _require(
        oracle.get("hot_cup_id") == hot_cup
        and oracle.get("cold_cup_id") == cold_cup
        and oracle.get("requested_cup_id") == requested_cup
        and dict(oracle.get("service_slot_by_cup", {})) == slots,
        f"seed {seed}: oracle assignment mismatch",
    )
    _require(
        dict(fill.get("recorded_temperature_by_cup", {}))
        == {hot_cup: "hot", cold_cup: "cold"},
        f"seed {seed}: observed fill assignment mismatch",
    )
    _require(
        dict(service.get("recorded_slot_by_cup", {})) == slots,
        f"seed {seed}: observed service assignment mismatch",
    )
    _require(
        selection.get("final_taken_cup_id") == requested_cup,
        f"seed {seed}: final selection mismatch",
    )

    stage_names = build_stage_names(hot_cup, cold_cup, slots, requested_cup)
    stages = planner.get("rollout", {}).get("stages")
    _require(
        isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expected {EXPECTED_STAGE_COUNT} stages",
    )
    assert isinstance(stages, list)
    episode_result = planner.get("episode_result")
    _require(isinstance(episode_result, Mapping), f"seed {seed}: missing episode result")
    assert isinstance(episode_result, Mapping)
    _require(
        episode_result.get("success") is True
        and episode_result.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: planner episode result mismatch",
    )

    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    field_totals = {
        "executed_steps": 0,
        "release_settle_steps": 0,
        "steps": 0,
        "base_translation_steps": 0,
    }
    keyframes: list[int] = []
    keyframe_last_stages = {2, 6, 10, 13, 17, 21}
    for index, (stage, name) in enumerate(zip(stages, stage_names)):
        _require(isinstance(stage, Mapping), f"seed {seed}: invalid stage {index}")
        assert isinstance(stage, Mapping)
        _require(stage.get("stage_index") == index, f"seed {seed}: stage index mismatch")
        _require(stage.get("success") is True, f"seed {seed}: stage {index} failed")
        expected_kind = "navigate" if index in NAVIGATE_STAGE_INDICES else "operate"
        _require(
            stage.get("stage_kind") == expected_kind,
            f"seed {seed}: stage {index} kind mismatch",
        )
        details = stage.get("details")
        _require(isinstance(details, Mapping), f"seed {seed}: stage {index} lacks details")
        assert isinstance(details, Mapping)
        steps, counts = count_stage_steps(
            details, path=f"seed[{seed}].stage[{index}].details"
        )
        start_frame = cumulative
        cumulative += steps
        for key, count in counts.items():
            field_totals[key] += count
        output_stages.append(
            {
                "index": index,
                "name": name,
                "kind": expected_kind,
                "primitive": str(details.get("primitive", "")),
                "steps": steps,
                "step_fields": counts,
                "start_frame": start_frame,
                "end_frame": cumulative,
            }
        )
        if index in keyframe_last_stages:
            _require(steps > 0, f"seed {seed}: keyframe stage {index} is empty")
            keyframes.append(cumulative - 1)
        if index == 22:
            keyframes.append(start_frame)

    recorded_total = _non_negative_int(episode.get("step_count"), f"seed[{seed}].step_count")
    _require(recorded_total > 0, f"seed {seed}: episode is empty")
    _require(
        cumulative == recorded_total,
        f"seed {seed}: recovered {cumulative} != recorded {recorded_total}; fields={field_totals}",
    )

    hot_service_side = SERVICE_SIDE_BY_SLOT[slots[hot_cup]]
    cold_service_side = SERVICE_SIDE_BY_SLOT[slots[cold_cup]]
    requested_service_side = SERVICE_SIDE_BY_SLOT[slots[requested_cup]]
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "hot_cup_id": hot_cup,
        "cold_cup_id": cold_cup,
        "requested_cup_id": requested_cup,
        "requested_temperature": requested_temperature,
        "service_slot_by_cup": slots,
        "hot_start_side": START_SIDE_BY_CUP[hot_cup],
        "cold_start_side": START_SIDE_BY_CUP[cold_cup],
        "hot_service_side": hot_service_side,
        "cold_service_side": cold_service_side,
        "requested_service_side": requested_service_side,
        "slot_swap": slots["cup_A"] == "slot_2",
        "high_level_instruction": build_task_instruction(requested_temperature),
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "keyframes": keyframes,
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
    instructions = len({str(record["high_level_instruction"]) for record in records})
    print(
        f"wrote {len(records)} WA02 boundary records ({total_frames} frames, "
        f"{instructions} task plans) to {args.output}"
    )


if __name__ == "__main__":
    main()
