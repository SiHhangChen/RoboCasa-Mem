#!/usr/bin/env python
"""Recover exact WA-05 stage boundaries from recorded planner summaries.

The native WA-05 planner records action counts for every primitive except the
compact answer-button diagnostic.  We count all other stages with the
aggregate-aware WA02 counter and recover the button stage as the exact
remainder of ``episode.json.step_count``.  This preserves an exact partition
of every recorded frame while keeping the hot/cold choice as a final-stage
semantic label only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.annotate.planner_to_stage_boundaries_wa02 import count_stage_steps


TASK_ID = "WA-05"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 13
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
TEMPERATURES = frozenset({"hot", "cold"})
NAVIGATE_STAGE_INDICES = frozenset({0, 3, 7})
STEP_KEYS = ("executed_steps", "release_settle_steps", "steps")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def non_negative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer, got {value!r}")
    return int(value)


def episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WA-05 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def resolved_facts(episode: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    metadata = episode.get("metadata")
    require(isinstance(metadata, Mapping), f"seed {seed}: missing metadata")
    require(metadata.get("task_id") == TASK_ID, f"seed {seed}: task mismatch")
    require(metadata.get("seed") == seed, f"seed {seed}: seed mismatch")
    resolved = metadata.get("resolved_episode")
    require(isinstance(resolved, Mapping), f"seed {seed}: missing resolved_episode")
    facts = resolved.get("facts")
    require(isinstance(facts, Mapping), f"seed {seed}: missing resolved facts")
    return facts


def expert_metadata(episode: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    metadata = episode.get("metadata")
    require(isinstance(metadata, Mapping), f"seed {seed}: missing metadata")
    expert = metadata.get("expert_plan")
    require(isinstance(expert, Mapping), f"seed {seed}: missing expert_plan")
    require(expert.get("version") == EXPERT_PLAN_VERSION,
            f"seed {seed}: expert-plan version mismatch")
    require(expert.get("stage_count") == EXPECTED_STAGE_COUNT,
            f"seed {seed}: expert-plan stage count mismatch")
    value = expert.get("metadata")
    require(isinstance(value, Mapping), f"seed {seed}: missing expert-plan metadata")
    return value


def stage_names(temperature: str) -> list[str]:
    require(temperature in TEMPERATURES, f"invalid temperature {temperature!r}")
    return [
        "navigate_pick_cup_A",
        "pick_cup_A",
        "return_after_pick_cup_A",
        f"navigate_fill_{temperature}_cup_A",
        f"fill_{temperature}_cup_A",
        f"return_after_fill_{temperature}_cup_A",
        "lift_after_sink_exit_cup_A",
        "navigate_service_slot_1",
        "place_cup_A_slot_1",
        "verify_staging_cup_A_slot_1",
        "return_after_service_cup_A",
        f"press_{temperature}_answer_button",
        f"return_after_{temperature}_answer",
    ]


def validate_diagnostics(
    planner: Mapping[str, Any], seed: int, cup_id: str, temperature: str
) -> None:
    diagnostics = planner.get("final_diagnostics")
    require(isinstance(diagnostics, Mapping), f"seed {seed}: missing final_diagnostics")
    require(diagnostics.get("success") is True, f"seed {seed}: final diagnostics failed")

    oracle = diagnostics.get("oracle")
    require(isinstance(oracle, Mapping), f"seed {seed}: missing oracle diagnostics")
    require(
        oracle.get("expert_fill_temperature") == temperature
        and oracle.get("filled_temperature") == temperature
        and oracle.get("pressed_temperature") == temperature,
        f"seed {seed}: oracle temperature mismatch",
    )
    fill = diagnostics.get("fill")
    require(isinstance(fill, Mapping), f"seed {seed}: missing fill diagnostics")
    require(fill.get("recorded_temperature") == temperature,
            f"seed {seed}: recorded fill temperature mismatch")
    cup_temperatures = fill.get("cup_temperatures")
    require(isinstance(cup_temperatures, Mapping), f"seed {seed}: missing cup temperatures")
    require(cup_temperatures.get(cup_id) == temperature,
            f"seed {seed}: filled cup temperature mismatch")
    service = diagnostics.get("service")
    require(isinstance(service, Mapping), f"seed {seed}: missing service diagnostics")
    require(service.get("physically_staged") is True and service.get("staged_before_answer") is True,
            f"seed {seed}: cup was not staged before answer")
    physical_events = diagnostics.get("physical_events")
    require(isinstance(physical_events, Mapping), f"seed {seed}: missing physical events")
    answer_event = physical_events.get("answer_button")
    require(isinstance(answer_event, Mapping), f"seed {seed}: missing answer-button event")
    require(answer_event.get("applied") is True and answer_event.get("temperature") == temperature,
            f"seed {seed}: answer-button event mismatch")


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = episode_seed(episode_dir)
    episode = read_json(episode_dir / "episode.json")
    planner = read_json(episode_dir / "planner.json")
    require(episode.get("accepted") is True, f"seed {seed}: episode not accepted")
    require(episode.get("planner_pass") is True, f"seed {seed}: planner did not pass")
    require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task mismatch")
    require(planner.get("expert_plan_version") == EXPERT_PLAN_VERSION,
            f"seed {seed}: planner version mismatch")
    require(planner.get("pass") is True, f"seed {seed}: planner pass is false")

    facts = resolved_facts(episode, seed)
    cup_id = str(facts.get("cup_id", ""))
    service_slot = str(facts.get("service_slot", ""))
    temperature = str(facts.get("expert_fill_temperature", "")).lower()
    require(cup_id == "cup_A", f"seed {seed}: expected cup_A, got {cup_id!r}")
    require(service_slot == "slot_1", f"seed {seed}: expected slot_1, got {service_slot!r}")
    require(temperature in TEMPERATURES, f"seed {seed}: invalid temperature {temperature!r}")
    expert = expert_metadata(episode, seed)
    require(
        expert.get("cup_id") == cup_id
        and expert.get("service_slot") == service_slot
        and expert.get("expert_fill_temperature") == temperature,
        f"seed {seed}: expert metadata differs from resolved facts",
    )
    validate_diagnostics(planner, seed, cup_id, temperature)

    stages = planner.get("rollout", {}).get("stages")
    require(isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
            f"seed {seed}: expected {EXPECTED_STAGE_COUNT} rollout stages")
    result = planner.get("episode_result")
    require(isinstance(result, Mapping), f"seed {seed}: missing episode_result")
    require(result.get("success") is True and result.get("stage_count") == EXPECTED_STAGE_COUNT,
            f"seed {seed}: invalid episode_result")

    names = stage_names(temperature)
    primitive_by_index = {
        0: "manual_base_path_navigate", 1: "pick", 2: "return_arm_to_regular_pose",
        3: "manual_base_path_navigate", 4: "fill_held_cup_under_faucet",
        5: "return_arm_holding_cup", 6: "lift_held_cup_after_sink_exit",
        7: "manual_base_path_navigate", 8: "place_cup_on_service_slot",
        9: "record_verified_staging", 10: "return_arm_to_regular_pose",
        11: "press_temperature_answer_button", 12: "return_arm_to_regular_pose",
    }
    counts: list[tuple[int, dict[str, int]]] = []
    for index, stage in enumerate(stages):
        require(isinstance(stage, Mapping), f"seed {seed}: invalid stage {index}")
        require(stage.get("stage_index") == index and stage.get("success") is True,
                f"seed {seed}: stage {index} failed or has wrong index")
        expected_kind = "navigate" if index in NAVIGATE_STAGE_INDICES else "operate"
        require(stage.get("stage_kind") == expected_kind,
                f"seed {seed}: stage {index} kind mismatch")
        details = stage.get("details")
        require(isinstance(details, Mapping), f"seed {seed}: stage {index} missing details")
        require(details.get("primitive") == primitive_by_index[index],
                f"seed {seed}: stage {index} primitive mismatch")
        if index in {4, 11}:
            require(details.get("temperature") == temperature,
                    f"seed {seed}: stage {index} temperature mismatch")
        if index in {2, 5, 6, 10, 12}:
            require(details.get("stage") == names[index],
                    f"seed {seed}: stage {index} diagnostic-name mismatch")
        if index == 11:
            counts.append((0, {key: 0 for key in STEP_KEYS}))
        else:
            steps, fields = count_stage_steps(
                details, path=f"seed[{seed}].stage[{index}].details"
            )
            counts.append((steps, fields))

    recorded_total = non_negative_int(episode.get("step_count"), f"seed[{seed}].step_count")
    require(recorded_total > 0, f"seed {seed}: empty episode")
    known_total = sum(steps for index, (steps, _) in enumerate(counts) if index != 11)
    inferred_button_steps = recorded_total - known_total
    require(inferred_button_steps > 0,
            f"seed {seed}: inferred button stage is not positive ({inferred_button_steps})")

    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    field_totals = {key: 0 for key in STEP_KEYS}
    base_translation_steps = 0
    for index, (stage, name) in enumerate(zip(stages, names)):
        if index == 11:
            steps = inferred_button_steps
            fields = {key: 0 for key in STEP_KEYS}
            inferred = True
        else:
            steps, fields = counts[index]
            inferred = False
        start_frame = cumulative
        cumulative += steps
        for key, value in fields.items():
            if key in field_totals:
                field_totals[key] += value
        base_translation_steps += int(fields.get("base_translation_steps", 0))
        output_stages.append({
            "index": index,
            "name": name,
            "kind": "navigate" if index in NAVIGATE_STAGE_INDICES else "operate",
            "primitive": str(stage["details"].get("primitive", "")),
            "steps": steps,
            "step_fields": fields,
            "start_frame": start_frame,
            "end_frame": cumulative,
            **({"inferred_from_episode_remainder": True} if inferred else {}),
        })
    require(cumulative == recorded_total,
            f"seed {seed}: recovered {cumulative} != recorded {recorded_total}")

    keyframes = [
        output_stages[2]["end_frame"] - 1,
        output_stages[6]["end_frame"] - 1,
        output_stages[10]["end_frame"] - 1,
        output_stages[11]["start_frame"],
    ]
    require(all(0 <= int(frame) < recorded_total for frame in keyframes),
            f"seed {seed}: keyframe outside episode: {keyframes}")
    return {
        "task_id": TASK_ID,
        "seed": seed,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "cup_id": cup_id,
        "service_slot": service_slot,
        "expert_fill_temperature": temperature,
        "high_level_instruction": "Fill the cup with either hot or cold water, place it in the service area, then press the matching button.",
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": {**field_totals, "base_translation_steps": base_translation_steps},
        "inferred_button_steps": inferred_button_steps,
        "source_episode_dir": episode_dir.name,
        "keyframes": keyframes,
        "stages": output_stages,
    }


def collect_boundary_records(episodes_root: Path, expected_episodes: int) -> list[dict[str, Any]]:
    episode_dirs = sorted(
        path for path in episodes_root.glob("seed_*_attempt_00")
        if path.is_dir() and EPISODE_DIR_PATTERN.fullmatch(path.name)
    )
    if expected_episodes > 0:
        require(len(episode_dirs) == expected_episodes,
                f"expected {expected_episodes} episodes, found {len(episode_dirs)}")
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    require(len(seeds) == len(set(seeds)), "duplicate seeds discovered")
    if expected_episodes > 0:
        require(seeds == list(range(expected_episodes)),
                f"expected seeds 0..{expected_episodes - 1}, got {seeds}")
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
    records = collect_boundary_records(args.episodes_root.resolve(), args.expected_episodes)
    write_boundary_records(records, args.output)
    temperatures = {str(record["expert_fill_temperature"]) for record in records}
    print(
        f"wrote {len(records)} WA05 boundary records "
        f"({sum(int(record['total_frames']) for record in records)} frames; "
        f"temperatures={sorted(temperatures)}) to {args.output}"
    )


if __name__ == "__main__":
    main()
