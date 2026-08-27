#!/usr/bin/env python
"""Recover exact WR05 stage boundaries and executable episode plans.

WR05 requires remembering which fixed open-shelf level initially holds each
condiment.  That home assignment stays hidden from the instruction.  The
seed-selected serving-plate sides are not visually inferable, so the bridge
does expose those sides in each episode's instruction for the downstream VLA.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "WR-05"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 26
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})
LEVEL_DISPLAY = {"shelf_lower": "lower", "shelf_upper": "upper"}
PLATE_DISPLAY = {"plate_left": "left", "plate_right": "right"}
CONDIMENTS = ("ketchup", "mustard")


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
    """Count ordinary action fields without nested aggregate duplication."""
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


def _validate_assignment(
    assignment: Mapping[str, Any], allowed: Mapping[str, str], *, name: str
) -> dict[str, str]:
    output = {item: str(assignment.get(item, "")) for item in CONDIMENTS}
    _require(set(output.values()) == set(allowed), f"invalid WR05 {name}: {output}")
    return output


def build_stage_names(
    home_assignment: Mapping[str, str], use_assignment: Mapping[str, str]
) -> list[str]:
    """Build the fixed WR05 plan, including the upper-shelf torso stage."""
    home = _validate_assignment(home_assignment, LEVEL_DISPLAY, name="home assignment")
    use = _validate_assignment(use_assignment, PLATE_DISPLAY, name="plate assignment")
    names = ["observe_condiment_shelf"]
    for item in CONDIMENTS:
        names.extend(
            [
                f"navigate_to_{item}_original_level",
                f"pick_{item}_from_original_level",
                f"stow_{item}_after_shelf_pick",
            ]
        )
        if home[item] == "shelf_upper":
            names.append(f"normalize_transport_posture_after_{item}_shelf_pick")
        names.extend(
            [
                f"navigate_{item}_to_plate_stage",
                f"place_{item}_at_{use[item]}",
                f"return_arm_after_{item}_plate_stage",
            ]
        )
    for item in CONDIMENTS:
        names.extend(
            [
                f"navigate_to_{item}_staged_item",
                f"pick_{item}_from_plate_stage",
                f"stow_{item}_for_return",
                f"navigate_{item}_to_remembered_level",
                f"return_{item}_to_original_level",
                f"return_arm_after_{item}_home_place",
            ]
        )
    _require(
        len(names) == EXPECTED_STAGE_COUNT,
        f"WR05 plan unexpectedly has {len(names)} stages",
    )
    return names


def build_task_instruction(use_assignment: Mapping[str, str]) -> str:
    use = _validate_assignment(use_assignment, PLATE_DISPLAY, name="plate assignment")
    ketchup_side = PLATE_DISPLAY[use["ketchup"]]
    mustard_side = PLATE_DISPLAY[use["mustard"]]
    return (
        "Observe and remember which open-shelf level initially holds the ketchup "
        "and mustard bottles. First stage the ketchup upright on the "
        f"{ketchup_side} side of the fixed serving plate, then stage the mustard "
        f"upright on the {mustard_side} side. Pick up the ketchup from the "
        f"{ketchup_side} side and return it upright to its original shelf level, "
        f"then pick up the mustard from the {mustard_side} side and return it "
        "upright to its original shelf level."
    )


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WR05 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def _expected_primitive(stage_name: str) -> str | None:
    if stage_name.startswith("navigate_") or stage_name == "observe_condiment_shelf":
        return None
    if stage_name.startswith("pick_"):
        return "pick"
    if stage_name.startswith("place_") or (
        stage_name.startswith("return_") and "_to_original_level" in stage_name
    ):
        return "place_at_accessible"
    if stage_name.startswith("normalize_transport_posture_"):
        return "set_torso_height"
    if stage_name.startswith("stow_") or stage_name.startswith("return_arm_"):
        return "return_arm_to_regular_pose"
    raise ValueError(f"no primitive contract for WR05 stage {stage_name!r}")


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

    raw_home = facts.get("home_assignment")
    raw_use = facts.get("use_assignment")
    _require(isinstance(raw_home, Mapping), f"seed {seed}: missing home assignment")
    _require(isinstance(raw_use, Mapping), f"seed {seed}: missing plate assignment")
    assert isinstance(raw_home, Mapping) and isinstance(raw_use, Mapping)
    home = _validate_assignment(raw_home, LEVEL_DISPLAY, name="home assignment")
    use = _validate_assignment(raw_use, PLATE_DISPLAY, name="plate assignment")
    _require(
        facts.get("transport_order") == list(CONDIMENTS),
        f"seed {seed}: unexpected transport order",
    )
    _require(
        expert_meta.get("home_assignment") == dict(raw_home)
        and expert_meta.get("use_assignment") == dict(raw_use),
        f"seed {seed}: expert assignments differ from resolved facts",
    )
    expected_case = (0 if home["ketchup"] == "shelf_upper" else 2) + (
        0 if use["ketchup"] == "plate_left" else 1
    )
    _require(facts.get("case_id") == expected_case, f"seed {seed}: case id mismatch")

    diagnostics = planner.get("final_diagnostics")
    _require(isinstance(diagnostics, Mapping), f"seed {seed}: missing diagnostics")
    assert isinstance(diagnostics, Mapping)
    home_correct = diagnostics.get("current_condiment_home_correct")
    relation = diagnostics.get("current_condiment_relation")
    _require(isinstance(home_correct, Mapping), f"seed {seed}: missing home diagnostics")
    _require(isinstance(relation, Mapping), f"seed {seed}: missing relation diagnostics")
    assert isinstance(home_correct, Mapping) and isinstance(relation, Mapping)
    expected_lower = next(item for item in CONDIMENTS if home[item] == "shelf_lower")
    expected_upper = next(item for item in CONDIMENTS if home[item] == "shelf_upper")
    _require(
        diagnostics.get("success") is True
        and diagnostics.get("terminal") is True
        and diagnostics.get("condiment_shelf_level_relation_correct") is True
        and diagnostics.get("condiments_in_return_shelf") is True
        and diagnostics.get("condiments_released") is True
        and diagnostics.get("ketchup_staged_at_plate") is True
        and diagnostics.get("mustard_staged_at_plate") is True
        and home_correct.get("ketchup") is True
        and home_correct.get("mustard") is True
        and relation.get("correct") is True
        and relation.get("valid") is True
        and relation.get("expected_lower") == expected_lower
        and relation.get("expected_upper") == expected_upper,
        f"seed {seed}: WR05 memory/return protocol validation failed",
    )

    episode_result = planner.get("episode_result")
    rollout = planner.get("rollout")
    _require(isinstance(episode_result, Mapping), f"seed {seed}: missing episode result")
    _require(isinstance(rollout, Mapping), f"seed {seed}: missing rollout")
    assert isinstance(episode_result, Mapping) and isinstance(rollout, Mapping)
    stages = rollout.get("stages")
    _require(
        isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expected {EXPECTED_STAGE_COUNT} stages",
    )
    assert isinstance(stages, list)
    _require(
        episode_result.get("success") is True
        and episode_result.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: planner episode result mismatch",
    )

    names = build_stage_names(home, use)
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
        expected_kind = (
            "navigate"
            if name == "observe_condiment_shelf" or name.startswith("navigate_")
            else "operate"
        )
        _require(
            stage.get("stage_kind") == expected_kind,
            f"seed {seed}: stage {index} kind mismatch for {name}",
        )
        details = stage.get("details")
        _require(isinstance(details, Mapping), f"seed {seed}: stage {index} details missing")
        assert isinstance(details, Mapping)
        expected_primitive = _expected_primitive(name)
        if expected_primitive is not None:
            _require(
                details.get("primitive") == expected_primitive,
                f"seed {seed}: stage {index} primitive mismatch for {name}",
            )
        steps, fields = count_stage_steps(stage, path=f"seed[{seed}].stage[{index}]")
        start_frame = cumulative
        cumulative += steps
        for key, value in fields.items():
            field_totals[key] += value
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

    # Keep the initial observation endpoint as memory-only context while the
    # executable subtask starts with the first bottle pickup. Then retain one
    # memory frame at the end of each of the eight semantic phases.
    phase_start_names = {
        "navigate_ketchup_to_plate_stage",
        "navigate_to_mustard_original_level",
        "navigate_mustard_to_plate_stage",
        "navigate_to_ketchup_staged_item",
        "navigate_ketchup_to_remembered_level",
        "navigate_to_mustard_staged_item",
        "navigate_mustard_to_remembered_level",
    }
    keyframes = [output_stages[0]["end_frame"] - 1]
    keyframes.extend(
        output_stages[index - 1]["end_frame"] - 1
        for index, stage in enumerate(output_stages)
        if stage["name"] in phase_start_names
    )
    keyframes.append(output_stages[-1]["end_frame"] - 1)

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
        "case_id": expected_case,
        "success": True,
        "expert_plan_version": EXPERT_PLAN_VERSION,
        "home_assignment": {item: LEVEL_DISPLAY[home[item]] for item in CONDIMENTS},
        "home_region_assignment": home,
        "use_assignment": {item: PLATE_DISPLAY[use[item]] for item in CONDIMENTS},
        "plate_region_assignment": use,
        "ketchup_original_level": LEVEL_DISPLAY[home["ketchup"]],
        "mustard_original_level": LEVEL_DISPLAY[home["mustard"]],
        "ketchup_plate_side": PLATE_DISPLAY[use["ketchup"]],
        "mustard_plate_side": PLATE_DISPLAY[use["mustard"]],
        "high_level_instruction": build_task_instruction(use),
        "total_frames": cumulative,
        "recorded_total_frames": recorded_total,
        "step_field_totals": field_totals,
        "source_episode_dir": episode_dir.name,
        "stages": output_stages,
        "keyframes": sorted(set(keyframes)),
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
    _require(len(seeds) == len(set(seeds)), "duplicate WR05 seeds discovered")
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
        f"wrote {len(records)} WR05 boundary records "
        f"({sum(int(record['total_frames']) for record in records)} frames, "
        f"{len({record['high_level_instruction'] for record in records})} task plans) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
