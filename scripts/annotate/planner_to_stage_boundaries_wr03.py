#!/usr/bin/env python
"""Recover exact WR03 stage boundaries from recorded planner summaries.

WR03 stores the can in either the lower or upper cabinet, closes both doors,
and later has to reopen the cabinet selected during storage.  The native
recorder persists exact action counts for all 20 expert-plan stages but not
their names.  This bridge reconstructs those names and records the physically
verified storage level used by the dynamic retrieval-door subtask label.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


TASK_ID = "WR-03"
EXPERT_PLAN_VERSION = "membench.expert_plan/v2"
EXPECTED_STAGE_COUNT = 20
EPISODE_DIR_PATTERN = re.compile(r"^seed_(\d{6})_attempt_00$")
STEP_KEYS = frozenset({"steps", "executed_steps", "release_settle_steps"})
NAVIGATE_STAGE_INDICES = frozenset({0, 1, 4, 7, 10, 14, 17})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def count_recorded_steps(
    value: Any, *, path: str = "details"
) -> tuple[int, dict[str, int]]:
    """Count only fields that the native recorder defines as action steps."""
    total = 0
    fields = {key: 0 for key in sorted(STEP_KEYS)}

    def visit(node: Any, node_path: str) -> None:
        nonlocal total
        if isinstance(node, Mapping):
            # Some WR03 navigation segments expose both an aggregate ``steps``
            # value and nested pre/rotate/post details carrying the same action
            # count.  Once a mapping declares its own aggregate, its descendants
            # are diagnostics rather than additional executed actions.
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
                            f"{child_path} must be a non-negative integer, got {child!r}"
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


def build_stage_names(storage_level: str, close_order: Sequence[str] = ("lower", "upper")) -> list[str]:
    """Rebuild the canonical 20-stage WR03 expert plan."""
    _require(
        storage_level in {"lower", "upper"},
        f"invalid WR03 storage level: {storage_level!r}",
    )
    order = tuple(str(value) for value in close_order)
    _require(len(order) == 2 and set(order) == {"lower", "upper"}, "invalid WR03 close order")
    names = ["observe_open_cabinets"]
    for index, level in enumerate(order, start=1):
        names.extend([f"navigate_to_door_{index}", f"close_door_{index}", f"return_after_door_{index}"])
    names.extend([
        "navigate_to_neutral_reset", "canonical_memory_reset", "latch_memory_handoff",
        "navigate_to_remembered_cabinet", "open_remembered_cabinet",
        "verify_remembered_cabinet", "return_after_opening_cabinet",
        "navigate_to_can", "retrieve_can",
        "return_with_can", "navigate_to_initial_table", "place_can_on_table",
        "verify_final_table_placement",
    ])
    return names


def _episode_seed(episode_dir: Path) -> int:
    match = EPISODE_DIR_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"unexpected WR03 episode directory: {episode_dir.name!r}")
    return int(match.group(1))


def _expert_storage_level(episode: Mapping[str, Any], *, seed: int) -> str:
    metadata = episode.get("metadata")
    _require(isinstance(metadata, Mapping), f"seed {seed}: missing episode metadata")
    assert isinstance(metadata, Mapping)
    expert_plan = metadata.get("expert_plan")
    _require(isinstance(expert_plan, Mapping), f"seed {seed}: missing expert plan")
    assert isinstance(expert_plan, Mapping)
    _require(
        expert_plan.get("version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: expert plan version mismatch",
    )
    _require(
        expert_plan.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expert plan stage count mismatch",
    )
    plan_metadata = expert_plan.get("metadata")
    _require(
        isinstance(plan_metadata, Mapping),
        f"seed {seed}: missing expert plan metadata",
    )
    assert isinstance(plan_metadata, Mapping)
    level = str(plan_metadata.get("stored_level", plan_metadata.get("demo_storage_level", "")))
    _require(level in {"lower", "upper"}, f"seed {seed}: invalid expert storage level")

    resolved = metadata.get("resolved_episode")
    _require(
        isinstance(resolved, Mapping), f"seed {seed}: missing resolved episode metadata"
    )
    assert isinstance(resolved, Mapping)
    facts = resolved.get("facts")
    _require(isinstance(facts, Mapping), f"seed {seed}: missing resolved facts")
    assert isinstance(facts, Mapping)
    _require(
        facts.get("stored_level", facts.get("demo_storage_level")) == level,
        f"seed {seed}: resolved and expert storage levels differ",
    )
    return level


def _observed_storage_level(planner: Mapping[str, Any], *, seed: int) -> str:
    diagnostics = planner.get("final_diagnostics")
    _require(
        isinstance(diagnostics, Mapping), f"seed {seed}: missing final diagnostics"
    )
    assert isinstance(diagnostics, Mapping)
    _require(diagnostics.get("success") is True, f"seed {seed}: final success is false")
    storage = diagnostics.get("storage")
    retrieval = diagnostics.get("retrieval")
    _require(isinstance(storage, Mapping), f"seed {seed}: missing storage diagnostics")
    _require(
        isinstance(retrieval, Mapping), f"seed {seed}: missing retrieval diagnostics"
    )
    assert isinstance(storage, Mapping)
    assert isinstance(retrieval, Mapping)
    level = str(storage.get("stored_level", ""))
    _require(level in {"lower", "upper"}, f"seed {seed}: invalid observed level")
    _require(
        retrieval.get("opened_level") == level,
        f"seed {seed}: opened level does not match stored level",
    )
    _require(
        retrieval.get("wrong_level_opened") is False,
        f"seed {seed}: wrong retrieval level was opened",
    )
    _require(
        retrieval.get("retrieved_item_id") == "red_can",
        f"seed {seed}: red can was not retrieved",
    )
    return level


def build_boundary_record(episode_dir: Path) -> dict[str, Any]:
    seed = _episode_seed(episode_dir)
    episode = _read_json(episode_dir / "episode.json")
    planner = _read_json(episode_dir / "planner.json")
    _require(episode.get("accepted") is True, f"seed {seed}: episode is not accepted")
    _require(episode.get("planner_pass") is True, f"seed {seed}: planner did not pass")
    _require(
        episode.get("metadata", {}).get("task_id") == TASK_ID,
        f"seed {seed}: episode task is not {TASK_ID}",
    )
    _require(
        episode.get("metadata", {}).get("seed") == seed,
        f"seed {seed}: episode metadata seed mismatch",
    )
    _require(planner.get("task_id") == TASK_ID, f"seed {seed}: planner task mismatch")
    _require(
        planner.get("expert_plan_version") == EXPERT_PLAN_VERSION,
        f"seed {seed}: planner version mismatch",
    )
    _require(planner.get("pass") is True, f"seed {seed}: planner pass is false")

    episode_result = planner.get("episode_result")
    _require(
        isinstance(episode_result, Mapping), f"seed {seed}: missing episode result"
    )
    assert isinstance(episode_result, Mapping)
    _require(episode_result.get("success") is True, f"seed {seed}: episode result failed")
    _require(
        episode_result.get("stage_count") == EXPECTED_STAGE_COUNT,
        f"seed {seed}: planner stage count mismatch",
    )

    stages = planner.get("rollout", {}).get("stages")
    _require(
        isinstance(stages, list) and len(stages) == EXPECTED_STAGE_COUNT,
        f"seed {seed}: expected {EXPECTED_STAGE_COUNT} planner stages",
    )
    assert isinstance(stages, list)
    recorded_total = episode.get("step_count")
    _require(
        isinstance(recorded_total, int)
        and not isinstance(recorded_total, bool)
        and recorded_total > 0,
        f"seed {seed}: invalid episode step count",
    )
    assert isinstance(recorded_total, int)

    expert_level = _expert_storage_level(episode, seed=seed)
    observed_level = _observed_storage_level(planner, seed=seed)
    _require(
        observed_level == expert_level,
        f"seed {seed}: observed level {observed_level!r} != expert level {expert_level!r}",
    )
    facts = (episode.get("metadata") or {}).get("resolved_episode", {}).get("facts", {})
    close_order = tuple(str(value) for value in facts.get("close_order", ("lower", "upper")))
    stage_names = build_stage_names(observed_level, close_order)

    cumulative = 0
    output_stages: list[dict[str, Any]] = []
    field_totals = {key: 0 for key in sorted(STEP_KEYS)}
    for index, (stage, name) in enumerate(zip(stages, stage_names)):
        _require(isinstance(stage, Mapping), f"seed {seed}: stage {index} is invalid")
        assert isinstance(stage, Mapping)
        _require(
            stage.get("stage_index") == index,
            f"seed {seed}: stage index mismatch at {index}",
        )
        _require(stage.get("success") is True, f"seed {seed}: stage {index} failed")
        expected_kind = "navigate" if index in NAVIGATE_STAGE_INDICES else "operate"
        _require(
            stage.get("stage_kind") == expected_kind,
            f"seed {seed}: stage {index} kind mismatch",
        )
        details = stage.get("details")
        _require(
            isinstance(details, Mapping), f"seed {seed}: stage {index} has no details"
        )
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
                "kind": expected_kind,
                "steps": steps,
                "step_fields": fields,
                "start_frame": start_frame,
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
        "stored_level": observed_level,
        "close_order": list(close_order),
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
            f"expected {expected_episodes} WR03 episodes, found {len(episode_dirs)}",
        )
    records = [build_boundary_record(path) for path in episode_dirs]
    seeds = [int(record["seed"]) for record in records]
    _require(len(seeds) == len(set(seeds)), "duplicate WR03 seeds discovered")
    if expected_episodes > 0:
        _require(
            seeds == list(range(expected_episodes)),
            f"expected contiguous seeds 0..{expected_episodes - 1}, got {seeds}",
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
        f"wrote {len(records)} WR03 boundary records "
        f"({sum(int(record['total_frames']) for record in records)} frames) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
