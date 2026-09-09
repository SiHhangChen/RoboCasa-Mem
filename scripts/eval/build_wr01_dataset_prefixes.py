#!/usr/bin/env python3
"""Freeze WR-01 LeRobot demonstrations before the retrieval-door subtask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.canonical_stage import resolve_suite_episode  # noqa: E402
from bench.evaluation.expert_prefix import (  # noqa: E402
    ExpertPrefixArtifact,
    load_expert_prefix,
    write_expert_prefix,
)
from bench.evaluation.stage_suite import load_stage_eval_suite  # noqa: E402
from bench.runtime.data_contract import ACTION_DIM  # noqa: E402
from bench.tasks import register_builtin_task_plugins  # noqa: E402


DEFAULT_DATASET_ROOT = Path(
    "/data/shared_workspace/chensihang/dataset/membench/wr01_200seeds_v061"
)
DEFAULT_SUBTASK_ROOT = Path(
    "/data/shared_workspace/chensihang/dataset/memer/pi05/"
    "wr01_200seeds_v061_subtasks_v2"
)
BOUNDARY_STAGE = "close_lower_cabinet"
TARGET_SUBTASK_BY_LEVEL = {"upper": 6, "lower": 7}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-suite", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--subtask-root", type=Path, default=DEFAULT_SUBTASK_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stage_suite = load_stage_eval_suite(args.stage_suite)
    if stage_suite.task_id != "WR-01":
        raise ValueError(f"expected a WR-01 stage suite, got {stage_suite.task_id!r}")
    if "open_cabinet" not in stage_suite.stage_ids:
        raise ValueError("WR-01 dataset prefix suite must target open_cabinet")

    register_builtin_task_plugins()
    output_root = (
        stage_suite.expert_prefix_root
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    seeds = tuple(args.seeds) if args.seeds else stage_suite.base_suite.seeds
    episodes = _episode_index_by_seed(args.dataset_root)

    for seed in seeds:
        destination = output_root / f"seed_{seed:06d}" / "manifest.json"
        if destination.is_file() and not args.overwrite:
            artifact = load_expert_prefix(output_root, seed)
            print(
                f"seed={seed} status=existing actions={artifact.actions.shape[0]} "
                f"boundary={artifact.stage_boundary_steps[BOUNDARY_STAGE]}"
            )
            continue
        try:
            episode_index = episodes[int(seed)]
        except KeyError as exc:
            raise ValueError(f"dataset has no episode for seed {seed}") from exc
        resolved = resolve_suite_episode(stage_suite.base_suite, seed)
        target_level = str(resolved.facts["target_level"])
        target_subtask = TARGET_SUBTASK_BY_LEVEL[target_level]
        actions = _episode_actions(args.dataset_root, episode_index)
        subtasks = _episode_subtasks(args.subtask_root, episode_index)
        if actions.shape[0] != subtasks.shape[0]:
            raise ValueError(
                f"seed {seed}: action/subtask length mismatch "
                f"{actions.shape[0]} != {subtasks.shape[0]}"
            )
        handoff_step = handoff_step_for_subtask(subtasks, target_subtask)
        prefix_actions = np.ascontiguousarray(actions[:handoff_step], dtype=np.float32)
        artifact = ExpertPrefixArtifact(
            task_id="WR-01",
            seed=seed,
            episode_identity=resolved.identity,
            expert_plan_version="membench.dataset_prefix/wr01-v1",
            stage_boundary_steps={BOUNDARY_STAGE: handoff_step},
            actions=prefix_actions,
            success=True,
            stage_order=(BOUNDARY_STAGE,),
            provenance={
                "source": "lerobot_dataset",
                "dataset_root": str(args.dataset_root.expanduser().resolve()),
                "subtask_root": str(args.subtask_root.expanduser().resolve()),
                "episode_index": episode_index,
                "target_level": target_level,
                "target_subtask_index": target_subtask,
                "handoff_semantics": "before_retrieval_door_subtask",
            },
        )
        write_expert_prefix(output_root, artifact)
        print(
            f"seed={seed} status=written episode={episode_index} "
            f"target_level={target_level} target_subtask={target_subtask} "
            f"prefix_actions={handoff_step}"
        )
    print(f"prefix_root={output_root}")
    print(f"completed={len(seeds)}")
    return 0


def handoff_step_for_subtask(subtasks: np.ndarray, target_subtask: int) -> int:
    values = np.asarray(subtasks, dtype=np.int64).reshape(-1)
    matches = np.flatnonzero(values == int(target_subtask))
    if matches.size == 0:
        raise ValueError(f"target subtask {target_subtask} is absent")
    handoff_step = int(matches[0])
    if handoff_step <= 0:
        raise ValueError("retrieval-door subtask cannot start at frame zero")
    return handoff_step


def _episode_index_by_seed(dataset_root: Path) -> dict[int, int]:
    path = dataset_root.expanduser().resolve() / "meta/seeds.jsonl"
    result: dict[int, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record: dict[str, Any] = json.loads(line)
            seed = int(record["seed"])
            episode_index = int(record["episode_index"])
            if seed in result:
                raise ValueError(f"duplicate dataset seed {seed}")
            result[seed] = episode_index
    return result


def _episode_actions(dataset_root: Path, episode_index: int) -> np.ndarray:
    path = _episode_file(dataset_root.expanduser().resolve() / "data", episode_index)
    table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
    _validate_episode_columns(table, episode_index)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"episode {episode_index}: invalid action shape {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"episode {episode_index}: non-finite actions")
    return actions


def _episode_subtasks(subtask_root: Path, episode_index: int) -> np.ndarray:
    path = _episode_file(
        subtask_root.expanduser().resolve() / "subtask_indices", episode_index
    )
    table = pq.read_table(
        path,
        columns=["episode_index", "frame_index", "subtask_index"],
    )
    _validate_episode_columns(table, episode_index)
    return np.asarray(table["subtask_index"].to_numpy(), dtype=np.int64)


def _episode_file(root: Path, episode_index: int) -> Path:
    matches = tuple(root.glob(f"chunk-*/file-{int(episode_index):03d}.parquet"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one parquet file for episode {episode_index}, got {matches}"
        )
    return matches[0]


def _validate_episode_columns(table: Any, episode_index: int) -> None:
    episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    if episodes.size == 0 or not np.all(episodes == int(episode_index)):
        raise ValueError(f"episode {episode_index}: invalid episode_index column")
    if not np.array_equal(frames, np.arange(frames.size, dtype=np.int64)):
        raise ValueError(f"episode {episode_index}: frame_index is not contiguous")


if __name__ == "__main__":
    raise SystemExit(main())
