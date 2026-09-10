#!/usr/bin/env python
"""Synchronize TM02 LeRobot metadata with its canonical policy prompt.

This migration changes ``meta/tasks.parquet``, the redundant ``task`` field in
``meta/seeds.jsonl``, and each episode metadata row's ``tasks`` list. Frame
data, task indices, observations, actions, videos, and MemER subtask
annotations remain untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.tasks.tm02.domain.contracts import (  # noqa: E402
    CANONICAL_PROMPT,
    TASK_ID,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"metadata file is empty: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"metadata file must contain JSON objects: {path}")
    return records


def _validate_root(
    root: Path,
    *,
    expected_episodes: int,
    task_id: str = TASK_ID,
) -> tuple[list[dict[str, Any]], list[Path]]:
    tasks_path = root / "meta" / "tasks.parquet"
    seeds_path = root / "meta" / "seeds.jsonl"
    info_path = root / "meta" / "info.json"
    for path in (tasks_path, seeds_path, info_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tasks = pq.read_table(tasks_path)
    if tasks.column_names != ["task_index", "task"] or tasks.num_rows != 1:
        raise ValueError(
            f"{tasks_path} must contain one task_index/task row, got "
            f"columns={tasks.column_names}, rows={tasks.num_rows}"
        )
    if tasks["task_index"].to_pylist() != [0]:
        raise ValueError(f"{tasks_path} must retain task_index=0")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info.get("total_tasks", -1)) != 1:
        raise ValueError(f"{info_path} must declare total_tasks=1")

    seeds = _read_jsonl(seeds_path)
    if expected_episodes > 0 and len(seeds) != expected_episodes:
        raise ValueError(
            f"{seeds_path} contains {len(seeds)} episodes; "
            f"expected {expected_episodes}"
        )
    episode_indices = [int(record["episode_index"]) for record in seeds]
    source_seeds = [int(record["seed"]) for record in seeds]
    if len(set(episode_indices)) != len(episode_indices):
        raise ValueError(f"{seeds_path} contains duplicate episode indices")
    if len(set(source_seeds)) != len(source_seeds):
        raise ValueError(f"{seeds_path} contains duplicate source seeds")
    for record in seeds:
        if str(record.get("task_id")) != task_id:
            raise ValueError(f"unexpected task_id in {seeds_path}: {record!r}")

    episode_paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    episode_metadata_indices: set[int] = set()
    for path in episode_paths:
        table = pq.read_table(path, columns=["episode_index", "tasks"])
        if table.num_rows != 1:
            raise ValueError(f"expected one episode metadata row in {path}")
        episode_index = int(table["episode_index"][0].as_py())
        if episode_index in episode_metadata_indices:
            raise ValueError(f"duplicate episode metadata for episode {episode_index}")
        episode_metadata_indices.add(episode_index)
        tasks_type = table.schema.field("tasks").type
        if not (pa.types.is_list(tasks_type) or pa.types.is_large_list(tasks_type)):
            raise ValueError(f"{path} tasks column must be a list, got {tasks_type}")
    if episode_metadata_indices != set(episode_indices):
        raise ValueError(
            f"{root} episode metadata/seed coverage differs: "
            f"metadata={len(episode_metadata_indices)}, seeds={len(episode_indices)}"
        )

    data_paths = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"no frame parquet files under {root / 'data'}")
    seen_episodes: set[int] = set()
    for path in data_paths:
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        task_indices = {int(value) for value in table["task_index"].to_pylist()}
        if task_indices != {0}:
            raise ValueError(f"{path} contains task indices {sorted(task_indices)}")
        seen_episodes.update(
            int(value) for value in table["episode_index"].to_pylist()
        )
    if seen_episodes != set(episode_indices):
        raise ValueError(
            f"{root} frame/seed episode coverage differs: "
            f"frames={len(seen_episodes)}, seeds={len(episode_indices)}"
        )
    return seeds, episode_paths


def _rewrite_root(
    root: Path,
    seeds: list[dict[str, Any]],
    episode_paths: list[Path],
    *,
    canonical_prompt: str = CANONICAL_PROMPT,
) -> None:
    tasks_path = root / "meta" / "tasks.parquet"
    tasks_tmp = tasks_path.with_name(tasks_path.name + ".tmp")
    task_frame = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([canonical_prompt], name="task"),
    )
    task_frame.to_parquet(tasks_tmp)
    os.replace(tasks_tmp, tasks_path)

    seeds_path = root / "meta" / "seeds.jsonl"
    seeds_tmp = seeds_path.with_name(seeds_path.name + ".tmp")
    with seeds_tmp.open("w", encoding="utf-8") as handle:
        for record in seeds:
            updated = dict(record)
            updated["task"] = canonical_prompt
            handle.write(
                json.dumps(updated, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(seeds_tmp, seeds_path)

    for episode_path in episode_paths:
        table = pq.read_table(episode_path)
        tasks_index = table.column_names.index("tasks")
        tasks_type = table.schema.field("tasks").type
        table = table.set_column(
            tasks_index,
            "tasks",
            pa.array([[canonical_prompt]], type=tasks_type),
        )
        episode_tmp = episode_path.with_name(episode_path.name + ".tmp")
        pq.write_table(table, episode_tmp, compression="snappy")
        os.replace(episode_tmp, episode_path)


def rewrite_roots(
    roots: list[Path],
    *,
    expected_episodes: int,
    task_id: str = TASK_ID,
    canonical_prompt: str = CANONICAL_PROMPT,
) -> None:
    resolved = [root.expanduser().resolve() for root in roots]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate --lerobot-root values")
    validated = [
        (
            root,
            _validate_root(
                root,
                expected_episodes=expected_episodes,
                task_id=task_id,
            ),
        )
        for root in resolved
    ]
    for root, (seeds, episode_paths) in validated:
        _rewrite_root(
            root,
            seeds,
            episode_paths,
            canonical_prompt=canonical_prompt,
        )
        print(
            f"updated {task_id} prompt metadata for {len(seeds)} episodes in {root}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        action="append",
        required=True,
        help="TM02 LeRobot dataset root; may be supplied more than once.",
    )
    parser.add_argument("--expected-episodes", type=int, default=200)
    args = parser.parse_args()
    if args.expected_episodes < 0:
        parser.error("--expected-episodes must be non-negative")
    rewrite_roots(
        args.lerobot_root,
        expected_episodes=args.expected_episodes,
    )


if __name__ == "__main__":
    main()
