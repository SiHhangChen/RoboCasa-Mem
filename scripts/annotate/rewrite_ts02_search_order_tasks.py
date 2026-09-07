#!/usr/bin/env python
"""Restore the original generic TS02 task instruction in a LeRobot copy.

Concrete left/center/right labels remain in ``subtask_index``. The sampled
expert search order is not exposed through the policy-visible task prompt.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="snappy")
    os.replace(temporary, path)


def _seed_to_episode_index(root: Path) -> dict[int, int]:
    rows = _read_jsonl(root / "meta" / "seeds.jsonl")
    mapping = {int(row["seed"]): int(row["episode_index"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("meta/seeds.jsonl contains duplicate source seeds")
    return mapping


def rewrite_tasks(root: Path, boundaries: Path) -> None:
    records = _read_jsonl(boundaries)
    seed_to_episode = _seed_to_episode_index(root)
    instruction_by_episode: dict[int, str] = {}
    for record in records:
        seed = int(record["seed"])
        if seed not in seed_to_episode:
            raise ValueError(f"boundary seed {seed} is absent from meta/seeds.jsonl")
        instruction = str(record.get("high_level_instruction", "")).strip()
        if not instruction:
            raise ValueError(f"boundary seed {seed} has no high-level instruction")
        instruction_by_episode[seed_to_episode[seed]] = instruction
    if len(instruction_by_episode) != len(seed_to_episode):
        missing = sorted(set(seed_to_episode.values()) - set(instruction_by_episode))
        raise ValueError(f"missing task instructions for episode indices {missing[:10]}")

    instructions = sorted(set(instruction_by_episode.values()))
    if len(instructions) != 1:
        raise ValueError(f"TS02 must expose one generic instruction, found {len(instructions)}")
    task_index_by_instruction = {
        instruction: index for index, instruction in enumerate(instructions)
    }

    tasks = pd.DataFrame(
        {"task_index": list(range(len(instructions)))},
        index=pd.Index(instructions, name="task"),
    )
    tasks.to_parquet(root / "meta" / "tasks.parquet")

    data_files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    if not data_files:
        raise FileNotFoundError(f"no data parquet files under {root / 'data'}")
    seen_episodes: set[int] = set()
    for raw_path in data_files:
        path = Path(raw_path)
        table = pq.read_table(path)
        episode_values = {int(value) for value in table["episode_index"].to_pylist()}
        if len(episode_values) != 1:
            raise ValueError(f"{path} contains multiple episode indices")
        episode_index = next(iter(episode_values))
        instruction = instruction_by_episode[episode_index]
        task_index = task_index_by_instruction[instruction]
        column = pa.array([task_index] * table.num_rows, type=pa.int64())
        index = table.schema.get_field_index("task_index")
        table = table.set_column(index, "task_index", column)
        _atomic_write_table(table, path)
        seen_episodes.add(episode_index)
    if seen_episodes != set(instruction_by_episode):
        raise ValueError("data parquet episode coverage differs from task mapping")

    episode_files = sorted(
        glob.glob(str(root / "meta" / "episodes" / "chunk-*" / "file-*.parquet"))
    )
    seen_episode_metadata: set[int] = set()
    for raw_path in episode_files:
        path = Path(raw_path)
        table = pq.read_table(path)
        values = table["episode_index"].to_pylist()
        rewritten = []
        for value in values:
            episode_index = int(value)
            rewritten.append([instruction_by_episode[episode_index]])
            seen_episode_metadata.add(episode_index)
        field = table.schema.field("tasks")
        column = pa.array(rewritten, type=field.type)
        table = table.set_column(table.schema.get_field_index("tasks"), field, column)
        _atomic_write_table(table, path)
    if seen_episode_metadata != set(instruction_by_episode):
        raise ValueError("meta/episodes coverage differs from task mapping")

    seeds_path = root / "meta" / "seeds.jsonl"
    seed_rows = _read_jsonl(seeds_path)
    temporary = seeds_path.with_name(seeds_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in seed_rows:
            episode_index = int(row["episode_index"])
            updated = dict(row)
            updated["task"] = instruction_by_episode[episode_index]
            handle.write(json.dumps(updated, sort_keys=True) + "\n")
    os.replace(temporary, seeds_path)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["total_tasks"] = len(instructions)
    info_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"rewrote {len(instruction_by_episode)} TS02 episodes with "
        "one generic task instruction"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    args = parser.parse_args()
    rewrite_tasks(args.lerobot_root.resolve(), args.boundaries.resolve())


if __name__ == "__main__":
    main()
