#!/usr/bin/env python
"""Rewrite a WR06 LeRobot dataset with one generic complete instruction."""

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


def rewrite_tasks(root: Path, boundaries: Path) -> None:
    records = _read_jsonl(boundaries)
    seed_rows = _read_jsonl(root / "meta" / "seeds.jsonl")
    seed_to_episode = {
        int(row["seed"]): int(row["episode_index"]) for row in seed_rows
    }
    if len(seed_to_episode) != len(seed_rows):
        raise ValueError("meta/seeds.jsonl contains duplicate source seeds")

    instruction_by_episode: dict[int, str] = {}
    for record in records:
        seed = int(record["seed"])
        if seed not in seed_to_episode:
            raise ValueError(f"boundary seed {seed} is absent from meta/seeds.jsonl")
        instruction = str(record.get("high_level_instruction", "")).strip()
        if not instruction:
            raise ValueError(f"boundary seed {seed} has no high-level instruction")
        episode_index = seed_to_episode[seed]
        if episode_index in instruction_by_episode:
            raise ValueError(f"duplicate boundary for episode {episode_index}")
        instruction_by_episode[episode_index] = instruction
    if set(instruction_by_episode) != set(seed_to_episode.values()):
        raise ValueError("per-episode task instruction coverage is incomplete")

    instructions = sorted(set(instruction_by_episode.values()))
    if len(instructions) != 1:
        raise ValueError(f"WR06 must expose one generic instruction, found {len(instructions)}")
    task_index_by_instruction = {
        instruction: index for index, instruction in enumerate(instructions)
    }
    pd.DataFrame(
        {"task_index": list(range(len(instructions)))},
        index=pd.Index(instructions, name="task"),
    ).to_parquet(root / "meta" / "tasks.parquet")

    data_files = sorted(
        glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet"))
    )
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
        task_index = task_index_by_instruction[instruction_by_episode[episode_index]]
        table = table.set_column(
            table.schema.get_field_index("task_index"),
            "task_index",
            pa.array([task_index] * table.num_rows, type=pa.int64()),
        )
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
        rewritten: list[list[str]] = []
        for value in table["episode_index"].to_pylist():
            episode_index = int(value)
            rewritten.append([instruction_by_episode[episode_index]])
            seen_episode_metadata.add(episode_index)
        field = table.schema.field("tasks")
        table = table.set_column(
            table.schema.get_field_index("tasks"),
            field,
            pa.array(rewritten, type=field.type),
        )
        _atomic_write_table(table, path)
    if seen_episode_metadata != set(instruction_by_episode):
        raise ValueError("meta/episodes coverage differs from task mapping")

    seeds_path = root / "meta" / "seeds.jsonl"
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
        f"rewrote {len(instruction_by_episode)} WR06 episodes with "
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
