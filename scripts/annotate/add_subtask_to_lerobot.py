#!/usr/bin/env python
"""Add per-frame subtask_index to an existing LeRobot dataset (in place).

Reads labeled episodes from ``apply_subtask_labels.py`` (subtasks.jsonl) and
writes, for every data parquet, a ``subtask_index`` int64 column; writes
``meta/subtasks.parquet`` (index = subtask label, column = subtask_index); and
registers the feature in ``meta/info.json``. Videos and all other columns are
left untouched. After this, lerobot >=0.4.4 returns ``item["subtask"]`` on
every frame, which is what MemER's generate_sft_data.py / eval rollout read.

Records may provide an explicit ``episode_index`` when the source recorder
seeds are not contiguous; otherwise the recorder seed is used as the index.

Usage:
    python scripts/annotate/add_subtask_to_lerobot.py \\
      --lerobot-root lerobot_data/wa01_200seeds_v061 \\
      --subtasks /tmp/subtasks_wa01.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def load_subtask_map(subtasks_path: Path) -> dict[int, dict]:
    records = [
        json.loads(line)
        for line in subtasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {int(r.get("episode_index", r["seed"])): r for r in records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--subtasks", type=Path, required=True)
    parser.add_argument(
        "--high-level-instruction",
        default=None,
        help="Replace the single task label in meta/tasks.parquet.",
    )
    parser.add_argument(
        "--include-memory-keyframes",
        action="store_true",
        help="Add a per-frame memory_keyframe bool column from record keyframes.",
    )
    parser.add_argument("--backup-info", action="store_true",
                        help="copy info.json to info.json.subtask_backup first")
    args = parser.parse_args()

    root = args.lerobot_root
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))

    if args.backup_info:
        shutil.copy2(info_path, info_path.with_suffix(".json.subtask_backup"))

    subtask_map = load_subtask_map(args.subtasks)
    print(f"loaded {len(subtask_map)} labeled episodes")

    data_files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    if not data_files:
        raise FileNotFoundError(f"no data parquet files under {root / 'data'}")

    n_columns_written = 0
    for path in data_files:
        table = pq.read_table(path)
        ep = table.column("episode_index").to_pylist()
        if not ep:
            continue
        ep_idx = int(ep[0])
        record = subtask_map.get(ep_idx)
        if record is None:
            print(f"WARN: no subtask record for episode_index={ep_idx} ({path}), skipping")
            continue
        indices = record["subtask_indices"]
        if len(indices) != table.num_rows:
            raise ValueError(
                f"episode {ep_idx}: subtask labels {len(indices)} != rows {table.num_rows}"
            )
        if "subtask_index" in table.column_names:
            table = table.drop(["subtask_index"])
        subtask_col = pa.array(indices, type=pa.int64())
        table = table.append_column("subtask_index", subtask_col)
        if args.include_memory_keyframes:
            keyframe_set = {int(value) for value in record.get("keyframes", [])}
            frame_indices = table["frame_index"].to_pylist()
            memory_col = pa.array(
                [int(frame) in keyframe_set for frame in frame_indices],
                type=pa.bool_(),
            )
            if "memory_keyframe" in table.column_names:
                table = table.drop(["memory_keyframe"])
            table = table.append_column("memory_keyframe", memory_col)
        tmp = path + ".tmp"
        pq.write_table(table, tmp, compression="snappy")
        Path(tmp).replace(path)
        n_columns_written += 1

    # meta/subtasks.parquet: index = label, column = subtask_index. The label
    # order must exactly match the index semantics used by subtask_indices,
    # which is the config's subtask order. apply_subtask_labels.py now emits an
    # explicit ordered `subtask_order` list to avoid any dict-ordering drift.
    first = next(iter(subtask_map.values()))
    if "subtask_order" in first and first["subtask_order"]:
        labels = [str(label) for label in first["subtask_order"]]
    else:
        labels = [str(label) for _, label in sorted(
            (int(k), v) for k, v in first["subtask_names"].items()
        )]
    order = list(range(len(labels)))
    sub_df = pd.DataFrame({"subtask_index": order}, index=labels)
    sub_df.index.name = "subtask"
    sub_df.to_parquet(root / "meta" / "subtasks.parquet")

    if args.high_level_instruction:
        tasks_path = root / "meta" / "tasks.parquet"
        tasks_df = pd.read_parquet(tasks_path)
        if len(tasks_df) != 1 or "task_index" not in tasks_df.columns:
            raise ValueError(
                "--high-level-instruction currently requires exactly one task row"
            )
        tasks_df.index = pd.Index([args.high_level_instruction], name="task")
        tasks_df.to_parquet(tasks_path)

    # register feature in info.json
    info["features"]["subtask_index"] = {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    }
    if args.include_memory_keyframes:
        info["features"]["memory_keyframe"] = {
            "dtype": "bool",
            "shape": [1],
            "names": None,
        }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote subtask_index to {n_columns_written} parquet files")
    print(f"wrote {root / 'meta' / 'subtasks.parquet'} with {len(labels)} subtasks")
    if args.high_level_instruction:
        print(f"updated {root / 'meta' / 'tasks.parquet'} task instruction")
    print(f"registered subtask_index feature in {info_path}")


if __name__ == "__main__":
    main()
