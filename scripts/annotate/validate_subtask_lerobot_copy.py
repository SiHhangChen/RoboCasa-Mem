#!/usr/bin/env python
"""Validate a LeRobot dataset copy produced by add_subtask_to_lerobot.py.

The checks are read-only. They prove that the copied dataset preserves every
source value, adds only the requested subtask artifacts, and can optionally be
loaded through LeRobot with resolved text labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SUBTASK_FEATURE = {"dtype": "int64", "shape": [1], "names": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--annotated-root", type=Path, required=True)
    parser.add_argument("--subtasks", type=Path, required=True)
    parser.add_argument("--expected-task-instruction", default=None)
    parser.add_argument(
        "--task-boundaries",
        type=Path,
        default=None,
        help=(
            "Optional boundary JSONL carrying per-seed high_level_instruction. "
            "Allows and validates a per-episode task_index/tasks metadata rewrite."
        ),
    )
    parser.add_argument("--expect-memory-keyframes", action="store_true")
    parser.add_argument(
        "--check-lerobot",
        action="store_true",
        help="Load metadata and representative frames through LeRobot.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parquet_paths(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/file-*.parquet"))


def assert_not_hardlinked(source: Path, annotated: Path) -> None:
    source_stat = source.stat()
    annotated_stat = annotated.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (
        annotated_stat.st_dev,
        annotated_stat.st_ino,
    ):
        raise AssertionError(f"copy is hardlinked to source: {annotated}")


def validate_storage(source_root: Path, annotated_root: Path) -> None:
    videos = annotated_root / "videos"
    if not videos.is_symlink():
        raise AssertionError(f"annotated videos is not a symlink: {videos}")
    if videos.resolve() != (source_root / "videos").resolve():
        raise AssertionError(
            f"annotated videos resolves to {videos.resolve()}, expected "
            f"{(source_root / 'videos').resolve()}"
        )

    for directory in ("data", "meta"):
        source_dir = source_root / directory
        annotated_dir = annotated_root / directory
        for source_path in source_dir.rglob("*"):
            if not source_path.is_file():
                continue
            relative = source_path.relative_to(source_root)
            annotated_path = annotated_root / relative
            if not annotated_path.is_file():
                raise FileNotFoundError(f"copied file is missing: {annotated_path}")
            assert_not_hardlinked(source_path, annotated_path)


def validate_info(
    source_root: Path,
    annotated_root: Path,
    *,
    expected_task_count: int | None = None,
) -> None:
    source_info = load_json(source_root / "meta" / "info.json")
    annotated_info = load_json(annotated_root / "meta" / "info.json")
    feature = annotated_info.get("features", {}).get("subtask_index")
    if feature != SUBTASK_FEATURE:
        raise AssertionError(f"unexpected subtask_index feature: {feature}")

    normalized = copy.deepcopy(annotated_info)
    del normalized["features"]["subtask_index"]
    if "memory_keyframe" in normalized.get("features", {}):
        del normalized["features"]["memory_keyframe"]
    if expected_task_count is not None:
        if annotated_info.get("total_tasks") != expected_task_count:
            raise AssertionError(
                f"info.json total_tasks={annotated_info.get('total_tasks')} != "
                f"{expected_task_count}"
            )
        normalized["total_tasks"] = source_info.get("total_tasks")
    if normalized != source_info:
        raise AssertionError("info.json changed beyond features.subtask_index")


def validate_meta_copy(
    source_root: Path,
    annotated_root: Path,
    *,
    allow_task_instruction_change: bool,
    allow_per_episode_task_change: bool,
) -> None:
    allowed_changes = {Path("meta/subtasks.parquet")}
    if allow_task_instruction_change:
        allowed_changes.add(Path("meta/tasks.parquet"))
    for source_path in (source_root / "meta").rglob("*"):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source_root)
        if allow_per_episode_task_change and (
            relative == Path("meta/tasks.parquet")
            or relative == Path("meta/seeds.jsonl")
            or relative.parts[:2] == ("meta", "episodes")
        ):
            continue
        if relative in allowed_changes or relative == Path("meta/info.json"):
            continue
        annotated_path = annotated_root / relative
        if source_path.read_bytes() != annotated_path.read_bytes():
            raise AssertionError(f"metadata changed unexpectedly: {relative}")


def ordered_labels(record: dict[str, Any]) -> list[str]:
    if record.get("subtask_order"):
        return [str(label) for label in record["subtask_order"]]
    return [
        str(label)
        for _, label in sorted(
            (int(index), label) for index, label in record["subtask_names"].items()
        )
    ]


def validate_subtask_metadata(
    annotated_root: Path, records: list[dict[str, Any]]
) -> list[str]:
    expected = ordered_labels(records[0])
    for record in records[1:]:
        if ordered_labels(record) != expected:
            raise AssertionError(f"subtask order differs for seed {record['seed']}")

    table = pq.read_table(annotated_root / "meta" / "subtasks.parquet")
    label_column = "subtask"
    if label_column not in table.column_names:
        label_column = "__index_level_0__"
    actual_labels = [str(value) for value in table[label_column].to_pylist()]
    actual_indices = [int(value) for value in table["subtask_index"].to_pylist()]
    if actual_labels != expected:
        raise AssertionError(
            f"subtasks.parquet labels {actual_labels} != expected {expected}"
        )
    if actual_indices != list(range(len(expected))):
        raise AssertionError(f"unexpected subtask indices: {actual_indices}")
    return expected


def validate_task_instruction(annotated_root: Path, expected: str | None) -> None:
    if expected is None:
        return
    table = pq.read_table(annotated_root / "meta" / "tasks.parquet")
    label_column = "task" if "task" in table.column_names else "__index_level_0__"
    actual = [str(value) for value in table[label_column].to_pylist()]
    if actual != [expected]:
        raise AssertionError(f"tasks.parquet instruction {actual} != {[expected]}")


def expected_per_episode_tasks(
    annotated_root: Path,
    boundary_records: list[dict[str, Any]],
) -> tuple[dict[int, str], dict[str, int]]:
    seed_rows = load_jsonl(annotated_root / "meta" / "seeds.jsonl")
    episode_by_seed = {
        int(row["seed"]): int(row["episode_index"]) for row in seed_rows
    }
    task_by_episode: dict[int, str] = {}
    for record in boundary_records:
        seed = int(record["seed"])
        instruction = str(record.get("high_level_instruction", "")).strip()
        if not instruction:
            raise AssertionError(f"boundary seed {seed} has no high_level_instruction")
        if seed not in episode_by_seed:
            raise AssertionError(f"boundary seed {seed} is absent from meta/seeds.jsonl")
        task_by_episode[episode_by_seed[seed]] = instruction
    if len(task_by_episode) != len(episode_by_seed):
        raise AssertionError("per-episode task boundary coverage is incomplete")
    task_index = {
        instruction: index
        for index, instruction in enumerate(sorted(set(task_by_episode.values())))
    }
    return task_by_episode, task_index


def validate_per_episode_task_metadata(
    annotated_root: Path,
    task_by_episode: dict[int, str],
    task_index: dict[str, int],
) -> None:
    tasks = pq.read_table(annotated_root / "meta" / "tasks.parquet")
    label_column = "task" if "task" in tasks.column_names else "__index_level_0__"
    actual = {
        str(label): int(index)
        for label, index in zip(
            tasks[label_column].to_pylist(), tasks["task_index"].to_pylist()
        )
    }
    if actual != task_index:
        raise AssertionError(f"tasks.parquet mapping {actual} != expected {task_index}")

    seed_rows = load_jsonl(annotated_root / "meta" / "seeds.jsonl")
    for row in seed_rows:
        episode_index = int(row["episode_index"])
        if row.get("task") != task_by_episode[episode_index]:
            raise AssertionError(f"seeds.jsonl task mismatch for episode {episode_index}")

    seen: set[int] = set()
    for path in sorted((annotated_root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "tasks"])
        for episode_index, values in zip(
            table["episode_index"].to_pylist(), table["tasks"].to_pylist()
        ):
            index = int(episode_index)
            if values != [task_by_episode[index]]:
                raise AssertionError(f"meta/episodes task mismatch for episode {index}")
            seen.add(index)
    if seen != set(task_by_episode):
        raise AssertionError("meta/episodes task coverage is incomplete")


def validate_data(
    source_root: Path,
    annotated_root: Path,
    records: list[dict[str, Any]],
    *,
    expect_memory_keyframes: bool,
    task_by_episode: dict[int, str] | None = None,
    task_index: dict[str, int] | None = None,
) -> int:
    by_episode = {
        int(record.get("episode_index", record["seed"])): record for record in records
    }
    if len(by_episode) != len(records):
        raise AssertionError("subtasks.jsonl contains duplicate episode indices")

    source_paths = parquet_paths(source_root)
    annotated_paths = parquet_paths(annotated_root)
    source_relatives = [path.relative_to(source_root) for path in source_paths]
    annotated_relatives = [path.relative_to(annotated_root) for path in annotated_paths]
    if source_relatives != annotated_relatives:
        raise AssertionError("source and annotated parquet file sets differ")

    seen_seeds: set[int] = set()
    total_rows = 0
    for relative in source_relatives:
        source = pq.read_table(source_root / relative)
        annotated = pq.read_table(annotated_root / relative)
        expected_columns = source.column_names + ["subtask_index"]
        if expect_memory_keyframes:
            expected_columns.append("memory_keyframe")
        if annotated.column_names != expected_columns:
            raise AssertionError(f"unexpected columns in {relative}")
        for name in source.column_names:
            if task_by_episode is not None and name == "task_index":
                continue
            if not source[name].equals(annotated[name]):
                raise AssertionError(f"column {name} changed in {relative}")

        episode_indices = set(source["episode_index"].to_pylist())
        if len(episode_indices) != 1:
            raise AssertionError(f"multiple episode indices in {relative}")
        episode_index = int(next(iter(episode_indices)))
        record = by_episode.get(episode_index)
        if record is None:
            raise AssertionError(f"no labels for episode_index {episode_index}")
        if task_by_episode is not None:
            assert task_index is not None
            expected_task_index = task_index[task_by_episode[episode_index]]
            if annotated["task_index"].to_pylist() != [expected_task_index] * annotated.num_rows:
                raise AssertionError(f"task_index sequence differs for episode {episode_index}")
        expected_indices = [int(value) for value in record["subtask_indices"]]
        actual_indices = annotated["subtask_index"].to_pylist()
        if actual_indices != expected_indices:
            raise AssertionError(f"subtask sequence differs for episode {episode_index}")
        if annotated.schema.field("subtask_index").type != pa.int64():
            raise AssertionError(f"subtask_index is not int64 in {relative}")
        if expect_memory_keyframes:
            expected_memory = [
                int(offset) in {int(value) for value in record.get("keyframes", [])}
                for offset in range(annotated.num_rows)
            ]
            if annotated["memory_keyframe"].to_pylist() != expected_memory:
                raise AssertionError(f"memory_keyframe sequence differs for episode {episode_index}")
        seen_seeds.add(episode_index)
        total_rows += annotated.num_rows

    if seen_seeds != set(by_episode):
        missing = sorted(set(by_episode) - seen_seeds)
        extra = sorted(seen_seeds - set(by_episode))
        raise AssertionError(f"episode/label seed mismatch: missing={missing}, extra={extra}")
    return total_rows


def dataframe_index_map(frame: Any, value_column: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for label, row in frame.iterrows():
        resolved_label = row.get("__index_level_0__", label)
        mapping[int(row[value_column])] = str(resolved_label)
    return mapping


def validate_lerobot(
    annotated_root: Path,
    records: list[dict[str, Any]],
    expected_labels: list[str],
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    repo_id = annotated_root.name.removeprefix(".").removesuffix(".staging")
    metadata = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=annotated_root.resolve(),
        force_cache_sync=False,
    )
    mapping = dataframe_index_map(metadata.subtasks, "subtask_index")
    if mapping != dict(enumerate(expected_labels)):
        raise AssertionError(f"LeRobot subtask mapping is incorrect: {mapping}")

    first = records[0]
    seed = int(first.get("episode_index", first["seed"]))
    indices = [int(value) for value in first["subtask_indices"]]
    sample_offsets = {0, len(indices) - 1}
    sample_offsets.update(int(value) for value in first.get("keyframes", []))
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=annotated_root.resolve(),
        episodes=[seed],
        force_cache_sync=False,
        tolerance_s=0.06,
    )
    for offset in sorted(sample_offsets):
        item = dataset[offset]
        expected_index = indices[offset]
        actual_index = int(item["subtask_index"].item())
        if actual_index != expected_index:
            raise AssertionError(
                f"LeRobot subtask_index mismatch at episode {seed}, frame {offset}"
            )
        if str(item["subtask"]) != expected_labels[expected_index]:
            raise AssertionError(
                f"LeRobot subtask label mismatch at episode {seed}, frame {offset}"
            )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    annotated_root = args.annotated_root.resolve()
    records = load_jsonl(args.subtasks)
    if not records:
        raise ValueError("subtasks.jsonl is empty")

    task_by_episode: dict[int, str] | None = None
    task_index: dict[str, int] | None = None
    if args.task_boundaries is not None:
        task_by_episode, task_index = expected_per_episode_tasks(
            annotated_root, load_jsonl(args.task_boundaries)
        )

    validate_storage(source_root, annotated_root)
    validate_info(
        source_root,
        annotated_root,
        expected_task_count=None if task_index is None else len(task_index),
    )
    validate_meta_copy(
        source_root,
        annotated_root,
        allow_task_instruction_change=args.expected_task_instruction is not None,
        allow_per_episode_task_change=task_by_episode is not None,
    )
    labels = validate_subtask_metadata(annotated_root, records)
    validate_task_instruction(annotated_root, args.expected_task_instruction)
    if task_by_episode is not None:
        assert task_index is not None
        validate_per_episode_task_metadata(
            annotated_root, task_by_episode, task_index
        )
    total_rows = validate_data(
        source_root,
        annotated_root,
        records,
        expect_memory_keyframes=args.expect_memory_keyframes,
        task_by_episode=task_by_episode,
        task_index=task_index,
    )
    if args.check_lerobot:
        validate_lerobot(annotated_root, records, labels)

    print(
        json.dumps(
            {
                "annotated_root": os.fspath(annotated_root),
                "episodes": len(records),
                "frames": total_rows,
                "lerobot_checked": bool(args.check_lerobot),
                "subtasks": labels,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
