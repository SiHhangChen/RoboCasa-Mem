#!/usr/bin/env python
"""Create a lightweight Pi0.5 view over a MemBench LeRobot dataset.

The physical trajectory remains in the original LeRobot dataset. A view only
adds a human-readable subtask table and one small, per-episode Parquet sidecar
whose path mirrors the corresponding ``data/chunk-*/file-*.parquet`` file.
Everything else is a relative symlink to the original dataset.

OpenPI must use the sidecar-aware MemBench v3 loader. It replaces each frame's
overall ``task_index`` with the sidecar ``subtask_index`` at load time; the
original observations, actions, videos, and metadata are never rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SIDECAR_SCHEMA_VERSION = "membench.pi05_subtask_sidecar/v1"
IDENTITY_COLUMNS = ("episode_index", "frame_index", "index")
SIDECAR_COLUMNS = (*IDENTITY_COLUMNS, "subtask_index")
ANNOTATED_DATASET_PATTERN = re.compile(r"^(?P<base>.+)_subtasks_v\d+$")


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _load_subtask_labels(source: Path) -> list[str]:
    path = source / "meta" / "subtasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = sorted(
        pq.read_table(path, columns=["subtask_index", "subtask"]).to_pylist(),
        key=lambda row: int(row["subtask_index"]),
    )
    expected = list(range(len(rows)))
    actual = [int(row["subtask_index"]) for row in rows]
    if actual != expected:
        raise ValueError(f"subtask indices must be contiguous: {actual}")

    labels = [str(row["subtask"]) for row in rows]
    if not labels:
        raise ValueError(f"no subtask labels found in {path}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"subtask labels must be unique: {labels}")
    for index, label in enumerate(labels):
        if not label.strip():
            raise ValueError(f"subtask label {index} is empty")
        if label != label.strip() or "\n" in label or "\r" in label or "\t" in label:
            raise ValueError(
                f"subtask label {index} cannot contain surrounding whitespace, "
                f"tabs, or newlines: {label!r}"
            )
    return labels


def _infer_data_source(source: Path) -> Path:
    match = ANNOTATED_DATASET_PATTERN.fullmatch(source.name)
    if match is None or source.parent.name != "memer":
        raise ValueError(
            "cannot infer the physical dataset root; pass --data-source explicitly"
        )
    return source.parent.parent / match.group("base")


def _relative_symlink(link: Path, target: Path) -> None:
    if not target.exists():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, start=link.parent)
    link.symlink_to(relative_target, target_is_directory=target.is_dir())


def _validate_dataset_pair(source: Path, data_source: Path) -> tuple[dict[str, Any], list[Path]]:
    source_info = _read_info(source)
    data_info = _read_info(data_source)
    for field in ("total_episodes", "total_frames", "fps"):
        if source_info.get(field) != data_info.get(field):
            raise ValueError(
                f"annotated/physical dataset {field} differs: "
                f"{source_info.get(field)!r} != {data_info.get(field)!r}"
            )

    source_files = sorted((source / "data").glob("chunk-*/*.parquet"))
    data_files = sorted((data_source / "data").glob("chunk-*/*.parquet"))
    if not source_files or not data_files:
        raise FileNotFoundError("both datasets must contain data/chunk-*/*.parquet files")
    source_relatives = [path.relative_to(source / "data") for path in source_files]
    data_relatives = [path.relative_to(data_source / "data") for path in data_files]
    if source_relatives != data_relatives:
        raise ValueError("annotated and physical datasets have different data file layouts")
    return data_info, source_files


def _integer_values(table: pa.Table, column: str) -> list[int]:
    return [int(value) for value in table[column].to_pylist()]


def _build_sidecar(
    *,
    annotated_path: Path,
    physical_path: Path,
    output_path: Path,
    label_count: int,
    source_relative_path: Path,
) -> tuple[int, int]:
    annotated = pq.read_table(annotated_path, columns=list(SIDECAR_COLUMNS))
    physical = pq.read_table(physical_path, columns=list(IDENTITY_COLUMNS))
    if annotated.num_rows != physical.num_rows:
        raise ValueError(
            f"row count differs for {source_relative_path}: "
            f"annotated={annotated.num_rows}, physical={physical.num_rows}"
        )
    for column in IDENTITY_COLUMNS:
        if _integer_values(annotated, column) != _integer_values(physical, column):
            raise ValueError(f"{column} differs for {source_relative_path}")

    episode_values = sorted(set(_integer_values(annotated, "episode_index")))
    if len(episode_values) != 1:
        raise ValueError(
            f"expected one episode in {source_relative_path}, got {episode_values}"
        )
    episode_index = episode_values[0]
    subtask_indices = _integer_values(annotated, "subtask_index")
    invalid = sorted(set(subtask_indices) - set(range(label_count)))
    if invalid:
        raise ValueError(
            f"invalid subtask indices in episode {episode_index}: {invalid}"
        )

    sidecar = pa.table(
        {
            "episode_index": pa.array(
                _integer_values(annotated, "episode_index"), type=pa.int64()
            ),
            "frame_index": pa.array(
                _integer_values(annotated, "frame_index"), type=pa.int64()
            ),
            "index": pa.array(_integer_values(annotated, "index"), type=pa.int64()),
            "subtask_index": pa.array(subtask_indices, type=pa.int16()),
        }
    ).replace_schema_metadata(
        {
            b"membench.schema": SIDECAR_SCHEMA_VERSION.encode("utf-8"),
            b"membench.source_data": source_relative_path.as_posix().encode("utf-8"),
            b"membench.episode_index": str(episode_index).encode("ascii"),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Match the physical LeRobot parquet files so lightweight parquet viewers
    # do not need an additional ZSTD codec just to inspect the sidecars.
    pq.write_table(sidecar, output_path, compression="snappy")
    return episode_index, sidecar.num_rows


def _write_subtasks_tsv(path: Path, labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{index}\t{label}\n" for index, label in enumerate(labels)),
        encoding="utf-8",
    )


def _link_physical_dataset(staging: Path, data_source: Path) -> None:
    _relative_symlink(staging / "data", data_source / "data")
    _relative_symlink(staging / "videos", data_source / "videos")
    meta_output = staging / "meta"
    meta_output.mkdir(parents=True, exist_ok=True)
    for source_entry in sorted((data_source / "meta").iterdir()):
        _relative_symlink(meta_output / source_entry.name, source_entry)


def convert_dataset(
    source: Path,
    output: Path,
    *,
    data_source: Path | None = None,
) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    data_source = (
        data_source.expanduser().resolve()
        if data_source is not None
        else _infer_data_source(source).resolve()
    )
    if not source.is_dir():
        raise FileNotFoundError(f"annotated dataset not found: {source}")
    if not data_source.is_dir():
        raise FileNotFoundError(f"physical dataset not found: {data_source}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")

    labels = _load_subtask_labels(source)
    data_info, annotated_files = _validate_dataset_pair(source, data_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"staging path already exists: {staging}")

    try:
        staging.mkdir()
        _link_physical_dataset(staging, data_source)
        _write_subtasks_tsv(staging / "meta" / "subtasks.tsv", labels)

        episodes: set[int] = set()
        total_frames = 0
        for annotated_path in annotated_files:
            relative = annotated_path.relative_to(source / "data")
            episode_index, frame_count = _build_sidecar(
                annotated_path=annotated_path,
                physical_path=data_source / "data" / relative,
                output_path=staging / "subtask_indices" / relative,
                label_count=len(labels),
                source_relative_path=Path("data") / relative,
            )
            if episode_index in episodes:
                raise ValueError(f"duplicate sidecar for episode {episode_index}")
            episodes.add(episode_index)
            total_frames += frame_count

        expected_episodes = int(data_info["total_episodes"])
        expected_frames = int(data_info["total_frames"])
        if len(episodes) != expected_episodes:
            raise ValueError(
                f"sidecar episode count {len(episodes)} != {expected_episodes}"
            )
        if total_frames != expected_frames:
            raise ValueError(
                f"sidecar frame count {total_frames} != {expected_frames}"
            )
        staging.replace(output)
    except Exception:
        if staging.exists() and staging.is_dir():
            shutil.rmtree(staging)
        raise

    print(f"physical dataset: {data_source}")
    print(f"annotated dataset: {source}")
    print(f"linked episodes: {len(episodes)}")
    print(f"indexed frames: {total_frames}")
    print(f"Pi0.5 subtask prompts: {len(labels)}")
    print(f"output view: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--data-source",
        type=Path,
        help="Physical LeRobot dataset. Inferred for standard lerobot_data/memer names.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert_dataset(args.source, args.output, data_source=args.data_source)


if __name__ == "__main__":
    main()
