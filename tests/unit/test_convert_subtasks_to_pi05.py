from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.annotate.convert_subtasks_to_pi05 import (
    SIDECAR_SCHEMA_VERSION,
    convert_dataset,
)


def _write_dataset_pair(tmp_path: Path) -> tuple[Path, Path]:
    physical = tmp_path / "toy"
    annotated = tmp_path / "memer" / "toy_subtasks_v2"
    for root in (physical, annotated):
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (root / "videos").mkdir()
        (root / "meta" / "info.json").write_text(
            json.dumps({"total_episodes": 1, "total_frames": 4, "fps": 20}),
            encoding="utf-8",
        )
        (root / "meta" / "stats.json").write_text("{}\n", encoding="utf-8")
        (root / "meta" / "seeds.jsonl").write_text(
            '{"episode_index": 0, "seed": 0}\n', encoding="utf-8"
        )
        pq.write_table(
            pa.table({"task_index": [0], "task": ["overall task"]}),
            root / "meta" / "tasks.parquet",
        )
        pq.write_table(
            pa.table({"episode_index": [0], "length": [4]}),
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        )

    identity = {
        "episode_index": pa.array([0, 0, 0, 0], type=pa.int64()),
        "frame_index": pa.array([0, 1, 2, 3], type=pa.int64()),
        "index": pa.array([0, 1, 2, 3], type=pa.int64()),
    }
    pq.write_table(
        pa.table({**identity, "task_index": pa.array([0, 0, 0, 0])}),
        physical / "data" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                **identity,
                "task_index": pa.array([0, 0, 0, 0]),
                "subtask_index": pa.array([0, 0, 1, 1]),
            }
        ),
        annotated / "data" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "subtask_index": pa.array([0, 1]),
                "subtask": pa.array(["first stage", "second stage"]),
            }
        ),
        annotated / "meta" / "subtasks.parquet",
    )
    return physical, annotated


def test_convert_dataset_creates_symlink_view_and_episode_sidecar(tmp_path: Path) -> None:
    physical, annotated = _write_dataset_pair(tmp_path)
    output = tmp_path / "pi05" / "toy_view"

    convert_dataset(annotated, output, data_source=physical)

    assert (output / "data").is_symlink()
    assert (output / "videos").is_symlink()
    assert (output / "meta" / "info.json").is_symlink()
    assert (output / "meta" / "episodes").is_symlink()
    assert (output / "meta" / "tasks.parquet").is_symlink()
    assert (output / "meta" / "subtasks.tsv").read_text() == (
        "0\tfirst stage\n1\tsecond stage\n"
    )

    sidecar_path = output / "subtask_indices" / "chunk-000" / "file-000.parquet"
    sidecar = pq.read_table(sidecar_path)
    assert sidecar.column_names == [
        "episode_index",
        "frame_index",
        "index",
        "subtask_index",
    ]
    assert sidecar["subtask_index"].to_pylist() == [0, 0, 1, 1]
    assert sidecar.schema.metadata[b"membench.schema"].decode() == SIDECAR_SCHEMA_VERSION
    metadata = pq.ParquetFile(sidecar_path).metadata
    assert {
        metadata.row_group(row_group).column(column).compression
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    } == {"SNAPPY"}


def test_convert_dataset_rejects_identity_mismatch(tmp_path: Path) -> None:
    physical, annotated = _write_dataset_pair(tmp_path)
    path = annotated / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    index_position = table.column_names.index("index")
    table = table.set_column(index_position, "index", pa.array([1, 2, 3, 4]))
    pq.write_table(table, path)

    try:
        convert_dataset(annotated, tmp_path / "bad_view", data_source=physical)
    except ValueError as exc:
        assert "index differs" in str(exc)
    else:
        raise AssertionError("identity mismatch was not rejected")
