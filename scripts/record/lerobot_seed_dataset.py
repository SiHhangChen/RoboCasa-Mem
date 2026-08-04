"""Transactional helpers for seed-addressable LeRobot v3 datasets."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SEED_REGISTRY_PATH = Path("meta/seeds.jsonl")
MUTABLE_GLOBAL_PATHS = (
    Path("meta/info.json"),
    Path("meta/stats.json"),
    Path("meta/tasks.parquet"),
    SEED_REGISTRY_PATH,
)


def parse_seed_values(value: str) -> list[int]:
    seeds: list[int] = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(item))
    return list(dict.fromkeys(seeds))


def load_seed_registry(root: Path) -> list[dict[str, Any]]:
    path = Path(root) / SEED_REGISTRY_PATH
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "seed" not in record or "episode_index" not in record:
            raise ValueError(f"invalid seed registry row {line_number}: {record}")
        records.append(record)
    return sorted(records, key=lambda item: int(item["episode_index"]))


def write_seed_registry(root: Path, records: list[dict[str, Any]]) -> None:
    path = Path(root) / SEED_REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: int(item["episode_index"]))
    payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in ordered)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def make_seed_record(
    *,
    seed: int,
    episode_index: int,
    task_id: str,
    task: str,
    num_frames: int,
) -> dict[str, Any]:
    return {
        "episode_uid": str(uuid.uuid4()),
        "seed": int(seed),
        "episode_index": int(episode_index),
        "task_id": str(task_id),
        "task": str(task),
        "num_frames": int(num_frames),
        "created_at_unix": time.time(),
    }


@contextlib.contextmanager
def dataset_lock(root: Path) -> Iterator[None]:
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def make_staging_root(target: Path) -> tuple[Path, Path]:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    container = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    return container, container / target.name


def clone_with_hardlinks(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=os.link)
    for relative_path in MUTABLE_GLOBAL_PATHS:
        path = destination / relative_path
        if not path.exists():
            continue
        replacement = path.with_name(path.name + ".copy")
        shutil.copy2(path, replacement)
        os.replace(replacement, path)


def install_staged_dataset(
    staged_root: Path,
    target_root: Path,
    *,
    keep_backup: bool,
    backup_label: str | None = None,
) -> Path | None:
    staged_root = Path(staged_root)
    target_root = Path(target_root)
    backup: Path | None = None
    if target_root.exists():
        if keep_backup:
            suffix = backup_label or time.strftime("%Y%m%d-%H%M%S")
            backup = target_root.with_name(f"{target_root.name}.backup-{suffix}")
            if backup.exists():
                raise FileExistsError(f"backup already exists: {backup}")
        else:
            backup = target_root.with_name(
                f".{target_root.name}.replaced-{uuid.uuid4().hex[:8]}"
            )
        os.replace(target_root, backup)
    try:
        os.replace(staged_root, target_root)
    except Exception:
        if backup is not None and backup.exists() and not target_root.exists():
            os.replace(backup, target_root)
        raise
    if backup is not None and not keep_backup:
        shutil.rmtree(backup)
        return None
    return backup


def episode_rows(root: Path) -> list[dict[str, Any]]:
    paths = sorted((Path(root) / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        return []
    table = pq.read_table(paths)
    return sorted(table.to_pylist(), key=lambda item: int(item["episode_index"]))


def chunk_file_path(
    template: str,
    *,
    chunk_index: int,
    file_index: int,
    video_key: str | None = None,
) -> Path:
    values: dict[str, Any] = {
        "chunk_index": int(chunk_index),
        "file_index": int(file_index),
    }
    if video_key is not None:
        values["video_key"] = video_key
    return Path(template.format(**values))


def write_single_row_parquet(path: Path, row: dict[str, Any], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(
        pa.Table.from_pylist([row], schema=schema),
        temporary,
        compression="snappy",
        use_dictionary=True,
    )
    os.replace(temporary, path)


def _replace_int_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(
        index,
        name,
        pa.array(values, type=table.schema.field(name).type),
    )


def _map_nested(value: Any, transform: Any) -> Any:
    if isinstance(value, list):
        return [_map_nested(item, transform) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return transform(value)
    return value


def _set_episode_index_stats(row: dict[str, Any], episode_index: int) -> None:
    for key in list(row):
        if not key.startswith("stats/episode_index/"):
            continue
        if key.endswith("/count"):
            continue
        if key.endswith("/std"):
            row[key] = _map_nested(row[key], lambda _: 0.0)
        else:
            row[key] = _map_nested(row[key], lambda _: float(episode_index))


def _shift_global_index_stats(row: dict[str, Any], delta: int) -> None:
    for key in list(row):
        if not key.startswith("stats/index/") or key.endswith(("/count", "/std")):
            continue
        row[key] = _map_nested(row[key], lambda value: value + delta)


def stats_from_episode_rows(
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from lerobot.datasets.compute_stats import aggregate_stats

    all_stats: list[dict[str, dict[str, np.ndarray]]] = []
    for row in rows:
        episode_stats: dict[str, dict[str, np.ndarray]] = {}
        for key, value in row.items():
            if not key.startswith("stats/"):
                continue
            feature_name, stat_name = key.removeprefix("stats/").rsplit("/", 1)
            if feature_name not in features:
                continue
            episode_stats.setdefault(feature_name, {})[stat_name] = np.asarray(value)
        all_stats.append(episode_stats)
    return aggregate_stats(all_stats)


def _depth_stats_from_parquet(
    paths: list[Path],
    feature_name: str,
    quantiles: tuple[tuple[str, float], ...],
) -> dict[str, np.ndarray]:
    """Compute exact uint16 depth statistics without loading all frames at once."""

    histogram = np.zeros(np.iinfo(np.uint16).max + 1, dtype=np.uint64)
    frame_count = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        if feature_name not in parquet.schema_arrow.names:
            raise KeyError(f"required depth feature missing from {path}: {feature_name}")
        for batch in parquet.iter_batches(batch_size=32, columns=[feature_name]):
            column = batch.column(0)
            if column.null_count:
                raise ValueError(f"depth feature contains null frames: {feature_name}")
            values = np.asarray(column.to_pylist())
            frame_count += batch.num_rows
            if not values.size:
                continue
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(
                    f"depth feature must contain integers, got {values.dtype}: "
                    f"{feature_name}"
                )
            minimum = int(values.min())
            maximum = int(values.max())
            if minimum < 0 or maximum > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"depth feature values must fit uint16, got [{minimum}, {maximum}]: "
                    f"{feature_name}"
                )
            counts = np.bincount(
                values.reshape(-1).astype(np.int64, copy=False),
                minlength=histogram.size,
            )
            histogram += counts.astype(np.uint64, copy=False)

    pixel_count = int(histogram.sum())
    if frame_count == 0 or pixel_count == 0:
        raise ValueError(f"depth feature contains no values: {feature_name}")

    levels = np.arange(histogram.size, dtype=np.float64)
    weights = histogram.astype(np.float64)
    mean = float(np.dot(levels, weights) / pixel_count)
    variance = float(np.dot(np.square(levels - mean), weights) / pixel_count)
    occupied = np.flatnonzero(histogram)
    stats = {
        "min": np.array([float(occupied[0])], dtype=np.float64),
        "max": np.array([float(occupied[-1])], dtype=np.float64),
        "mean": np.array([mean], dtype=np.float64),
        "std": np.array([np.sqrt(max(0.0, variance))], dtype=np.float64),
        "count": np.array([frame_count], dtype=np.int64),
    }
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    for name, quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {quantile}")
        position = (pixel_count - 1) * quantile
        lower_index = int(np.floor(position))
        upper_index = int(np.ceil(position))
        lower = float(np.searchsorted(cumulative, lower_index + 1, side="left"))
        upper = float(np.searchsorted(cumulative, upper_index + 1, side="left"))
        value = lower + (upper - lower) * (position - lower_index)
        stats[name] = np.array([value], dtype=np.float64)
    return stats


def recompute_global_stats(
    root: Path,
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute exact numeric stats from final parquet rows; retain standard video aggregates."""
    stats = stats_from_episode_rows(rows, features)
    data_paths = sorted((Path(root) / "data").glob("chunk-*/*.parquet"))
    if not data_paths:
        raise ValueError(f"dataset contains no parquet data files: {root}")
    column_names = set(pq.read_schema(data_paths[0]).names)
    quantiles = (
        ("q01", 0.01),
        ("q10", 0.10),
        ("q50", 0.50),
        ("q90", 0.90),
        ("q99", 0.99),
    )
    for feature_name, feature in features.items():
        if feature["dtype"] in {"video", "image", "string", "language"}:
            continue
        if feature_name not in column_names:
            continue
        if feature["dtype"] == "uint16" and feature_name.startswith(
            "observation.depth."
        ):
            stats[feature_name] = _depth_stats_from_parquet(
                data_paths,
                feature_name,
                quantiles,
            )
            continue
        column = pq.read_table(data_paths, columns=[feature_name])[feature_name]
        values = np.asarray(column.to_pylist())
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        numeric = values.astype(np.float64, copy=False)
        feature_stats: dict[str, np.ndarray] = {
            "min": np.min(numeric, axis=0),
            "max": np.max(numeric, axis=0),
            "mean": np.mean(numeric, axis=0),
            "std": np.std(numeric, axis=0),
            "count": np.array([numeric.shape[0]], dtype=np.int64),
        }
        for key, quantile in quantiles:
            feature_stats[key] = np.quantile(numeric, quantile, axis=0)
        stats[feature_name] = feature_stats
    return stats


def canonicalize_seed_order(
    root: Path,
    registry: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Sort episodes by seed and compact all file/global indices without re-encoding video."""
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH

    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    rows = episode_rows(root)
    rows_by_index = {int(row["episode_index"]): row for row in rows}
    registry = list(registry if registry is not None else load_seed_registry(root))
    ordered_registry = sorted(registry, key=lambda item: int(item["seed"]))
    selected_indices = [int(item["episode_index"]) for item in ordered_registry]
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError(f"duplicate episode indices in registry: {selected_indices}")
    missing_indices = sorted(set(selected_indices) - set(rows_by_index))
    if missing_indices:
        raise ValueError(f"registry references missing episodes: {missing_indices}")

    video_keys = [
        key for key, feature in info["features"].items() if feature["dtype"] == "video"
    ]
    container = Path(tempfile.mkdtemp(prefix=".canonical-", dir=root))
    canonical_root = container / "dataset"
    new_rows: list[dict[str, Any]] = []
    new_registry: list[dict[str, Any]] = []
    global_index = 0
    try:
        for new_episode_index, registry_item in enumerate(ordered_registry):
            previous_episode_index = int(registry_item["episode_index"])
            row = dict(rows_by_index[previous_episode_index])
            length = int(row["length"])
            new_chunk_index, new_file_index = divmod(
                new_episode_index, int(info["chunks_size"])
            )

            previous_data_path = root / chunk_file_path(
                info["data_path"],
                chunk_index=int(row["data/chunk_index"]),
                file_index=int(row["data/file_index"]),
            )
            data = pq.read_table(previous_data_path)
            data = _replace_int_column(
                data,
                "episode_index",
                np.full(data.num_rows, new_episode_index, dtype=np.int64),
            )
            data = _replace_int_column(
                data,
                "index",
                np.arange(global_index, global_index + data.num_rows, dtype=np.int64),
            )
            new_data_path = canonical_root / chunk_file_path(
                info["data_path"],
                chunk_index=new_chunk_index,
                file_index=new_file_index,
            )
            new_data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                data,
                new_data_path,
                compression="snappy",
                use_dictionary=True,
            )

            previous_from_index = int(row["dataset_from_index"])
            row["episode_index"] = new_episode_index
            row["data/chunk_index"] = new_chunk_index
            row["data/file_index"] = new_file_index
            row["meta/episodes/chunk_index"] = new_chunk_index
            row["meta/episodes/file_index"] = new_file_index
            row["dataset_from_index"] = global_index
            row["dataset_to_index"] = global_index + length
            _set_episode_index_stats(row, new_episode_index)
            _shift_global_index_stats(row, global_index - previous_from_index)

            for video_key in video_keys:
                previous_video_path = root / chunk_file_path(
                    info["video_path"],
                    video_key=video_key,
                    chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
                    file_index=int(row[f"videos/{video_key}/file_index"]),
                )
                new_video_path = canonical_root / chunk_file_path(
                    info["video_path"],
                    video_key=video_key,
                    chunk_index=new_chunk_index,
                    file_index=new_file_index,
                )
                new_video_path.parent.mkdir(parents=True, exist_ok=True)
                os.link(previous_video_path, new_video_path)
                row[f"videos/{video_key}/chunk_index"] = new_chunk_index
                row[f"videos/{video_key}/file_index"] = new_file_index

            previous_meta_path = root / DEFAULT_EPISODES_PATH.format(
                chunk_index=int(rows_by_index[previous_episode_index]["meta/episodes/chunk_index"]),
                file_index=int(rows_by_index[previous_episode_index]["meta/episodes/file_index"]),
            )
            new_meta_path = canonical_root / DEFAULT_EPISODES_PATH.format(
                chunk_index=new_chunk_index,
                file_index=new_file_index,
            )
            write_single_row_parquet(new_meta_path, row, pq.read_schema(previous_meta_path))
            new_rows.append(row)

            updated_registry = dict(registry_item)
            updated_registry["episode_index"] = new_episode_index
            new_registry.append(updated_registry)
            global_index += length

        shutil.rmtree(root / "data")
        shutil.rmtree(root / "videos")
        shutil.rmtree(root / "meta" / "episodes")
        os.replace(canonical_root / "data", root / "data")
        os.replace(canonical_root / "videos", root / "videos")
        os.replace(canonical_root / "meta" / "episodes", root / "meta" / "episodes")

        info["total_episodes"] = len(new_rows)
        info["total_frames"] = global_index
        info["splits"] = {"train": f"0:{len(new_rows)}"}
        temporary_info = info_path.with_name(info_path.name + ".tmp")
        temporary_info.write_text(json.dumps(info, indent=4) + "\n")
        os.replace(temporary_info, info_path)
        write_seed_registry(root, new_registry)
        write_stats(recompute_global_stats(root, new_rows, info["features"]), root)
    finally:
        shutil.rmtree(container, ignore_errors=True)
    return new_registry


def validate_seed_dataset(root: Path, *, require_registry: bool = True) -> dict[str, Any]:
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    rows = episode_rows(root)
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    if len(rows) != total_episodes:
        raise ValueError(f"episode metadata count {len(rows)} != {total_episodes}")
    indices = [int(row["episode_index"]) for row in rows]
    if indices != list(range(total_episodes)):
        raise ValueError(f"episode indices are not contiguous: {indices}")

    video_keys = [
        key for key, feature in info["features"].items() if feature["dtype"] == "video"
    ]
    data_refs: set[tuple[int, int]] = set()
    meta_refs: set[tuple[int, int]] = set()
    video_refs: dict[str, set[tuple[int, int]]] = {key: set() for key in video_keys}
    frame_count = 0
    expected_global_index = 0
    for row in rows:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        expected_ref = divmod(episode_index, int(info["chunks_size"]))
        frame_count += length
        if int(row["dataset_from_index"]) != expected_global_index:
            raise ValueError(f"invalid dataset_from_index for episode {episode_index}")
        expected_global_index += length
        if int(row["dataset_to_index"]) != expected_global_index:
            raise ValueError(f"invalid dataset_to_index for episode {episode_index}")

        data_ref = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        if data_ref != expected_ref:
            raise ValueError(
                f"non-canonical data file for episode {episode_index}: "
                f"{data_ref} != {expected_ref}"
            )
        if data_ref in data_refs:
            raise ValueError(f"data file is shared by multiple episodes: {data_ref}")
        data_refs.add(data_ref)
        data_path = root / chunk_file_path(
            info["data_path"], chunk_index=data_ref[0], file_index=data_ref[1]
        )
        data = pq.read_table(data_path, columns=["episode_index", "index"])
        if data.num_rows != length:
            raise ValueError(f"data length mismatch for episode {episode_index}")
        if set(data["episode_index"].to_pylist()) != {episode_index}:
            raise ValueError(f"data episode_index mismatch for episode {episode_index}")
        if data["index"].to_pylist() != list(
            range(int(row["dataset_from_index"]), int(row["dataset_to_index"]))
        ):
            raise ValueError(f"global frame index mismatch for episode {episode_index}")

        meta_ref = (
            int(row["meta/episodes/chunk_index"]),
            int(row["meta/episodes/file_index"]),
        )
        if meta_ref != expected_ref:
            raise ValueError(
                f"non-canonical metadata file for episode {episode_index}: "
                f"{meta_ref} != {expected_ref}"
            )
        if meta_ref in meta_refs:
            raise ValueError(f"metadata file is shared by multiple episodes: {meta_ref}")
        meta_refs.add(meta_ref)

        for video_key in video_keys:
            video_ref = (
                int(row[f"videos/{video_key}/chunk_index"]),
                int(row[f"videos/{video_key}/file_index"]),
            )
            if video_ref != expected_ref:
                raise ValueError(
                    f"non-canonical video file for episode {episode_index}, "
                    f"{video_key}: {video_ref} != {expected_ref}"
                )
            if video_ref in video_refs[video_key]:
                raise ValueError(
                    f"video file is shared by multiple episodes for {video_key}: {video_ref}"
                )
            video_refs[video_key].add(video_ref)
            video_path = root / chunk_file_path(
                info["video_path"],
                video_key=video_key,
                chunk_index=video_ref[0],
                file_index=video_ref[1],
            )
            if not video_path.is_file() or video_path.stat().st_size == 0:
                raise FileNotFoundError(f"missing video for episode {episode_index}: {video_path}")
            if abs(float(row[f"videos/{video_key}/from_timestamp"])) > 1e-6:
                raise ValueError(f"video does not start at zero for episode {episode_index}")

    if frame_count != total_frames:
        raise ValueError(f"frame count {frame_count} != {total_frames}")

    stored_stats = json.loads((root / "meta" / "stats.json").read_text())
    expected_stats = recompute_global_stats(root, rows, info["features"])
    for feature_name, feature in info["features"].items():
        if feature["dtype"] in {"video", "image", "string", "language"}:
            continue
        if feature_name not in expected_stats or feature_name not in stored_stats:
            raise ValueError(f"numeric stats missing for {feature_name}")
        for stat_name, expected_value in expected_stats[feature_name].items():
            actual_value = np.asarray(stored_stats[feature_name][stat_name])
            if not np.allclose(
                actual_value,
                np.asarray(expected_value),
                rtol=1e-7,
                atol=1e-9,
            ):
                raise ValueError(
                    f"numeric stats mismatch for {feature_name}/{stat_name}: "
                    f"stored={actual_value.tolist()} expected={np.asarray(expected_value).tolist()}"
                )

    registry = load_seed_registry(root)
    if require_registry:
        if len(registry) != total_episodes:
            raise ValueError(f"seed registry count {len(registry)} != {total_episodes}")
        registry_indices = [int(item["episode_index"]) for item in registry]
        if registry_indices != list(range(total_episodes)):
            raise ValueError(f"seed registry indices are not contiguous: {registry_indices}")
        seeds = [int(item["seed"]) for item in registry]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds in registry: {seeds}")
        if seeds != sorted(seeds):
            raise ValueError(f"seed registry is not sorted: {seeds}")

    return {
        "root": str(root),
        "episodes": total_episodes,
        "frames": total_frames,
        "seeds": [int(item["seed"]) for item in registry],
        "video_keys": video_keys,
    }
