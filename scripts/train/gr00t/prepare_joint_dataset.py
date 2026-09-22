#!/usr/bin/env python3
"""Build and validate the merged LeRobot dataset used for joint GR00T training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "configs/train/gr00t/release15_v1.json"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
MANIFEST_SCHEMA = "membench.gr00t_joint_manifest/v1"
PROVENANCE_SCHEMA = "membench.gr00t_joint_aggregate/v1"
VIDEO_ENCODER_INFO_KEYS = {
    "video.crf",
    "video.extra_options",
    "video.fast_decode",
    "video.g",
    "video.preset",
    "video.video_backend",
}


class ValidationError(RuntimeError):
    """Raised when source or aggregate data violates the training contract."""


@dataclass(frozen=True)
class DatasetSpec:
    task_id: str
    repo_id: str
    path: str
    expected_episodes: int
    expected_frames: int
    expected_task_prompts: int
    info_sha256: str
    tasks_sha256: str
    stats_sha256: str
    metadata_overrides: dict[str, Any]


@dataclass(frozen=True)
class JointManifest:
    path: Path
    payload: dict[str, Any]
    release_id: str
    aggregate: dict[str, Any]
    datasets: tuple[DatasetSpec, ...]


@dataclass(frozen=True)
class SourceReport:
    spec: DatasetSpec
    root: Path
    raw_info: dict[str, Any]
    normalized_info: dict[str, Any]
    task_prompts: tuple[str, ...]
    episode_tasks: tuple[tuple[str, ...], ...]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON file {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def load_manifest(path: Path) -> JointManifest:
    path = path.expanduser().resolve()
    payload = read_json(path)
    require(payload.get("schema_version") == MANIFEST_SCHEMA, f"unsupported manifest: {path}")

    aggregate = payload.get("aggregate")
    datasets = payload.get("datasets")
    require(isinstance(aggregate, dict), "manifest.aggregate must be an object")
    require(isinstance(datasets, list) and datasets, "manifest.datasets must be a non-empty list")

    specs: list[DatasetSpec] = []
    for index, item in enumerate(datasets):
        require(isinstance(item, dict), f"manifest.datasets[{index}] must be an object")
        relative_path = str(item.get("path", ""))
        relative = Path(relative_path)
        require(
            relative_path and not relative.is_absolute(),
            f"dataset path must be relative: {relative_path}",
        )
        require(
            ".." not in relative.parts,
            f"dataset path may not escape data root: {relative_path}",
        )
        overrides = item.get("metadata_overrides", {})
        require(
            isinstance(overrides, dict),
            f"metadata_overrides must be an object: {relative_path}",
        )
        try:
            spec = DatasetSpec(
                task_id=str(item["task_id"]),
                repo_id=str(item["repo_id"]),
                path=relative_path,
                expected_episodes=int(item["expected_episodes"]),
                expected_frames=int(item["expected_frames"]),
                expected_task_prompts=int(item["expected_task_prompts"]),
                info_sha256=str(item["info_sha256"]),
                tasks_sha256=str(item["tasks_sha256"]),
                stats_sha256=str(item["stats_sha256"]),
                metadata_overrides=overrides,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"invalid dataset entry at index {index}: {exc}") from exc
        require(spec.expected_episodes > 0, f"invalid episode count for {spec.repo_id}")
        require(spec.expected_frames > 0, f"invalid frame count for {spec.repo_id}")
        specs.append(spec)

    task_ids = [spec.task_id for spec in specs]
    repo_ids = [spec.repo_id for spec in specs]
    paths = [spec.path for spec in specs]
    require(len(task_ids) == len(set(task_ids)), "manifest contains duplicate task IDs")
    require(len(repo_ids) == len(set(repo_ids)), "manifest contains duplicate repo IDs")
    require(len(paths) == len(set(paths)), "manifest contains duplicate dataset paths")
    require(
        len(specs) == 15,
        f"release15 manifest must contain exactly 15 datasets, got {len(specs)}",
    )

    expected_episodes = sum(spec.expected_episodes for spec in specs)
    expected_frames = sum(spec.expected_frames for spec in specs)
    expected_prompts = sum(spec.expected_task_prompts for spec in specs)
    require(
        aggregate.get("expected_episodes") == expected_episodes,
        "aggregate episode total is stale",
    )
    require(aggregate.get("expected_frames") == expected_frames, "aggregate frame total is stale")
    require(
        aggregate.get("expected_source_task_prompts") == expected_prompts,
        "aggregate source prompt total is stale",
    )
    require(
        aggregate.get("canonical_features_repo_id") in repo_ids,
        "canonical feature dataset is not in the manifest",
    )
    require(aggregate.get("sampling") == "frame_uniform", "unsupported sampling strategy")

    return JointManifest(
        path=path,
        payload=payload,
        release_id=str(payload.get("release_id", "")),
        aggregate=aggregate,
        datasets=tuple(specs),
    )


def get_nested(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValidationError(f"metadata path does not exist: {dotted_path}")
        current = current[part]
    return current


def set_nested(value: dict[str, Any], dotted_path: str, replacement: Any) -> None:
    parts = dotted_path.split(".")
    current: Any = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ValidationError(f"metadata path does not exist: {dotted_path}")
        current = current[part]
    require(isinstance(current, dict), f"metadata parent is not an object: {dotted_path}")
    require(parts[-1] in current, f"metadata path does not exist: {dotted_path}")
    current[parts[-1]] = copy.deepcopy(replacement)


def normalized_info(spec: DatasetSpec, info: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(info)
    for dotted_path, replacement in spec.metadata_overrides.items():
        set_nested(result, dotted_path, replacement)
    return result


def merge_comparable_features(info: dict[str, Any]) -> dict[str, Any]:
    features = copy.deepcopy(info.get("features", {}))
    for feature in features.values():
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            continue
        encoder_info = feature.get("info")
        if isinstance(encoder_info, dict):
            for key in VIDEO_ENCODER_INFO_KEYS:
                encoder_info.pop(key, None)
    return features


def value_differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(value_differences(left[key], right[key], child))
        return result
    return [] if left == right else [path]


def assert_finite_json(value: Any, path: str = "stats") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite value at {path}: {value}")


def check_hash(path: Path, expected: str) -> None:
    require(path.is_file(), f"missing required file: {path}")
    actual = sha256_file(path)
    require(actual == expected, f"metadata hash mismatch for {path}: {actual} != {expected}")


def read_task_prompts(path: Path) -> tuple[str, ...]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["task_index", "task"])
    indices = table["task_index"].to_pylist()
    prompts = table["task"].to_pylist()
    require(indices == list(range(len(indices))), f"task indices are not contiguous: {path}")
    require(
        all(isinstance(prompt, str) and prompt for prompt in prompts),
        f"invalid task prompt: {path}",
    )
    require(len(prompts) == len(set(prompts)), f"duplicate task prompt: {path}")
    return tuple(prompts)


def read_episode_columns(root: Path, columns: list[str]) -> dict[str, list[Any]]:
    import pyarrow.dataset as pads

    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    require(files, f"no episode metadata found under {root}")
    combined = pads.dataset([str(path) for path in files], format="parquet").to_table(
        columns=columns
    )
    rows = combined.to_pydict()
    permutation = sorted(range(len(rows["episode_index"])), key=rows["episode_index"].__getitem__)
    return {name: [values[index] for index in permutation] for name, values in rows.items()}


def validate_episode_index(
    root: Path,
    expected_episodes: int,
    expected_frames: int,
    action_horizon: int,
) -> tuple[tuple[str, ...], ...]:
    columns = ["episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"]
    rows = read_episode_columns(root, columns)
    episode_indices = rows["episode_index"]
    lengths = rows["length"]
    from_indices = rows["dataset_from_index"]
    to_indices = rows["dataset_to_index"]
    require(len(episode_indices) == expected_episodes, f"wrong episode count under {root}")
    require(
        episode_indices == list(range(expected_episodes)),
        f"episode indices are not contiguous: {root}",
    )
    require(
        all(length >= action_horizon for length in lengths),
        f"episode shorter than the {action_horizon}-frame action horizon under {root}",
    )
    require(sum(lengths) == expected_frames, f"episode lengths do not sum to total frames: {root}")

    expected_start = 0
    for episode_index, (length, start, stop) in enumerate(zip(lengths, from_indices, to_indices)):
        require(start == expected_start, f"bad frame start for episode {episode_index}: {root}")
        require(stop == start + length, f"bad frame stop for episode {episode_index}: {root}")
        expected_start = stop
    require(expected_start == expected_frames, f"bad final frame boundary: {root}")

    episode_tasks: list[tuple[str, ...]] = []
    for episode_index, tasks in enumerate(rows["tasks"]):
        require(
            isinstance(tasks, list) and tasks,
            f"episode {episode_index} has no task prompt: {root}",
        )
        require(all(isinstance(task, str) and task for task in tasks), f"bad task prompt: {root}")
        episode_tasks.append(tuple(tasks))
    return tuple(episode_tasks)


def validate_source(manifest: JointManifest, spec: DatasetSpec, data_root: Path) -> SourceReport:
    root = (data_root / spec.path).resolve()
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    stats_path = root / "meta" / "stats.json"
    check_hash(info_path, spec.info_sha256)
    check_hash(tasks_path, spec.tasks_sha256)
    check_hash(stats_path, spec.stats_sha256)
    for directory in (root / "data", root / "meta" / "episodes", root / "videos"):
        require(directory.is_dir(), f"missing dataset directory: {directory}")

    info = read_json(info_path)
    aggregate = manifest.aggregate
    expected_values = {
        "codebase_version": aggregate["codebase_version"],
        "robot_type": aggregate["robot_type"],
        "fps": aggregate["fps"],
        "total_episodes": spec.expected_episodes,
        "total_frames": spec.expected_frames,
        "total_tasks": spec.expected_task_prompts,
    }
    for key, expected in expected_values.items():
        require(info.get(key) == expected, f"{spec.repo_id}: info.{key} != {expected!r}")

    features = info.get("features", {})
    require(isinstance(features, dict), f"{spec.repo_id}: info.features is invalid")
    for key, expected_dim in (
        ("observation.state", aggregate["state_dim"]),
        ("action", aggregate["action_dim"]),
    ):
        feature = features.get(key, {})
        require(feature.get("dtype") == "float32", f"{spec.repo_id}: invalid {key} dtype")
        require(feature.get("shape") == [expected_dim], f"{spec.repo_id}: invalid {key} shape")
    cameras = [key for key, value in features.items() if value.get("dtype") == "video"]
    require(len(cameras) == aggregate["video_cameras"], f"{spec.repo_id}: wrong camera count")
    for camera in cameras:
        require(features[camera].get("shape") == [224, 224, 3], f"{spec.repo_id}: bad camera shape")

    prompts = read_task_prompts(tasks_path)
    require(len(prompts) == spec.expected_task_prompts, f"{spec.repo_id}: wrong task prompt count")
    episode_tasks = validate_episode_index(
        root,
        spec.expected_episodes,
        spec.expected_frames,
        int(aggregate["action_horizon"]),
    )
    observed_prompts = {task for tasks in episode_tasks for task in tasks}
    require(
        observed_prompts == set(prompts),
        f"{spec.repo_id}: episode and task-table prompts differ",
    )
    assert_finite_json(read_json(stats_path))

    return SourceReport(
        spec=spec,
        root=root,
        raw_info=info,
        normalized_info=normalized_info(spec, info),
        task_prompts=prompts,
        episode_tasks=episode_tasks,
    )


def ensure_lerobot_version(expected: str) -> str:
    try:
        actual = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValidationError(
            "LeRobot is not installed; activate the membench conda environment"
        ) from exc
    require(actual == expected, f"LeRobot {expected} is required, got {actual}")
    return actual


def validate_sources(manifest: JointManifest, data_root: Path) -> list[SourceReport]:
    ensure_lerobot_version(str(manifest.aggregate["lerobot_version"]))
    data_root = data_root.expanduser().resolve()
    reports = []
    for spec in manifest.datasets:
        print(f"Checking {spec.task_id} ({spec.repo_id})...", file=sys.stderr, flush=True)
        reports.append(validate_source(manifest, spec, data_root))
    by_repo_id = {report.spec.repo_id: report for report in reports}
    canonical = by_repo_id[str(manifest.aggregate["canonical_features_repo_id"])]
    canonical_features = merge_comparable_features(canonical.raw_info)

    from lerobot.datasets.feature_utils import features_equal_for_merge

    for report in reports:
        raw_features = merge_comparable_features(report.raw_info)
        differences = value_differences(canonical_features, raw_features, "features")
        allowed = tuple(report.spec.metadata_overrides)
        unexpected = [
            path
            for path in differences
            if not any(path == prefix or path.startswith(f"{prefix}.") for prefix in allowed)
        ]
        require(
            not unexpected,
            f"{report.spec.repo_id}: unapproved feature differences: {', '.join(unexpected[:10])}",
        )
        for dotted_path, replacement in report.spec.metadata_overrides.items():
            canonical_value = get_nested(canonical.raw_info, dotted_path)
            require(
                replacement == canonical_value,
                f"{report.spec.repo_id}: override {dotted_path} differs from canonical metadata",
            )
        require(
            features_equal_for_merge(
                canonical.normalized_info["features"], report.normalized_info["features"]
            ),
            f"{report.spec.repo_id}: normalized features are not merge-compatible",
        )

    total_episodes = sum(report.spec.expected_episodes for report in reports)
    total_frames = sum(report.spec.expected_frames for report in reports)
    source_prompts = sum(len(report.task_prompts) for report in reports)
    unique_prompts = {prompt for report in reports for prompt in report.task_prompts}
    aggregate = manifest.aggregate
    require(total_episodes == aggregate["expected_episodes"], "source episode total mismatch")
    require(total_frames == aggregate["expected_frames"], "source frame total mismatch")
    require(
        source_prompts == aggregate["expected_source_task_prompts"],
        "source prompt total mismatch",
    )
    require(
        len(unique_prompts) == aggregate["expected_aggregate_task_prompts"],
        "deduplicated prompt total mismatch",
    )
    return reports


def print_source_report(manifest: JointManifest, reports: list[SourceReport]) -> None:
    print("task    repo_id                         episodes     frames prompts normalization")
    for report in reports:
        normalization = ",".join(report.spec.metadata_overrides) or "none"
        print(
            f"{report.spec.task_id:<7} {report.spec.repo_id:<31} "
            f"{report.spec.expected_episodes:>8} {report.spec.expected_frames:>10} "
            f"{len(report.task_prompts):>7} {normalization}"
        )
    aggregate = manifest.aggregate
    horizon_tail = (int(aggregate["action_horizon"]) - 1) * int(aggregate["expected_episodes"])
    usable_frames = int(aggregate["expected_frames"]) - horizon_tail
    duration_hours = int(aggregate["expected_frames"]) / int(aggregate["fps"]) / 3600
    print()
    print(
        f"PASS: {len(reports)} release tasks, {aggregate['expected_episodes']} episodes, "
        f"{aggregate['expected_frames']} frames ({duration_hours:.2f} h)"
    )
    print(
        f"GR00T horizon={aggregate['action_horizon']} leaves {usable_frames} trainable frame starts; "
        f"aggregate task prompts={aggregate['expected_aggregate_task_prompts']}"
    )


def aggregate_write_size_bytes(reports: list[SourceReport]) -> int:
    """Estimate newly allocated bytes; videos are hard-linked and consume no new data blocks."""
    total = 0
    for report in reports:
        for directory_name in ("data", "meta"):
            for path in (report.root / directory_name).rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
    return total


def create_normalized_view(report: SourceReport, staging_root: Path) -> Path:
    destination = staging_root / report.spec.repo_id
    destination.mkdir(parents=True)
    for directory_name in ("data", "videos"):
        source = (report.root / directory_name).resolve()
        (destination / directory_name).symlink_to(source, target_is_directory=True)

    source_meta = report.root / "meta"
    destination_meta = destination / "meta"
    destination_meta.mkdir()
    for source in source_meta.iterdir():
        target = destination_meta / source.name
        if source.name == "info.json":
            target.write_text(
                json.dumps(report.normalized_info, indent=4, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
        elif source.is_dir():
            target.symlink_to(source.resolve(), target_is_directory=True)
        else:
            shutil.copy2(source, target)
    return destination


def provenance_payload(manifest: JointManifest, reports: list[SourceReport]) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "release_id": manifest.release_id,
        "manifest_sha256": sha256_file(manifest.path),
        "aggregate_repo_id": manifest.aggregate["repo_id"],
        "lerobot_version": manifest.aggregate["lerobot_version"],
        "data_aggregation": "arrow_native_index_remap",
        "video_aggregation": "hardlink",
        "concatenate_data": False,
        "concatenate_videos": False,
        "sources": [
            {
                "task_id": report.spec.task_id,
                "repo_id": report.spec.repo_id,
                "episodes": report.spec.expected_episodes,
                "frames": report.spec.expected_frames,
                "metadata_overrides": report.spec.metadata_overrides,
            }
            for report in reports
        ],
    }


def remap_data_table(table: Any, src_meta: Any, dst_meta: Any) -> Any:
    """Offset aggregate indices without round-tripping Arrow extension arrays through pandas."""
    import pyarrow as pa
    import pyarrow.compute as pc

    result = table
    offsets = {
        "episode_index": int(dst_meta.info.total_episodes),
        "index": int(dst_meta.info.total_frames),
    }
    for column_name, offset in offsets.items():
        column_index = result.schema.get_field_index(column_name)
        require(column_index >= 0, f"data table is missing {column_name}")
        field = result.schema.field(column_index)
        shifted = pc.add(result[column_name], pa.scalar(offset, type=field.type))
        result = result.set_column(column_index, field, shifted)

    source_to_destination: dict[int, int] = {}
    for task_prompt, source_task_index in src_meta.tasks["task_index"].items():
        source_to_destination[int(source_task_index)] = int(
            dst_meta.tasks.loc[task_prompt, "task_index"]
        )
    task_column_index = result.schema.get_field_index("task_index")
    require(task_column_index >= 0, "data table is missing task_index")
    task_field = result.schema.field(task_column_index)
    source_indices = result["task_index"].to_numpy(zero_copy_only=False)
    try:
        destination_indices = [source_to_destination[int(value)] for value in source_indices]
    except KeyError as exc:
        raise ValidationError(f"data table contains an unknown task index: {exc.args[0]}") from exc
    result = result.set_column(
        task_column_index,
        task_field,
        pa.array(destination_indices, type=task_field.type),
    )
    return result


def aggregate_data_arrow_native(
    src_meta: Any,
    dst_meta: Any,
    data_idx: dict[str, Any],
    data_files_size_in_mb: float,
    chunk_size: int,
    concatenate_data: bool = True,
) -> dict[str, Any]:
    """LeRobot aggregate_data replacement for non-concatenated MemBench parquet files."""
    import pyarrow.parquet as pq
    from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
    from lerobot.datasets.utils import DEFAULT_DATA_PATH, update_chunk_file_indices

    del data_files_size_in_mb
    require(not concatenate_data, "Arrow-native aggregation requires concatenate_data=False")
    source_files = sorted(
        {
            (int(chunk), int(file))
            for chunk, file in zip(
                src_meta.episodes["data/chunk_index"],
                src_meta.episodes["data/file_index"],
            )
        }
    )
    source_to_destination: dict[tuple[int, int], tuple[int, int]] = {}

    for source_chunk, source_file in source_files:
        source_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=source_chunk,
            file_index=source_file,
        )
        destination_chunk = int(data_idx["chunk"])
        destination_file = int(data_idx["file"])
        destination_path = dst_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=destination_chunk,
            file_index=destination_file,
        )
        if destination_path.exists():
            destination_chunk, destination_file = update_chunk_file_indices(
                destination_chunk,
                destination_file,
                chunk_size,
            )
            data_idx["chunk"] = destination_chunk
            data_idx["file"] = destination_file
            destination_path = dst_meta.root / DEFAULT_DATA_PATH.format(
                chunk_index=destination_chunk,
                file_index=destination_file,
            )

        table = remap_data_table(pq.read_table(source_path), src_meta, dst_meta)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        write_table_one_row_group_per_episode(table, destination_path)
        source_to_destination[(source_chunk, source_file)] = (
            destination_chunk,
            destination_file,
        )

    data_idx["src_to_dst"] = source_to_destination
    return data_idx


def aggregate_videos_hardlink(
    src_meta: Any,
    dst_meta: Any,
    videos_idx: dict[str, dict[str, Any]],
    video_files_size_in_mb: float,
    chunk_size: int,
    concatenate_videos: bool = True,
) -> dict[str, dict[str, Any]]:
    """LeRobot aggregate_videos replacement that hard-links immutable source MP4 files."""
    from lerobot.datasets.utils import DEFAULT_VIDEO_PATH, update_chunk_file_indices
    from lerobot.datasets.video_utils import get_video_duration_in_s

    del video_files_size_in_mb
    require(not concatenate_videos, "Hard-link aggregation requires concatenate_videos=False")
    for video_index in videos_idx.values():
        video_index["episode_duration"] = 0.0
        video_index["src_to_offset"] = {}
        video_index["src_to_dst"] = {}
        video_index.setdefault("dst_file_durations", {})

    for video_key, video_index in videos_idx.items():
        source_files = sorted(
            {
                (int(chunk), int(file))
                for chunk, file in zip(
                    src_meta.episodes[f"videos/{video_key}/chunk_index"],
                    src_meta.episodes[f"videos/{video_key}/file_index"],
                )
            }
        )
        destination_chunk = int(video_index["chunk"])
        destination_file = int(video_index["file"])
        for source_chunk, source_file in source_files:
            source_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=source_chunk,
                file_index=source_file,
            )
            destination_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key,
                chunk_index=destination_chunk,
                file_index=destination_file,
            )
            if destination_path.exists():
                destination_chunk, destination_file = update_chunk_file_indices(
                    destination_chunk,
                    destination_file,
                    chunk_size,
                )
                destination_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=video_key,
                    chunk_index=destination_chunk,
                    file_index=destination_file,
                )

            destination_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_path, destination_path)
            except OSError as exc:
                raise ValidationError(
                    f"cannot hard-link {source_path} to {destination_path}: {exc}"
                ) from exc
            require(
                source_path.samefile(destination_path),
                f"video target is not a hard link of its source: {destination_path}",
            )
            duration = get_video_duration_in_s(source_path)
            source_key = (source_chunk, source_file)
            destination_key = (destination_chunk, destination_file)
            video_index["src_to_offset"][source_key] = 0.0
            video_index["src_to_dst"][source_key] = destination_key
            video_index["dst_file_durations"][destination_key] = duration
            video_index["episode_duration"] += duration

        video_index["chunk"] = destination_chunk
        video_index["file"] = destination_file
    return videos_idx


def check_referenced_files(root: Path, info: dict[str, Any], rows: dict[str, list[Any]]) -> None:
    data_path_template = info.get("data_path")
    video_path_template = info.get("video_path")
    require(isinstance(data_path_template, str), "aggregate info.data_path is missing")
    cameras = sorted(
        key for key, value in info["features"].items() if value.get("dtype") == "video"
    )
    require(isinstance(video_path_template, str), "aggregate info.video_path is missing")

    data_pairs = set(zip(rows["data/chunk_index"], rows["data/file_index"]))
    for chunk_index, file_index in data_pairs:
        path = root / data_path_template.format(chunk_index=chunk_index, file_index=file_index)
        require(path.is_file(), f"aggregate references missing data file: {path}")
    for camera in cameras:
        chunk_column = f"videos/{camera}/chunk_index"
        file_column = f"videos/{camera}/file_index"
        pairs = set(zip(rows[chunk_column], rows[file_column]))
        for chunk_index, file_index in pairs:
            path = root / video_path_template.format(
                video_key=camera,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            require(path.is_file(), f"aggregate references missing video file: {path}")


def decode_boundary_samples(
    manifest: JointManifest,
    reports: list[SourceReport],
    root: Path,
    rows: dict[str, list[Any]],
) -> None:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    aggregate = manifest.aggregate
    cameras = sorted(
        key for key, value in read_json(root / "meta" / "info.json")["features"].items()
        if value.get("dtype") == "video"
    )
    horizon = int(aggregate["action_horizon"])
    delta_timestamps = {"action": [index / int(aggregate["fps"]) for index in range(horizon)]}
    dataset = LeRobotDataset(
        str(aggregate["repo_id"]),
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )
    require(len(dataset) == aggregate["expected_frames"], "LeRobot aggregate frame count mismatch")
    require(
        dataset.num_episodes == aggregate["expected_episodes"],
        "LeRobot episode count mismatch",
    )

    episode_offset = 0
    for report in reports:
        sample_index = int(rows["dataset_from_index"][episode_offset])
        sample = dataset[sample_index]
        require(tuple(sample["observation.state"].shape) == (aggregate["state_dim"],), "bad state")
        require(tuple(sample["action"].shape) == (horizon, aggregate["action_dim"]), "bad action")
        require(isinstance(sample.get("task"), str) and sample["task"], "sample has no task prompt")
        for camera in cameras:
            image = sample[camera]
            require(tuple(image.shape) == (3, 224, 224), f"bad decoded image for {camera}")
            require(image.dtype == torch.uint8, f"decoded image is not uint8 for {camera}")
        episode_offset += report.spec.expected_episodes


def validate_aggregate(
    manifest: JointManifest,
    reports: list[SourceReport],
    root: Path,
    *,
    check_files: bool,
    decode_samples: bool,
) -> None:
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    tasks_path = root / "meta" / "tasks.parquet"
    stats_path = root / "meta" / "stats.json"
    provenance_path = root / "meta" / "membench_aggregation.json"
    for path in (info_path, tasks_path, stats_path, provenance_path):
        require(path.is_file(), f"aggregate is incomplete; missing {path}")

    info = read_json(info_path)
    aggregate = manifest.aggregate
    expected_values = {
        "codebase_version": aggregate["codebase_version"],
        "robot_type": aggregate["robot_type"],
        "fps": aggregate["fps"],
        "total_episodes": aggregate["expected_episodes"],
        "total_frames": aggregate["expected_frames"],
        "total_tasks": aggregate["expected_aggregate_task_prompts"],
    }
    for key, expected in expected_values.items():
        require(info.get(key) == expected, f"aggregate info.{key} != {expected!r}")

    from lerobot.datasets.feature_utils import features_equal_for_merge

    canonical_repo_id = str(aggregate["canonical_features_repo_id"])
    canonical = next(report for report in reports if report.spec.repo_id == canonical_repo_id)
    require(
        features_equal_for_merge(canonical.normalized_info["features"], info["features"]),
        "aggregate feature schema differs from the canonical source",
    )
    prompts = read_task_prompts(tasks_path)
    expected_prompts = {prompt for report in reports for prompt in report.task_prompts}
    require(set(prompts) == expected_prompts, "aggregate task prompts differ from source union")
    require(
        len(prompts) == aggregate["expected_aggregate_task_prompts"],
        "wrong aggregate prompt count",
    )
    assert_finite_json(read_json(stats_path))

    provenance = read_json(provenance_path)
    require(
        provenance.get("schema_version") == PROVENANCE_SCHEMA,
        "bad aggregate provenance schema",
    )
    require(
        provenance.get("manifest_sha256") == sha256_file(manifest.path),
        "manifest hash mismatch",
    )

    cameras = sorted(
        key for key, value in info["features"].items() if value.get("dtype") == "video"
    )
    episode_columns = [
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "data/chunk_index",
        "data/file_index",
    ]
    for camera in cameras:
        episode_columns.extend(
            [f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"]
        )
    rows = read_episode_columns(root, episode_columns)
    episode_indices = rows["episode_index"]
    lengths = rows["length"]
    require(
        episode_indices == list(range(aggregate["expected_episodes"])),
        "aggregate episode indices are not contiguous",
    )
    require(sum(lengths) == aggregate["expected_frames"], "aggregate episode lengths are wrong")
    require(
        all(length >= aggregate["action_horizon"] for length in lengths),
        "aggregate contains an episode shorter than the action horizon",
    )
    expected_start = 0
    for episode_index, (length, start, stop) in enumerate(
        zip(lengths, rows["dataset_from_index"], rows["dataset_to_index"])
    ):
        require(start == expected_start, f"aggregate episode {episode_index} starts at {start}")
        require(stop == start + length, f"aggregate episode {episode_index} has a bad stop index")
        expected_start = stop

    episode_offset = 0
    for report in reports:
        block = rows["tasks"][episode_offset : episode_offset + report.spec.expected_episodes]
        block_prompts = {prompt for tasks in block for prompt in tasks}
        require(
            block_prompts == set(report.task_prompts),
            f"aggregate episode block no longer matches {report.spec.task_id}",
        )
        episode_offset += report.spec.expected_episodes
    if check_files:
        check_referenced_files(root, info, rows)
    if decode_samples:
        decode_boundary_samples(manifest, reports, root, rows)

    print(
        f"PASS: aggregate {root} has {len(reports)} tasks, {aggregate['expected_episodes']} episodes, "
        f"{aggregate['expected_frames']} frames, and {len(prompts)} unique prompts"
    )


def aggregate_dataset(
    manifest: JointManifest,
    reports: list[SourceReport],
    output_root: Path,
    reserve_free_gib: float,
) -> None:
    output_root = output_root.expanduser().resolve()
    require(not output_root.exists(), f"aggregate output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    working_root = output_root.parent / f".{output_root.name}.building-{os.getpid()}"
    require(not working_root.exists(), f"temporary aggregate path already exists: {working_root}")

    estimated_write_bytes = aggregate_write_size_bytes(reports)
    free_bytes = shutil.disk_usage(output_root.parent).free
    reserve_bytes = int(reserve_free_gib * 1024**3)
    required_bytes = estimated_write_bytes + reserve_bytes
    require(
        free_bytes >= required_bytes,
        f"insufficient disk: need about {required_bytes / 1024**3:.1f} GiB free, "
        f"have {free_bytes / 1024**3:.1f} GiB",
    )
    print(
        f"Hard-linking videos and rewriting about {estimated_write_bytes / 1024**3:.2f} GiB "
        f"of data/metadata into {working_root}; {free_bytes / 1024**3:.1f} GiB is free",
        flush=True,
    )

    import lerobot.datasets.aggregate as aggregate_module

    # LeRobot 0.6.1 converts non-image data through pandas while aggregating.
    # That loses the HF Array2D dtype carried by MemBench camera matrices. Keep
    # the official orchestration and metadata logic, but remap the three index
    # columns with Arrow and hard-link immutable videos to avoid a 56 GiB copy.
    aggregate_module.aggregate_data = aggregate_data_arrow_native
    aggregate_module.aggregate_videos = aggregate_videos_hardlink

    try:
        with tempfile.TemporaryDirectory(prefix="membench-release15-") as temporary:
            staging_root = Path(temporary)
            roots: list[Path] = []
            for report in reports:
                if report.spec.metadata_overrides:
                    roots.append(create_normalized_view(report, staging_root))
                else:
                    roots.append(report.root)
            aggregate_module.aggregate_datasets(
                repo_ids=[report.spec.repo_id for report in reports],
                aggr_repo_id=str(manifest.aggregate["repo_id"]),
                roots=roots,
                aggr_root=working_root,
                concatenate_videos=False,
                concatenate_data=False,
            )
        provenance_path = working_root / "meta" / "membench_aggregation.json"
        provenance_path.write_text(
            json.dumps(provenance_payload(manifest, reports), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        validate_aggregate(
            manifest,
            reports,
            working_root,
            check_files=True,
            decode_samples=False,
        )
        working_root.rename(output_root)
    except Exception:
        print(f"Aggregate failed; partial output is retained at {working_root}", file=sys.stderr)
        raise
    print(f"Aggregate ready: {output_root}")


def configure_cache() -> None:
    cache_root = REPO_ROOT / "outputs" / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = common_parser()

    subparsers.add_parser("check", parents=[common], help="validate all source datasets")

    aggregate = subparsers.add_parser(
        "aggregate",
        parents=[common],
        help="create the immutable merged LeRobot dataset",
    )
    aggregate.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "membench_release15_v1",
    )
    aggregate.add_argument(
        "--reserve-free-gib",
        type=float,
        default=100.0,
        help="free space that must remain in addition to the estimated aggregate size",
    )

    validate = subparsers.add_parser(
        "validate",
        parents=[common],
        help="validate an existing merged dataset",
    )
    validate.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "membench_release15_v1",
    )
    validate.add_argument("--decode-samples", action="store_true")
    validate.add_argument("--skip-file-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cache()
    manifest = load_manifest(args.manifest)
    reports = validate_sources(manifest, args.data_root)
    if args.command == "check":
        print_source_report(manifest, reports)
    elif args.command == "aggregate":
        aggregate_dataset(manifest, reports, args.output_root, args.reserve_free_gib)
    elif args.command == "validate":
        validate_aggregate(
            manifest,
            reports,
            args.dataset_root,
            check_files=not args.skip_file_check,
            decode_samples=args.decode_samples,
        )
    else:
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
