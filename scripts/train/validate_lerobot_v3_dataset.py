#!/usr/bin/env python3
"""Validate a local LeRobot v3 dataset before a long training run."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-fps", type=int, required=True)
    parser.add_argument("--expected-state-dim", type=int, required=True)
    parser.add_argument("--expected-action-dim", type=int, required=True)
    parser.add_argument("--expected-cameras", type=int, required=True)
    parser.add_argument("--action-horizon", type=int, default=40)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"unexpected indexed filename: {path}") from exc


def indexed_files(directory: Path, suffix: str) -> list[Path]:
    files = sorted(directory.rglob(f"*{suffix}"), key=file_index)
    actual = [file_index(path) for path in files]
    require(actual == list(range(len(files))), f"non-contiguous files under {directory}: {actual}")
    return files


def assert_finite_json(value: Any, path: str = "stats") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite value at {path}: {value}")


def validate_info(args: argparse.Namespace, root: Path) -> tuple[dict[str, Any], list[str]]:
    info_path = root / "meta" / "info.json"
    require(info_path.is_file(), f"missing dataset metadata: {info_path}")
    info = json.loads(info_path.read_text())

    expected_values = {
        "codebase_version": "v3.0",
        "total_episodes": args.expected_episodes,
        "fps": args.expected_fps,
    }
    for key, expected in expected_values.items():
        require(
            info.get(key) == expected,
            f"info.{key}: expected {expected!r}, got {info.get(key)!r}",
        )

    features = info.get("features", {})
    expected_features = {
        "observation.state": ("float32", [args.expected_state_dim]),
        "action": ("float32", [args.expected_action_dim]),
    }
    for key, (dtype, shape) in expected_features.items():
        feature = features.get(key, {})
        require(feature.get("dtype") == dtype, f"{key} dtype is not {dtype}: {feature}")
        require(feature.get("shape") == shape, f"{key} shape is not {shape}: {feature}")

    camera_keys = sorted(key for key, value in features.items() if value.get("dtype") == "video")
    require(
        len(camera_keys) == args.expected_cameras,
        f"expected {args.expected_cameras} video cameras, found {camera_keys}",
    )
    for key in camera_keys:
        require(features[key].get("shape") == [224, 224, 3], f"unexpected camera shape: {key}")

    stats_path = root / "meta" / "stats.json"
    require(stats_path.is_file(), f"missing dataset statistics: {stats_path}")
    assert_finite_json(json.loads(stats_path.read_text()))
    return info, camera_keys


def validate_tables(
    args: argparse.Namespace,
    root: Path,
) -> tuple[list[int], Counter[int], int]:
    import numpy as np
    import pyarrow.parquet as pq

    data_files = indexed_files(root / "data", ".parquet")
    episode_files = indexed_files(root / "meta" / "episodes", ".parquet")
    require(len(data_files) == args.expected_episodes, f"found {len(data_files)} data files")
    require(
        len(episode_files) == args.expected_episodes,
        f"found {len(episode_files)} episode files",
    )

    columns = [
        "observation.state",
        "action",
        "next.terminated",
        "next.done",
        "next.success",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    lengths: list[int] = []
    task_episode_counts: Counter[int] = Counter()
    global_index = 0

    for episode_index, path in enumerate(data_files):
        table = pq.read_table(path, columns=columns)
        size = len(table)
        require(size > args.action_horizon, f"episode {episode_index} is too short: {size}")
        lengths.append(size)

        state_values = table["observation.state"].combine_chunks().values
        action_values = table["action"].combine_chunks().values
        state = state_values.to_numpy(zero_copy_only=False).reshape(size, -1)
        action = action_values.to_numpy(zero_copy_only=False).reshape(size, -1)
        require(state.shape == (size, args.expected_state_dim), f"bad state shape in {path}")
        require(action.shape == (size, args.expected_action_dim), f"bad action shape in {path}")
        require(np.isfinite(state).all(), f"non-finite state value in {path}")
        require(np.isfinite(action).all(), f"non-finite action value in {path}")

        frame_indices = table["frame_index"].to_numpy()
        episode_indices = table["episode_index"].to_numpy()
        global_indices = table["index"].to_numpy()
        timestamps = table["timestamp"].to_numpy()
        require(np.array_equal(frame_indices, np.arange(size)), f"bad frame indices in {path}")
        require(np.all(episode_indices == episode_index), f"bad episode indices in {path}")
        require(
            np.array_equal(global_indices, np.arange(global_index, global_index + size)),
            f"bad global indices in {path}",
        )
        require(
            np.allclose(timestamps, np.arange(size) / args.expected_fps, atol=1e-4),
            f"bad timestamps in {path}",
        )

        task_indices = np.unique(table["task_index"].to_numpy())
        require(len(task_indices) == 1, f"episode {episode_index} contains multiple tasks")
        task_episode_counts[int(task_indices[0])] += 1
        require(
            bool(table["next.terminated"][-1].as_py()),
            f"episode {episode_index} is not terminated",
        )
        require(bool(table["next.done"][-1].as_py()), f"episode {episode_index} is not done")
        require(
            bool(table["next.success"][-1].as_py()),
            f"episode {episode_index} is not successful",
        )
        global_index += size

    for episode_index, path in enumerate(episode_files):
        table = pq.read_table(path, columns=["episode_index", "length"])
        require(len(table) == 1, f"expected one metadata row in {path}, got {len(table)}")
        require(table["episode_index"][0].as_py() == episode_index, f"bad metadata index in {path}")
        require(
            table["length"][0].as_py() == lengths[episode_index],
            f"bad metadata length in {path}",
        )

    return lengths, task_episode_counts, global_index


def validate_videos(
    args: argparse.Namespace,
    root: Path,
    camera_keys: list[str],
    lengths: list[int],
) -> None:
    import av

    for camera_key in camera_keys:
        files = indexed_files(root / "videos" / camera_key, ".mp4")
        require(len(files) == args.expected_episodes, f"{camera_key}: found {len(files)} videos")
        for episode_index, path in enumerate(files):
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                frames = stream.frames or sum(1 for _ in container.decode(video=0))
                rate = float(stream.average_rate)
                require(frames == lengths[episode_index], f"frame count mismatch: {path}")
                require(
                    (stream.width, stream.height) == (224, 224),
                    f"bad video dimensions: {path}",
                )
                require(abs(rate - args.expected_fps) < 1e-6, f"bad video frame rate: {path}")


def validate_lerobot_loading(
    args: argparse.Namespace,
    root: Path,
    camera_keys: list[str],
    total_frames: int,
    expected_tasks: int,
) -> None:
    # PyAV is selected explicitly below, so a broken optional TorchCodec install is irrelevant.
    logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    delta_timestamps = {
        "action": [index / args.expected_fps for index in range(args.action_horizon)]
    }
    dataset = LeRobotDataset(
        args.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )
    require(
        len(dataset) == total_frames,
        f"LeRobot loaded {len(dataset)} frames, expected {total_frames}",
    )
    require(dataset.num_episodes == args.expected_episodes, "LeRobot episode count mismatch")
    require(len(dataset.meta.tasks) == expected_tasks, "task table mismatch")

    sample_indices = sorted({0, total_frames // 2, total_frames - 1})
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        require(
            tuple(sample["observation.state"].shape) == (args.expected_state_dim,),
            "bad state sample",
        )
        require(
            tuple(sample["action"].shape) == (args.action_horizon, args.expected_action_dim),
            "bad action horizon sample",
        )
        require(isinstance(sample.get("task"), str) and sample["task"], "missing task text")
        for camera_key in camera_keys:
            image = sample[camera_key]
            require(tuple(image.shape) == (3, 224, 224), f"bad decoded image: {camera_key}")
            require(image.dtype == torch.uint8, f"decoded image is not uint8: {camera_key}")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    require(root.is_dir(), f"dataset root does not exist: {root}")

    print("Validating metadata and feature schema...", flush=True)
    info, camera_keys = validate_info(args, root)
    print("Validating parquet tables...", flush=True)
    lengths, task_episode_counts, total_frames = validate_tables(args, root)
    require(
        info.get("total_frames") == total_frames,
        "info.total_frames does not match parquet rows",
    )
    require(
        info.get("total_tasks") == len(task_episode_counts),
        "info.total_tasks does not match data",
    )
    print("Validating video containers...", flush=True)
    validate_videos(args, root, camera_keys, lengths)
    print("Validating LeRobot loading and decoded samples...", flush=True)
    validate_lerobot_loading(args, root, camera_keys, total_frames, len(task_episode_counts))

    print(f"PASS: {root}")
    print(
        f"episodes={len(lengths)} frames={total_frames} fps={args.expected_fps} "
        f"length_range={min(lengths)}..{max(lengths)}"
    )
    print(
        f"state_dim={args.expected_state_dim} action_dim={args.expected_action_dim} "
        f"cameras={len(camera_keys)}"
    )
    print(f"task_episode_counts={dict(sorted(task_episode_counts.items()))}")


if __name__ == "__main__":
    main()
