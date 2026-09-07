#!/usr/bin/env python
"""Query a remote OpenPI policy with one recorded MemBench dataset frame."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import imageio.v3 as iio
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = ROOT / "lerobot_data" / "membench_hs_wx_01"
DEFAULT_OPENPI_CLIENT_SRC = ROOT / "policy" / "openpi" / "packages" / "openpi-client" / "src"
CAMERA_KEY_CANDIDATES = {
    "agentview_left": (
        "observation.images.agentview_left",
        "observation.images.robot0_agentview_left",
    ),
    "agentview_right": (
        "observation.images.agentview_right",
        "observation.images.robot0_agentview_right",
    ),
    "eye_in_hand": (
        "observation.images.eye_in_hand",
        "observation.images.robot0_eye_in_hand",
    ),
}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def wait_for_policy_server(host: str, port: int, timeout_sec: float) -> None:
    if timeout_sec <= 0:
        return
    parsed = urlparse(host)
    tcp_host = parsed.hostname if parsed.scheme.startswith("ws") else host
    tcp_port = parsed.port if parsed.scheme.startswith("ws") and parsed.port is not None else port
    deadline = time.monotonic() + timeout_sec
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((tcp_host, tcp_port), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"could not connect to {tcp_host}:{tcp_port} within {timeout_sec:.1f}s: {last_error}")


def load_prompt(dataset_root: Path, task_index: int) -> str:
    subtasks_path = dataset_root / "meta" / "subtasks.tsv"
    if subtasks_path.is_file():
        tasks: dict[int, str] = {}
        for line_number, raw_line in enumerate(
            subtasks_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                raw_index, label = raw_line.split("\t", 1)
                index = int(raw_index)
            except ValueError as exc:
                raise ValueError(
                    f"invalid subtask row in {subtasks_path} at line {line_number}"
                ) from exc
            if index != len(tasks) or not label or label != label.strip():
                raise ValueError(
                    f"invalid subtask row in {subtasks_path} at line {line_number}: {raw_line!r}"
                )
            tasks[index] = label
        if int(task_index) not in tasks:
            raise KeyError(f"subtask_index={task_index} not found in {subtasks_path}")
        return tasks[int(task_index)]

    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    matches = tasks[tasks["task_index"].astype(int) == int(task_index)]
    if matches.empty:
        raise KeyError(f"task_index={task_index} not found")
    prompt = matches.index[0] if tasks.index.name == "task" else matches.iloc[0].get("task")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"invalid prompt for task_index={task_index}: {prompt!r}")
    return prompt.strip()


def resolve_camera_key(info: dict[str, Any], episode_row: pd.Series, short_key: str) -> str:
    candidates = CAMERA_KEY_CANDIDATES[short_key]
    feature_keys = set(info.get("features", {}))
    episode_keys = set(str(key) for key in episode_row.index)
    for camera_key in candidates:
        if camera_key in feature_keys or f"videos/{camera_key}/chunk_index" in episode_keys:
            return camera_key
    raise KeyError(
        f"could not resolve video key for {short_key}; tried {list(candidates)}, "
        f"available video columns: {sorted(key for key in episode_keys if key.startswith('videos/'))}"
    )


def episode_value(episode_row: pd.Series, key: str, default: Any = None) -> Any:
    return episode_row[key] if key in episode_row.index else default


def load_dataset_frame(dataset_root: Path, episode_index: int, frame_offset: int, horizon: int) -> dict[str, Any]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"no episode metadata under {dataset_root / 'meta' / 'episodes'}")
    episodes = pd.concat(
        [pd.read_parquet(path) for path in episode_files],
        ignore_index=True,
    )
    episode = episodes[episodes["episode_index"].astype(int) == int(episode_index)]
    if episode.empty:
        raise KeyError(f"episode_index={episode_index} not found")
    episode_row = episode.iloc[0]

    data_file = dataset_root / "data" / f"chunk-{int(episode_row['data/chunk_index']):03d}" / (
        f"file-{int(episode_row['data/file_index']):03d}.parquet"
    )
    frame_data = pd.read_parquet(data_file)
    start = int(episode_row["dataset_from_index"])
    length = int(episode_row["length"])
    if frame_offset < 0 or frame_offset >= length:
        raise ValueError(f"frame_offset must be in [0, {length}), got {frame_offset}")
    global_index = start + int(frame_offset)
    row = frame_data[frame_data["index"].astype(int) == global_index].iloc[0]

    task_index = int(row["task_index"])
    subtasks_path = dataset_root / "meta" / "subtasks.tsv"
    if subtasks_path.is_file():
        sidecar_file = dataset_root / "subtask_indices" / (
            data_file.relative_to(dataset_root / "data")
        )
        if not sidecar_file.is_file():
            raise FileNotFoundError(sidecar_file)
        sidecar = pd.read_parquet(sidecar_file)
        sidecar_match = sidecar[sidecar["index"].astype(int) == global_index]
        if len(sidecar_match) != 1:
            raise ValueError(
                f"expected one sidecar row for index={global_index}, got {len(sidecar_match)}"
            )
        sidecar_row = sidecar_match.iloc[0]
        for column in ("episode_index", "frame_index", "index"):
            if int(sidecar_row[column]) != int(row[column]):
                raise ValueError(
                    f"sidecar {column} mismatch for index={global_index}: "
                    f"{sidecar_row[column]} != {row[column]}"
                )
        task_index = int(sidecar_row["subtask_index"])

    actions = []
    for offset in range(horizon):
        idx = min(global_index + offset, start + length - 1)
        actions.append(np.asarray(frame_data[frame_data["index"].astype(int) == idx].iloc[0]["action"], dtype=np.float32))
    gt_actions = np.stack(actions, axis=0)

    images = {}
    frame_numbers = {}
    for short_key in ("agentview_left", "agentview_right", "eye_in_hand"):
        camera_key = resolve_camera_key(info, episode_row, short_key)
        chunk_index = int(episode_row[f"videos/{camera_key}/chunk_index"])
        file_index = int(episode_row[f"videos/{camera_key}/file_index"])
        from_timestamp = float(episode_value(episode_row, f"videos/{camera_key}/from_timestamp", 0.0))
        frame_number = int(round((from_timestamp + float(row["timestamp"])) * fps))
        frame_numbers[short_key] = frame_number
        video_path = dataset_root / info["video_path"].format(
            video_key=camera_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        images[short_key] = np.asarray(iio.imread(video_path, index=frame_number), dtype=np.uint8)

    return {
        "state": np.asarray(row["observation.state"], dtype=np.float32),
        "gt_actions": gt_actions,
        "images": images,
        "task_index": task_index,
        "timestamp": float(row["timestamp"]),
        "global_index": int(global_index),
        "frame_number": int(frame_numbers["agentview_left"]),
        "frame_numbers": frame_numbers,
    }


def make_client(host: str, port: int, api_key: str | None, client_src: Path):
    client_src = client_src.expanduser().resolve()
    if str(client_src) not in sys.path:
        sys.path.insert(0, str(client_src))
    from openpi_client import websocket_client_policy  # noqa: PLC0415

    return websocket_client_policy.WebsocketClientPolicy(host=host, port=port, api_key=api_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--connect-timeout-sec", type=float, default=60.0)
    parser.add_argument("--openpi-client-src", type=Path, default=DEFAULT_OPENPI_CLIENT_SRC)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=13)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    frame = load_dataset_frame(dataset_root, args.episode_index, args.frame_offset, args.horizon)
    prompt = args.prompt or load_prompt(dataset_root, frame["task_index"])
    request = {
        "images": frame["images"],
        "state": frame["state"],
        "prompt": prompt,
    }

    wait_for_policy_server(args.host, args.port, args.connect_timeout_sec)
    client = make_client(args.host, args.port, args.api_key, args.openpi_client_src)
    response = client.infer(request)
    pred = np.asarray(response["actions"], dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[None, :]
    pred = pred[:, : args.action_dim]
    gt = frame["gt_actions"][: pred.shape[0], : args.action_dim].astype(np.float64)
    abs_err = np.abs(pred - gt)

    summary = {
        "server": {"host": args.host, "port": args.port, "metadata": client.get_server_metadata()},
        "dataset_root": str(dataset_root),
        "episode_index": args.episode_index,
        "frame_offset": args.frame_offset,
        "global_index": frame["global_index"],
        "timestamp": frame["timestamp"],
        "frame_number": frame["frame_number"],
        "prompt": prompt,
        "response_keys": sorted(str(key) for key in response),
        "pred_shape": list(pred.shape),
        "gt_shape": list(gt.shape),
        "pred_first": pred[0].tolist(),
        "gt_first": gt[0].tolist(),
        "pred_first_10": pred[:10].tolist(),
        "gt_first_10": gt[:10].tolist(),
        "mae_by_dim": abs_err.mean(axis=0).tolist() if abs_err.size else [],
        "mae": float(abs_err.mean()) if abs_err.size else None,
        "server_timing": response.get("server_timing"),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
