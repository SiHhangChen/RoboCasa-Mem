#!/usr/bin/env python
"""Compare remote OpenPI actions against recorded WX01 LeRobot actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import query_pi05_remote_dataset_frame as dataset_query
from bench.runtime.data_contract import ACTION_AXES


ACTION_LABELS = tuple(axis.name for axis in ACTION_AXES)


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(item))
    if not result:
        raise ValueError("at least one offset is required")
    return result


def jsonable(value: Any) -> Any:
    return dataset_query.jsonable(value)


def action_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def per_dim_dict(values: np.ndarray) -> dict[str, float]:
    return {
        label: float(values[idx])
        for idx, label in enumerate(ACTION_LABELS[: values.shape[0]])
    }


def compare_frame(
    *,
    client: Any,
    dataset_root: Path,
    episode_index: int,
    frame_offset: int,
    horizon: int,
    action_dim: int,
    prompt: str | None,
) -> dict[str, Any]:
    frame = dataset_query.load_dataset_frame(dataset_root, episode_index, frame_offset, horizon)
    resolved_prompt = prompt or dataset_query.load_prompt(dataset_root, frame["task_index"])
    response = client.infer(
        {
            "images": frame["images"],
            "state": frame["state"],
            "prompt": resolved_prompt,
        }
    )
    pred = np.asarray(response["actions"], dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[None, :]
    pred = pred[:, :action_dim]
    gt = frame["gt_actions"][: pred.shape[0], :action_dim].astype(np.float64)
    abs_err = np.abs(pred - gt)
    delta_first = pred[0] - gt[0]
    return {
        "episode_index": int(episode_index),
        "frame_offset": int(frame_offset),
        "global_index": int(frame["global_index"]),
        "timestamp": float(frame["timestamp"]),
        "frame_numbers": frame.get("frame_numbers"),
        "prompt": resolved_prompt,
        "pred_shape": list(pred.shape),
        "gt_shape": list(gt.shape),
        "mae": float(abs_err.mean()),
        "mae_by_dim": per_dim_dict(abs_err.mean(axis=0)),
        "max_abs_by_dim": per_dim_dict(abs_err.max(axis=0)),
        "pred_first": per_dim_dict(pred[0]),
        "gt_first": per_dim_dict(gt[0]),
        "delta_first": per_dim_dict(delta_first),
        "server_timing": response.get("server_timing"),
        "_pred": pred,
        "_gt": gt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--connect-timeout-sec", type=float, default=60.0)
    parser.add_argument("--openpi-client-src", type=Path, default=dataset_query.DEFAULT_OPENPI_CLIENT_SRC)
    parser.add_argument("--dataset-root", type=Path, default=dataset_query.DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-offsets", default="0,25,50,100,200,400,600")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=13)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/wx01_remote_action_compare.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    offsets = parse_int_list(args.frame_offsets)
    if args.action_dim > len(ACTION_LABELS):
        raise ValueError(f"action_dim={args.action_dim} exceeds known label count {len(ACTION_LABELS)}")

    dataset_query.wait_for_policy_server(args.host, args.port, args.connect_timeout_sec)
    client = dataset_query.make_client(args.host, args.port, args.api_key, args.openpi_client_src)

    frames = [
        compare_frame(
            client=client,
            dataset_root=dataset_root,
            episode_index=args.episode_index,
            frame_offset=offset,
            horizon=args.horizon,
            action_dim=args.action_dim,
            prompt=args.prompt,
        )
        for offset in offsets
    ]
    pred_all = np.concatenate([item.pop("_pred") for item in frames], axis=0)
    gt_all = np.concatenate([item.pop("_gt") for item in frames], axis=0)
    abs_err_all = np.abs(pred_all - gt_all)
    delta_all = pred_all - gt_all

    summary = {
        "server": {
            "host": args.host,
            "port": args.port,
            "metadata": client.get_server_metadata(),
        },
        "dataset_root": str(dataset_root),
        "episode_index": args.episode_index,
        "frame_offsets": offsets,
        "horizon": args.horizon,
        "action_dim": args.action_dim,
        "action_labels": list(ACTION_LABELS[: args.action_dim]),
        "aggregate": {
            "mae": float(abs_err_all.mean()),
            "mae_by_dim": per_dim_dict(abs_err_all.mean(axis=0)),
            "rmse_by_dim": per_dim_dict(np.sqrt((delta_all * delta_all).mean(axis=0))),
            "bias_by_dim": per_dim_dict(delta_all.mean(axis=0)),
            "pred": action_stats(pred_all),
            "gt": action_stats(gt_all),
        },
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
