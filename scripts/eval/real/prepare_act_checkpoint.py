#!/usr/bin/env python3
"""Attach the frozen WR07 real-robot contract to a LeRobot checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.policy.backends.lerobot import resolve_pretrained_model_dir
from bench.real_robot.contracts import (
    ACTION_FIELDS,
    ACTION_SCHEMA,
    CAMERA_ALIASES,
    CONTROL_FREQUENCY_HZ,
    IMAGE_SIZE,
    RUNTIME_PROFILE,
    STATE_FIELDS,
    STATE_SCHEMA,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/real_robot/wr07_lerobot_v3"),
    )
    parser.add_argument("--output-name", default="membench_policy.json")
    return parser.parse_args()


def build_metadata(checkpoint: Path, dataset_root: Path) -> dict[str, Any]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info.get("features")
    if not isinstance(features, dict):
        raise TypeError("dataset meta/info.json has no features map")
    _validate_feature(
        features,
        "observation.state",
        dtype="float64",
        shape=[17],
        names=[list(STATE_FIELDS)],
    )
    _validate_feature(
        features,
        "action",
        dtype="float64",
        shape=[17],
        names=[list(ACTION_FIELDS)],
    )
    for camera in CAMERA_ALIASES:
        _validate_feature(
            features,
            f"observation.images.{camera}",
            dtype="video",
            shape=[3, 480, 640],
        )
    if float(info.get("fps", 0)) != CONTROL_FREQUENCY_HZ:
        raise ValueError(f"dataset fps must be {CONTROL_FREQUENCY_HZ:g}")

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "act":
        raise ValueError(f"checkpoint must be ACT, got {config.get('type')!r}")
    input_state = config.get("input_features", {}).get("observation.state", {})
    output_action = config.get("output_features", {}).get("action", {})
    if input_state.get("shape") != [17] or output_action.get("shape") != [17]:
        raise ValueError("checkpoint must have observation.state/action shape [17]")
    for camera in CAMERA_ALIASES:
        feature = config.get("input_features", {}).get(f"observation.images.{camera}")
        if not isinstance(feature, dict) or feature.get("shape") != [3, 224, 224]:
            raise ValueError(f"checkpoint camera {camera!r} must have shape [3,224,224]")
    if int(config.get("chunk_size", 0)) <= 0:
        raise ValueError("checkpoint chunk_size must be positive")

    return {
        "runtime_profile": RUNTIME_PROFILE,
        "state_schema": STATE_SCHEMA,
        "state_dim": 17,
        "state_fields": list(STATE_FIELDS),
        "action_schema": ACTION_SCHEMA,
        "action_dim": 17,
        "action_fields": list(ACTION_FIELDS),
        "camera_aliases": list(CAMERA_ALIASES),
        "camera_mapping": {
            alias: f"observation.images.{alias}" for alias in CAMERA_ALIASES
        },
        "image_size": list(IMAGE_SIZE),
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "hardware_execution_semantics": "unverified",
    }


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_feature(
    features: dict[str, Any],
    name: str,
    *,
    dtype: str,
    shape: list[int],
    names: list[list[str]] | None = None,
) -> None:
    feature = features.get(name)
    if not isinstance(feature, dict):
        raise TypeError(f"dataset feature missing or invalid: {name}")
    if feature.get("dtype") != dtype or feature.get("shape") != shape:
        raise ValueError(
            f"dataset feature {name} must be dtype={dtype!r}, shape={shape}; "
            f"got dtype={feature.get('dtype')!r}, shape={feature.get('shape')!r}"
        )
    if names is not None and feature.get("names") != names:
        raise ValueError(f"dataset feature {name} names do not match WR07 contract")


def main() -> None:
    args = parse_args()
    checkpoint = resolve_pretrained_model_dir(args.checkpoint)
    dataset_root = args.dataset_root.expanduser().resolve()
    payload = build_metadata(checkpoint, dataset_root)
    destination = checkpoint / args.output_name
    write_metadata(destination, payload)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
