#!/usr/bin/env python
"""Validate WX-01 data and invoke the upstream LeRobot ACT trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs" / "policies" / "act_wx01_v1.yaml"

from bench.runtime.data_contract import (  # noqa: E402
    ACTION_DIM,
    ACTION_SCHEMA,
    POLICY_PROTOCOL_VERSION,
    STATE_DIM,
    STATE_SCHEMA,
    action_schema_metadata,
    state_schema_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def validate_dataset_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _mapping(config, "dataset")
    root = _resolve_path(_string(dataset, "root"))
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset info not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = _mapping(info, "features")
    state_feature = _string(dataset, "state_feature")
    action_feature = _string(dataset, "action_feature")
    _require_feature(features, state_feature, shape=(STATE_DIM,), dtype="float32")
    raw_state_info = features[state_feature].get("info", {})
    if not isinstance(raw_state_info, Mapping) or dict(raw_state_info) != state_schema_metadata():
        raise ValueError(
            "LeRobot state feature metadata does not match the robot_state37 contract"
        )
    _require_feature(features, action_feature, shape=(ACTION_DIM,), dtype="float32")
    raw_action_info = features[action_feature].get("info", {})
    if not isinstance(raw_action_info, Mapping) or dict(raw_action_info) != action_schema_metadata():
        raise ValueError(
            "LeRobot action feature metadata does not match the v2 action contract"
        )
    expected_image_size = tuple(int(item) for item in dataset["image_size"])
    camera_features = _mapping(dataset, "camera_features")
    expected_camera_aliases = (
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    )
    if tuple(camera_features) != expected_camera_aliases:
        raise ValueError(
            "ACT WX-01 must explicitly select agentview_left, agentview_right, "
            "and eye_in_hand cameras"
        )
    for alias, feature_name in camera_features.items():
        feature = _mapping(features, str(feature_name))
        shape = tuple(int(item) for item in feature.get("shape", ()))
        expected_shape = (*expected_image_size, 3)
        if shape != expected_shape:
            raise ValueError(
                f"Camera {alias} must have shape {expected_shape}, got {shape}"
            )
    quaternion_audit = audit_quaternion_sign_policy(root)
    declared_schema = _string(dataset, "state_schema")
    if quaternion_audit["state_schema"] != declared_schema:
        raise ValueError(
            f"Dataset quaternion audit resolved {quaternion_audit['state_schema']}, "
            f"config declares {declared_schema}"
        )
    return {
        "root": str(root),
        "info": info,
        "dataset_hash": hash_dataset_metadata(root),
        "quaternion_audit": quaternion_audit,
    }


def audit_quaternion_sign_policy(dataset_root: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to audit the WX-01 dataset") from exc
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No dataset parquet files under {dataset_root / 'data'}")
    negative_initial_dominant = [0, 0]
    sign_discontinuities = [0, 0]
    invalid_quaternions = [0, 0]
    references: dict[tuple[int, int], np.ndarray] = {}
    frames = 0
    for path in files:
        table = parquet.read_table(
            path,
            columns=["observation.state", "episode_index"],
        )
        states = np.asarray(
            table.column("observation.state").to_pylist(),
            dtype=np.float32,
        )
        episode_indices = np.asarray(
            table.column("episode_index").to_pylist(),
            dtype=np.int64,
        ).reshape(-1)
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"Expected state matrix [N,{STATE_DIM}] in {path}, got {states.shape}")
        if episode_indices.shape != (len(states),):
            raise ValueError(
                f"Expected one episode_index per state in {path}, "
                f"got {episode_indices.shape}"
            )
        for row, episode_index in zip(states, episode_indices, strict=True):
            for field_index, (start, stop) in enumerate(((21, 25), (33, 37))):
                quaternion = np.asarray(row[start:stop], dtype=np.float64)
                norm = float(np.linalg.norm(quaternion))
                if not np.all(np.isfinite(quaternion)) or norm < 1e-12:
                    invalid_quaternions[field_index] += 1
                    continue
                quaternion /= norm
                reference_key = (int(episode_index), field_index)
                reference = references.get(reference_key)
                if reference is None:
                    dominant_index = int(np.argmax(np.abs(quaternion)))
                    if quaternion[dominant_index] < 0.0:
                        negative_initial_dominant[field_index] += 1
                elif float(np.dot(quaternion, reference)) < -1e-6:
                    sign_discontinuities[field_index] += 1
                references[reference_key] = quaternion
        frames += len(states)
    if negative_initial_dominant != [0, 0] or sign_discontinuities != [0, 0] or invalid_quaternions != [0, 0]:
        raise ValueError("dataset violates the v2 episode-continuous quaternion contract")
    state_schema = STATE_SCHEMA
    return {
        "frames": frames,
        "negative_initial_dominant_base_quaternion": negative_initial_dominant[0],
        "negative_initial_dominant_eef_quaternion": (
            negative_initial_dominant[1]
        ),
        "base_quaternion_sign_discontinuities": sign_discontinuities[0],
        "eef_quaternion_sign_discontinuities": sign_discontinuities[1],
        "invalid_base_quaternions": invalid_quaternions[0],
        "invalid_eef_quaternions": invalid_quaternions[1],
        "sign_policy": "dominant_positive_first_then_episode_continuous",
        "state_schema": state_schema,
    }


def build_training_config(
    config: Mapping[str, Any],
    *,
    dataset_contract: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = _mapping(config, "dataset")
    policy = dict(_mapping(config, "policy"))
    training = _mapping(config, "training")
    height, width = (int(item) for item in dataset["image_size"])
    camera_features = _mapping(dataset, "camera_features")
    policy["input_features"] = {
        str(feature_name): {"type": "VISUAL", "shape": [3, height, width]}
        for feature_name in camera_features.values()
    }
    policy["input_features"][_string(dataset, "state_feature")] = {
        "type": "STATE",
        "shape": [STATE_DIM],
    }
    policy["output_features"] = {
        _string(dataset, "action_feature"): {
            "type": "ACTION",
            "shape": [ACTION_DIM],
        }
    }
    return {
        "dataset": {
            "repo_id": _string(dataset, "repo_id"),
            "root": dataset_contract["root"],
        },
        "policy": policy,
        "output_dir": str(_resolve_path(_string(training, "output_dir"))),
        "job_name": _string(training, "job_name"),
        "seed": int(training["seed"]),
        "steps": int(training["steps"]),
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["num_workers"]),
        "log_freq": int(training["log_freq"]),
        "save_freq": int(training["save_freq"]),
        "save_checkpoint": bool(training["save_checkpoint"]),
        "wandb": {"enable": bool(training.get("wandb_enable", False))},
    }


def write_checkpoint_contract(
    output_dir: Path,
    *,
    source_config: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
) -> list[Path]:
    dataset = _mapping(source_config, "dataset")
    metadata = {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "state_schema": _string(dataset, "state_schema"),
        "state_dim": STATE_DIM,
        "action_schema": ACTION_SCHEMA,
        "action_dim": ACTION_DIM,
        "action_metadata": action_schema_metadata(),
        "camera_mapping": dict(_mapping(dataset, "camera_features")),
        "image_size": list(dataset["image_size"]),
        "dataset_root": dataset_contract["root"],
        "dataset_hash": dataset_contract["dataset_hash"],
        "quaternion_audit": dict(dataset_contract["quaternion_audit"]),
    }
    written = []
    for checkpoint in sorted(output_dir.glob("checkpoints/*/pretrained_model")):
        if not (checkpoint / "config.json").is_file():
            continue
        path = checkpoint / "membench_policy.json"
        _atomic_json(path, metadata)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    source_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(source_config, dict) or int(source_config.get("schema_version", 0)) != 1:
        raise ValueError("ACT WX-01 training config requires schema_version=1")
    dataset_contract = validate_dataset_contract(source_config)
    training_config = build_training_config(
        source_config,
        dataset_contract=dataset_contract,
    )
    output_dir = Path(training_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = output_dir / "resolved_lerobot_train.json"
    resolved_config.write_text(
        json.dumps(training_config, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={resolved_config}",
    ]
    print(f"dataset_contract={dataset_contract}")
    print(f"resolved_config={resolved_config}")
    print(f"command={' '.join(command)}")
    if args.dry_run:
        return
    env = dict(os.environ)
    lerobot_source = ROOT / "policy" / "lerobot" / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(lerobot_source), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    written = write_checkpoint_contract(
        output_dir,
        source_config=source_config,
        dataset_contract=dataset_contract,
    )
    if not written:
        raise RuntimeError(f"Training completed but no checkpoints found under {output_dir}")
    print(f"checkpoint_metadata_written={len(written)}")


def hash_dataset_metadata(dataset_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((dataset_root / "meta").glob("*")):
        if not path.is_file():
            continue
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _require_feature(
    features: Mapping[str, Any],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    feature = _mapping(features, name)
    actual_shape = tuple(int(item) for item in feature.get("shape", ()))
    actual_dtype = str(feature.get("dtype"))
    if actual_shape != shape or actual_dtype != dtype:
        raise ValueError(
            f"Feature {name} must be {dtype}{list(shape)}, "
            f"got {actual_dtype}{list(actual_shape)}"
        )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a map")
    return value


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
