#!/usr/bin/env python
"""Run remote policy inference on WX-01 dataset frames for diagnostics only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = ROOT / "policy" / "lerobot" / "src"
for source in (ROOT, LEROBOT_SRC):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from bench.evaluation.client import RemotePolicyClient  # noqa: E402
from bench.policy.protocol import CanonicalObservation, validate_observation  # noqa: E402
from bench.runtime.data_contract import STATE_SCHEMA  # noqa: E402


DEFAULT_PROMPT = (
    "Start with the sponge already grasped and the cup displaced. "
    "Wipe the remembered cup spot, put down the sponge, then return "
    "the cup to that original spot."
)
DATASET_CAMERA_FEATURES = {
    "agentview_right": "observation.images.robot0_agentview_right",
    "agentview_left": "observation.images.robot0_agentview_left",
    "eye_in_hand": "observation.images.robot0_eye_in_hand",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "lerobot_data" / "wx01")
    parser.add_argument("--repo-id", default="wx01")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def iter_dataset_requests(
    dataset: Any,
    *,
    contract: Any,
    stride: int,
    max_frames: int,
    prompt: str,
):
    if stride <= 0 or max_frames <= 0:
        raise ValueError("stride and max_frames must be positive")
    for index in range(0, len(dataset), stride):
        if index // stride >= max_frames:
            break
        sample = dataset[index]
        state = _numpy(sample["observation.state"]).astype(np.float32)
        images = {
            alias: _image_hwc_uint8(sample[DATASET_CAMERA_FEATURES[alias]])
            for alias in contract.observation.required_cameras
        }
        observation = CanonicalObservation(
            task_id="WX-01",
            state=np.ascontiguousarray(state),
            images=images,
            prompt=prompt,
        )
        validate_observation(observation, contract)
        yield {
            "index": index,
            "episode_index": int(_numpy(sample["episode_index"]).reshape(-1)[0]),
            "frame_index": int(_numpy(sample["frame_index"]).reshape(-1)[0]),
            "observation": observation,
            "ground_truth_action": _numpy(sample["action"]).astype(np.float32),
        }


def evaluate_dataset(
    *,
    client: RemotePolicyClient,
    dataset: Any,
    stride: int,
    max_frames: int,
    prompt: str,
) -> dict[str, Any]:
    if client.contract.observation.state_schema != STATE_SCHEMA:
        raise ValueError(
            "Current lerobot_data/wx01 is episode-continuous canonical_xyzw; "
            "the remote policy contract "
            f"declares {client.contract.observation.state_schema}"
        )
    records = []
    active_episode = None
    for item in iter_dataset_requests(
        dataset,
        contract=client.contract,
        stride=stride,
        max_frames=max_frames,
        prompt=prompt,
    ):
        episode_id = f"dataset-episode-{item['episode_index']:06d}"
        if episode_id != active_episode:
            client.reset(episode_id)
            active_episode = episode_id
        response = client.infer(
            item["observation"],
            episode_id=episode_id,
            step=item["frame_index"],
        )
        prediction = response.actions[0]
        ground_truth = item["ground_truth_action"]
        error = np.abs(prediction - ground_truth)
        records.append(
            {
                "dataset_index": item["index"],
                "episode_index": item["episode_index"],
                "frame_index": item["frame_index"],
                "prediction": prediction.tolist(),
                "ground_truth": ground_truth.tolist(),
                "mae": float(error.mean()),
                "per_dimension_absolute_error": error.tolist(),
                "server_timing": dict(response.timing),
            }
        )
    if not records:
        raise RuntimeError("No dataset frames were evaluated")
    return {
        "diagnostic_only": True,
        "frames": len(records),
        "action_mae": float(np.mean([record["mae"] for record in records])),
        "policy_contract": client.contract.to_hello(),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot dataset dependencies are required for eval_remote_dataset.py"
        ) from exc
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root.expanduser().resolve(),
        return_uint8=True,
        video_backend=args.video_backend,
    )
    client = RemotePolicyClient(
        args.endpoint,
        api_key=os.environ.get(args.api_key_env),
    )
    try:
        client.connect()
        result = evaluate_dataset(
            client=client,
            dataset=dataset,
            stride=args.stride,
            max_frames=args.max_frames,
            prompt=args.prompt,
        )
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output.resolve()}")
    print(f"action_mae={result['action_mae']:.6f}")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_hwc_uint8(value: Any) -> np.ndarray:
    image = _numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


if __name__ == "__main__":
    main()
