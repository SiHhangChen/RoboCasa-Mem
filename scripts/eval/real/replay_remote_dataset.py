#!/usr/bin/env python3
"""Replay recorded WR07 observations through a remote policy diagnostically."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = ROOT / "policy" / "lerobot" / "src"
for source in (ROOT, LEROBOT_SRC):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from bench.evaluation.client import RemotePolicyClient
from bench.policy.protocol import CanonicalObservation, validate_observation
from bench.real_robot.contracts import RUNTIME_PROFILE

CAMERA_KEYS = {alias: f"observation.images.{alias}" for alias in ("front", "left", "right")}
DEFAULT_PROMPT = (
    "Remember the color of the container holding the movable item. Move the movable item "
    "from the source container to the matching-color container on the opposite side."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/real_robot/wr07_lerobot_v3")
    parser.add_argument("--repo-id", default="wr07-real-v3")
    parser.add_argument("--expected-config", default=None)
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--prompt-source",
        choices=("auto", "manual", "annotation"),
        default="auto",
        help="For WR07 subgoal checkpoints, auto asks for a color once per episode.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def evaluate_dataset(*, client: RemotePolicyClient, dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    if client.contract.runtime_profile != RUNTIME_PROFILE:
        raise ValueError(f"remote server must use {RUNTIME_PROFILE}, got {client.contract.runtime_profile}")
    if args.expected_config is not None and client.contract.policy.get("config") != args.expected_config:
        raise ValueError(
            f"remote config must be {args.expected_config}, got {client.contract.policy.get('config')!r}"
        )
    if client.contract.policy.get("family") == "openpi" and client.contract.policy.get("prompt_mode") not in (
        "fixed", "episode_choice"
    ):
        raise ValueError("Real-robot OpenPI server did not declare a WR07 prompt mode")
    records: list[dict[str, Any]] = []
    active_episode: str | None = None
    episode_prompts: dict[int, str] = {}
    baseline_omega = client.contract.policy.get("family") == "baseline_wr07_omega"
    for index in _sample_indices(dataset, args, baseline_omega=baseline_omega):
        if len(records) >= args.max_frames:
            break
        sample = dataset[index]
        episode_index = int(_numpy(sample["episode_index"]).reshape(-1)[0])
        frame_index = int(_numpy(sample["frame_index"]).reshape(-1)[0])
        episode_id = f"wr07-real-seed-{episode_index:06d}"
        if episode_id != active_episode:
            episode_prompts[episode_index] = _select_episode_prompt(
                client.contract.policy, args, episode_index
            )
            client.reset(episode_id)
            active_episode = episode_id
        prompt = episode_prompts[episode_index]
        raw_images = (
            {alias: _raw_image_hwc_uint8(sample[key]) for alias, key in CAMERA_KEYS.items()}
            if baseline_omega else None
        )
        observation = CanonicalObservation(
            task_id="WR-07",
            state=np.ascontiguousarray(_numpy(sample["observation.state"]).astype(np.float32)),
            images={
                alias: _image_hwc_uint8(
                    raw_images[alias] if raw_images is not None else sample[key],
                    target_size=client.contract.observation.image_size,
                    preserve_aspect=client.contract.policy.get("family") in ("openpi", "baseline_wr07_omega"),
                )
                for alias, key in CAMERA_KEYS.items()
            },
            prompt=prompt,
            extra_modalities={"raw_images": raw_images} if raw_images is not None else {},
        )
        validate_observation(observation, client.contract)
        response = client.infer(observation, episode_id=episode_id, step=frame_index)
        prediction = response.actions[0]
        ground_truth = _numpy(sample["action"]).astype(np.float32)
        absolute_error = np.abs(prediction - ground_truth)
        records.append({
            "dataset_index": index,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "prompt": prompt,
            "prediction": prediction.tolist(),
            "ground_truth": ground_truth.tolist(),
            "mae": float(absolute_error.mean()),
            "per_dimension_absolute_error": absolute_error.tolist(),
            "server_timing": dict(response.timing),
        })
    if not records:
        raise RuntimeError("No dataset frames were evaluated")
    return {
        "diagnostic_only": True,
        "hardware_execution": "disabled",
        "runtime_profile": RUNTIME_PROFILE,
        "frames": len(records),
        "episode_prompts": {str(key): value for key, value in episode_prompts.items()},
        "action_mae": float(np.mean([record["mae"] for record in records])),
        "policy_contract": client.contract.to_hello(),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if args.stride <= 0 or args.max_frames <= 0:
        raise ValueError("stride and max-frames must be positive")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("LeRobot dataset dependencies are required") from exc
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root.expanduser().resolve(),
        return_uint8=True,
        video_backend=args.video_backend,
    )
    client = RemotePolicyClient(args.endpoint, api_key=os.environ.get(args.api_key_env))
    try:
        client.connect()
        result = evaluate_dataset(client=client, dataset=dataset, args=args)
    finally:
        client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output={args.output.resolve()}")
    print(f"action_mae={result['action_mae']:.6f}")


def _episode_prompt(annotation_root: Path | None, episode_index: int) -> str | None:
    if annotation_root is None:
        return None
    path = annotation_root / "teacher_raw" / f"episode_{episode_index:06d}" / "annotation.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing WR07 annotation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages")
    if isinstance(stages, list) and stages:
        prompt = stages[0].get("active_subgoal")
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    prompt = payload.get("overall_goal")
    return prompt if isinstance(prompt, str) and prompt.strip() else None


def _select_episode_prompt(policy: Any, args: argparse.Namespace, episode_index: int) -> str:
    mode = policy.get("prompt_mode")
    if mode == "fixed":
        prompt = policy.get("default_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Server did not provide its fixed WR07 training prompt")
        if args.prompt is not None and args.prompt != prompt:
            raise ValueError("--prompt differs from the server's fixed training prompt")
        return prompt
    if mode == "episode_choice":
        choices = policy.get("prompt_choices")
        if not isinstance(choices, dict) or set(choices) != {"brown", "white", "pink"}:
            raise ValueError("Server did not provide the three WR07 subgoal prompts")
        if args.prompt is not None:
            raise ValueError(
                "Use --prompt-source manual or annotation for a WR07 subgoal checkpoint"
            )
        if args.prompt_source == "annotation":
            prompt = _episode_prompt(args.annotation_root, episode_index)
            if prompt not in choices.values():
                raise ValueError(f"Episode {episode_index} annotation has no known WR07 subgoal prompt")
            return prompt
        if args.prompt_source not in ("auto", "manual"):
            raise ValueError(f"Unsupported prompt source: {args.prompt_source}")
        if not sys.stdin.isatty():
            raise ValueError(
                "WR07 subgoal selection needs a terminal; use --prompt-source annotation for replay"
            )
        print(f"Episode {episode_index}: choose the source container color", flush=True)
        colors = tuple(choices)
        for number, color in enumerate(colors, start=1):
            print(f"  {number}. {color}: {choices[color]}", flush=True)
        while True:
            answer = input("Color [1/2/3 or name]: ").strip().lower()
            color = colors[int(answer) - 1] if answer in ("1", "2", "3") else answer
            if color in choices:
                return choices[color]
            print("Choose 1, 2, 3, brown, white, or pink.", flush=True)
    if args.prompt_source == "manual":
        raise ValueError("Manual WR07 color selection requires a subgoal OpenPI checkpoint")
    if args.prompt_source == "annotation" and args.annotation_root is None:
        raise ValueError("--prompt-source annotation requires --annotation-root")
    return args.prompt or _episode_prompt(args.annotation_root, episode_index) or DEFAULT_PROMPT


def _sample_indices(dataset: Any, args: argparse.Namespace, *, baseline_omega: bool):
    if not baseline_omega or not hasattr(args, "dataset_root"):
        return range(0, len(dataset), args.stride)
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((args.dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No WR07 frame table under {args.dataset_root / 'data'}")
    tables = [pq.read_table(path, columns=["index", "frame_index"]) for path in files]
    table = pa.concat_tables(tables).sort_by("index")
    indices = table.column("index").to_numpy()
    frames = table.column("frame_index").to_numpy()
    if len(indices) != len(dataset) or not np.array_equal(indices, np.arange(len(dataset))):
        raise ValueError("WR07 replay frame indices do not match the dataset")
    return indices[frames % args.stride == 0]


def _raw_image_hwc_uint8(value: Any) -> np.ndarray:
    image = _numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected raw RGB image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected raw RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_hwc_uint8(
    value: Any, *, target_size: tuple[int, int], preserve_aspect: bool = False
) -> np.ndarray:
    image = _numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(image.max()) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    target_height, target_width = target_size
    if image.shape[:2] != (target_height, target_width):
        source_height, source_width = image.shape[:2]
        if preserve_aspect:
            import cv2

            ratio = max(source_width / target_width, source_height / target_height)
            resized_height, resized_width = int(source_height / ratio), int(source_width / ratio)
            image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            top = (target_height - resized_height) // 2
            left = (target_width - resized_width) // 2
            padded[top : top + resized_height, left : left + resized_width] = image
            image = padded
        else:
            rows = np.linspace(0, source_height - 1, target_height).round().astype(np.intp)
            columns = np.linspace(0, source_width - 1, target_width).round().astype(np.intp)
            image = image[rows[:, None], columns[None, :]]
    return np.ascontiguousarray(image)


if __name__ == "__main__":
    main()
