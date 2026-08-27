#!/usr/bin/env python
"""Cross-query a remote OpenPI policy with dataset and rollout image/state hybrids."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageFilter

import query_pi05_remote_dataset_frame as dataset_query


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = ROOT / "lerobot_data" / "membench_hs_wx_01"
DEFAULT_OPENPI_CLIENT_SRC = ROOT / "policy" / "openpi" / "packages" / "openpi-client" / "src"
DEFAULT_ROLLOUT_QUERY = (
    ROOT
    / "outputs"
    / "eval"
    / "pi05_membench_hs_wx_01_lora_55000"
    / "remote_flipfix_1200_replan5_novideo"
    / "query_rollout_state_steps_0_200_500_800_1200.json"
)


def load_rollout_record(path: Path, step: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    for record in payload["records"]:
        if int(record["applied_steps"]) == int(step):
            return record
    raise KeyError(f"step {step} not found in {path}")


def transform_image(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return image
    pil_image = Image.fromarray(image)
    if mode.startswith("jpeg"):
        quality = int(mode.removeprefix("jpeg"))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
    if mode.startswith("blur"):
        radius = float(mode.removeprefix("blur"))
        return np.asarray(pil_image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.uint8)
    if mode == "mp4":
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            imageio.mimsave(tmp.name, [image], fps=20, codec="libx264", quality=8)
            return np.asarray(imageio.mimread(tmp.name)[0], dtype=np.uint8)
    raise ValueError(f"unsupported image transform: {mode}")


def load_rollout_images(record: dict[str, Any], transform: str) -> dict[str, np.ndarray]:
    images = record.get("images", {})
    return {
        "agentview_left": transform_image(
            np.asarray(imageio.imread(images["agentview_left_policy"]), dtype=np.uint8),
            transform,
        ),
        "agentview_right": transform_image(
            np.asarray(imageio.imread(images["agentview_right_policy"]), dtype=np.uint8),
            transform,
        ),
        "eye_in_hand": transform_image(
            np.asarray(imageio.imread(images["eye_in_hand_policy"]), dtype=np.uint8),
            transform,
        ),
    }


def summarize(pred: np.ndarray, gt: np.ndarray | None) -> dict[str, Any]:
    result = {
        "pred_first": pred[0].astype(float).tolist(),
        "pred_first_tail_8_13": pred[0, 8:13].astype(float).tolist(),
        "pred_first_gripper": float(pred[0, 7]),
        "pred_first_10_tail_8_13": pred[:10, 8:13].astype(float).tolist(),
    }
    if gt is not None:
        n = min(len(pred), len(gt))
        err = np.abs(pred[:n] - gt[:n])
        result.update(
            {
                "gt_first": gt[0].astype(float).tolist(),
                "gt_first_tail_8_13": gt[0, 8:13].astype(float).tolist(),
                "mae": float(err.mean()),
                "mae_by_dim": err.mean(axis=0).astype(float).tolist(),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--connect-timeout-sec", type=float, default=60.0)
    parser.add_argument("--openpi-client-src", type=Path, default=DEFAULT_OPENPI_CLIENT_SRC)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-offset", type=int, default=133)
    parser.add_argument("--rollout-query", type=Path, default=DEFAULT_ROLLOUT_QUERY)
    parser.add_argument("--rollout-step", type=int, default=500)
    parser.add_argument(
        "--rollout-image-transform",
        choices=("none", "jpeg95", "jpeg85", "jpeg75", "jpeg50", "blur0.5", "blur1.0", "mp4"),
        default="none",
    )
    parser.add_argument("--save-transformed-images-dir", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=13)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    rollout_query = args.rollout_query.expanduser().resolve()
    if args.output is None:
        args.output = rollout_query.parent / f"hybrid_dataset_{args.frame_offset}_rollout_{args.rollout_step}.json"
    args.output = args.output.expanduser().resolve()

    dataset_frame = dataset_query.load_dataset_frame(
        dataset_root,
        args.episode_index,
        args.frame_offset,
        args.horizon,
    )
    rollout_record = load_rollout_record(rollout_query, args.rollout_step)
    rollout_images = load_rollout_images(rollout_record, args.rollout_image_transform)
    if args.save_transformed_images_dir is not None:
        image_dir = args.save_transformed_images_dir.expanduser().resolve()
        image_dir.mkdir(parents=True, exist_ok=True)
        for name, image in rollout_images.items():
            imageio.imwrite(image_dir / f"rollout_step_{args.rollout_step}_{name}_{args.rollout_image_transform}.png", image)
    rollout_state = np.asarray(rollout_record["state"]["state"], dtype=np.float32)
    dataset_state = np.asarray(dataset_frame["state"], dtype=np.float32)
    gt = np.asarray(dataset_frame["gt_actions"][:, : args.action_dim], dtype=np.float64)
    prompt = dataset_query.load_prompt(dataset_root, int(dataset_frame["task_index"]))

    dataset_query.wait_for_policy_server(args.host, args.port, args.connect_timeout_sec)
    client = dataset_query.make_client(args.host, args.port, args.api_key, args.openpi_client_src)

    combos = {
        "dataset_images_dataset_state": (dataset_frame["images"], dataset_state),
        "dataset_images_rollout_state": (dataset_frame["images"], rollout_state),
        "rollout_images_dataset_state": (rollout_images, dataset_state),
        "rollout_images_rollout_state": (rollout_images, rollout_state),
    }
    results = {}
    for name, (images, state) in combos.items():
        response = client.infer({"images": images, "state": state, "prompt": prompt})
        pred = np.asarray(response["actions"], dtype=np.float64)
        if pred.ndim == 1:
            pred = pred[None, :]
        pred = pred[:, : args.action_dim]
        results[name] = summarize(pred[: args.horizon], gt)

    state_delta = rollout_state.astype(np.float64) - dataset_state.astype(np.float64)
    summary = {
        "server": {"host": args.host, "port": args.port, "metadata": client.get_server_metadata()},
        "dataset_root": str(dataset_root),
        "episode_index": args.episode_index,
        "frame_offset": args.frame_offset,
        "rollout_query": str(rollout_query),
        "rollout_step": args.rollout_step,
        "rollout_image_transform": args.rollout_image_transform,
        "prompt": prompt,
        "state_delta_l2": float(np.linalg.norm(state_delta)),
        "state_delta_base_xy_l2": float(np.linalg.norm(state_delta[18:20])),
        "state_delta_eef_l2": float(np.linalg.norm(state_delta[30:33])),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset_query.jsonable(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(dataset_query.jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
