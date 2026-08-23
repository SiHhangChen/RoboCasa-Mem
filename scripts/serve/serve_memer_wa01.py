#!/usr/bin/env python
"""Serve the WA-01 MemER + OpenPI composite policy."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for source in (
    ROOT,
    ROOT / "policy" / "openpi" / "src",
    ROOT / "policy" / "openpi" / "packages" / "openpi-client" / "src",
    ROOT / "policy" / "memer",
):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from bench.policy.backends.memer_openpi import (  # noqa: E402
    MemEROpenPIBackend,
    load_allowed_subtasks,
)
from bench.policy.backends.openpi import OpenPIBackend  # noqa: E402
from bench.policy.server import WebsocketPolicyServer  # noqa: E402
from bench.runtime.data_contract import STATE_SCHEMA  # noqa: E402
from memer_eval import DeploymentConfig, MemERDeploymentPolicy  # noqa: E402


DEFAULT_INSTRUCTION = (
    "Start at the coffee machine with two identical prepared coffees. Add one "
    "sugar cube to either coffee, carry both coffees to the dining counter, and "
    "identify the coffee you sugared."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-checkpoint", type=Path, required=True)
    parser.add_argument("--openpi-checkpoint", type=Path, required=True)
    parser.add_argument("--openpi-config", default="memer_wa01")
    parser.add_argument(
        "--openpi-norm-asset-id",
        default="wa01_200seeds_v061_subtasks_v2",
    )
    parser.add_argument(
        "--subtask-config",
        type=Path,
        default=ROOT / "configs/memer/wa01_subtasks.json",
    )
    parser.add_argument(
        "--camera-layout-config",
        type=Path,
        default=ROOT / "configs/memer/wa01_rgb_camera_layout.json",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    # The rollout client sends one observation every 10 physical steps.  From
    # MemER's perspective that is already the stride-10 sampled stream, so its
    # internal frame subsample must be one.
    parser.add_argument("--frame-subsample", type=int, default=1)
    parser.add_argument("--recent-frames-length", type=int, default=8)
    parser.add_argument("--memory-length", type=int, default=8)
    parser.add_argument("--merge-distance", type=int, default=5)
    parser.add_argument("--qwen-processor")
    parser.add_argument("--qwen-device", default=None)
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--qwen-attn-implementation", default="sdpa")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=128)
    parser.add_argument("--openpi-pytorch-device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image-size", type=int, choices=(224, 256), default=224)
    parser.add_argument("--state-schema", choices=(STATE_SCHEMA,), default=STATE_SCHEMA)
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--inference-timeout-s", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    _require_qwen_transformers()
    for path in (args.qwen_checkpoint, args.openpi_checkpoint, args.subtask_config):
        if not path.exists():
            raise FileNotFoundError(path)

    logging.info("Loading MemER high-level checkpoint from %s", args.qwen_checkpoint)
    memer_policy = MemERDeploymentPolicy.from_qwen_checkpoint(
        model_path=str(args.qwen_checkpoint),
        processor_path=args.qwen_processor,
        instruction=args.instruction,
        device=args.qwen_device,
        dtype=args.qwen_dtype,
        attn_implementation=args.qwen_attn_implementation,
        max_new_tokens=args.qwen_max_new_tokens,
        config=DeploymentConfig(
            frame_subsample=args.frame_subsample,
            recent_frames_length=args.recent_frames_length,
            memory_length=args.memory_length,
            merge_distance=args.merge_distance,
            camera_layout_config=str(args.camera_layout_config),
        ),
    )
    logging.info("Loading OpenPI low-level checkpoint from %s", args.openpi_checkpoint)
    low_level = OpenPIBackend.from_checkpoint(
        config_name=args.openpi_config,
        checkpoint_dir=args.openpi_checkpoint,
        image_size=(args.image_size, args.image_size),
        state_schema=args.state_schema,
        pytorch_device=args.openpi_pytorch_device,
        norm_asset_id=args.openpi_norm_asset_id,
    )
    backend = MemEROpenPIBackend(
        memer_policy,
        low_level,
        instruction=args.instruction,
        allowed_subtasks=load_allowed_subtasks(args.subtask_config),
    )
    logging.info(
        "Composite policy ready: action_horizon=%d, frame_subsample=%d",
        backend.contract.action.horizon,
        args.frame_subsample,
    )
    WebsocketPolicyServer(
        backend,
        host=args.host,
        port=args.port,
        api_key=os.environ.get(args.api_key_env),
        inference_timeout_s=args.inference_timeout_s,
    ).serve_forever()


def _require_qwen_transformers() -> None:
    """Fail early with the actionable dependency error from Qwen3-VL."""

    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "MemER serving requires transformers>=4.57.0 in the serving Python environment."
        ) from exc
    parts = tuple(int(part) for part in version.split(".")[:2])
    if parts < (4, 57):
        raise RuntimeError(
            f"MemER Qwen3-VL checkpoint requires transformers>=4.57.0; found {version}."
        )


if __name__ == "__main__":
    main()
