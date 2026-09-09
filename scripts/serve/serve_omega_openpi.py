#!/usr/bin/env python
"""Serve an MME-VLA Omega checkpoint over membench-policy/v2."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
POLICY_ROOT = ROOT / "policy" / "robomme_policy_learning"
POLICY_SOURCE = Path(
    os.environ.get("MME_VLA_POLICY_SOURCE", str(POLICY_ROOT / "src"))
)
for source in (ROOT, POLICY_SOURCE):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from bench.policy.backends.omega_openpi import OmegaOpenPIBackend  # noqa: E402
from bench.policy.server import WebsocketPolicyServer  # noqa: E402
from bench.runtime.data_contract import STATE_SCHEMA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="mme_vla_omega_ts01_action_modulation",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-uri")
    parser.add_argument("--omega-repo", type=Path, required=True)
    parser.add_argument("--omega-checkpoint", type=Path, required=True)
    parser.add_argument("--omega-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image-size", type=int, choices=(224,), default=224)
    parser.add_argument("--state-schema", choices=(STATE_SCHEMA,), default=STATE_SCHEMA)
    parser.add_argument("--default-prompt")
    parser.add_argument("--norm-asset-id")
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--inference-timeout-s", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    logging.info(
        "Loading Omega policy from %s with scene encoder %s",
        args.checkpoint,
        args.omega_checkpoint,
    )
    backend = OmegaOpenPIBackend.from_checkpoint(
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        omega_repo=args.omega_repo,
        omega_checkpoint=args.omega_checkpoint,
        omega_device=args.omega_device,
        image_size=(args.image_size, args.image_size),
        state_schema=args.state_schema,
        seed=args.seed,
        default_prompt=args.default_prompt,
        norm_asset_id=args.norm_asset_id,
        checkpoint_uri=args.checkpoint_uri,
        code_revision={
            "membench": _git_revision(ROOT),
            "policy_repo": _git_revision(POLICY_ROOT),
            "vggt_omega": _git_revision(args.omega_repo),
        },
    )
    logging.info("Omega policy loaded")
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    WebsocketPolicyServer(
        backend,
        host=args.host,
        port=args.port,
        api_key=os.environ.get(args.api_key_env),
        inference_timeout_s=args.inference_timeout_s,
    ).serve_forever()


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


if __name__ == "__main__":
    main()
