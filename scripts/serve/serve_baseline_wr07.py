#!/usr/bin/env python3
"""Serve baseline's Omega-conditioned real-robot WR07 checkpoint."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "policy" / "baseline"
for source in (ROOT, BASELINE / "src", BASELINE / "packages" / "openpi-client" / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from bench.policy.backends.baseline_wr07 import BaselineWR07Backend
from bench.policy.server import WebsocketPolicyServer

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "data/real_robot/wr07_lerobot_v3"
    )
    parser.add_argument("--omega-repo", type=Path, required=True)
    parser.add_argument("--omega-checkpoint", type=Path, required=True)
    parser.add_argument("--omega-device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=59219)
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--inference-timeout-s", type=float, default=120.0)
    return parser.parse_args()


def _revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    backend = BaselineWR07Backend.from_checkpoint(
        checkpoint_dir=args.checkpoint,
        dataset_root=args.dataset_root,
        omega_repo=args.omega_repo,
        omega_checkpoint=args.omega_checkpoint,
        omega_device=args.omega_device,
        code_revision={
            "membench": _revision(ROOT),
            "policy_repo": _revision(BASELINE),
            "vggt_omega": _revision(args.omega_repo),
        },
    )
    LOGGER.info("Loaded %s with checkpoint %s", backend.contract.server_id, args.checkpoint)
    WebsocketPolicyServer(
        backend,
        host=args.host,
        port=args.port,
        api_key=os.environ.get(args.api_key_env),
        inference_timeout_s=args.inference_timeout_s,
    ).serve_forever()


if __name__ == "__main__":
    main()
