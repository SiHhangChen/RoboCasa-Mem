#!/usr/bin/env python
"""Serve one read-only LeRobot ACT checkpoint over membench-policy/v1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = ROOT / "policy" / "lerobot" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if LEROBOT_SRC.is_dir() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from bench.policy.backends.lerobot import LeRobotACTBackend  # noqa: E402
from bench.policy.server import WebsocketPolicyServer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-uri")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--inference-timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = LeRobotACTBackend.from_checkpoint(
        args.checkpoint,
        device=args.device,
        checkpoint_uri=args.checkpoint_uri,
        code_revision={
            "membench": _git_revision(ROOT),
            "policy_repo": _git_revision(ROOT / "policy" / "lerobot"),
        },
    )
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
