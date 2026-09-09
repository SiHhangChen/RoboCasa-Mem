#!/usr/bin/env python
"""Serve one read-only OpenPI checkpoint over membench-policy/v2."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
OPENPI_SOURCE = Path(
    os.environ.get("OPENPI_SOURCE", str(ROOT / "policy" / "openpi" / "src"))
)
for source in (
    ROOT,
    OPENPI_SOURCE,
    ROOT / "policy" / "openpi" / "packages" / "openpi-client" / "src",
):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from bench.policy.backends.openpi import OpenPIBackend  # noqa: E402
from bench.policy.server import WebsocketPolicyServer  # noqa: E402
from bench.runtime.data_contract import STATE_SCHEMA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-uri")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image-size", type=int, choices=(224, 256), required=True)
    parser.add_argument(
        "--state-schema",
        choices=(STATE_SCHEMA,),
        required=True,
    )
    parser.add_argument("--default-prompt")
    parser.add_argument("--pytorch-device")
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--inference-timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    # Surface INFO logs from the model loader and the policy server, matching the
    # original OpenPI serve script.
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    logging.info("Loading model from %s (config: %s)", args.checkpoint, args.config)
    backend = OpenPIBackend.from_checkpoint(
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        image_size=(args.image_size, args.image_size),
        state_schema=args.state_schema,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
        checkpoint_uri=args.checkpoint_uri,
        code_revision={"membench": _git_revision(ROOT), "policy_repo": _git_revision(ROOT / "policy" / "openpi")},
    )
    logging.info("Model loaded")
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
