#!/usr/bin/env python
"""Run a deterministic CPU policy server for integration testing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.policy.backends.fake import FakePolicyBackend  # noqa: E402
from bench.policy.server import WebsocketPolicyServer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY")
    parser.add_argument("--inference-timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = FakePolicyBackend(
        horizon=args.horizon,
        image_size=(args.image_size, args.image_size),
    )
    server = WebsocketPolicyServer(
        backend,
        host=args.host,
        port=args.port,
        api_key=os.environ.get(args.api_key_env),
        inference_timeout_s=args.inference_timeout_s,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
