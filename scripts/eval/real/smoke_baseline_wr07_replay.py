#!/usr/bin/env python3
"""Exercise one recorded WR07 frame through a fake baseline WebSocket server."""

from __future__ import annotations

import argparse
import multiprocessing
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SOURCE = ROOT / "policy/lerobot/src"
for source in (ROOT, LEROBOT_SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from bench.evaluation.client import RemotePolicyClient
from bench.policy.backends.baseline_wr07 import BaselineWR07Backend
from bench.policy.protocol import CheckpointIdentity
from bench.policy.server import WebsocketPolicyServer
from scripts.eval.real.replay_remote_dataset import evaluate_dataset


class _FakePolicy:
    def infer(self, observation):
        if observation["omega_memory"].shape != (1, 3, 1, 17, 2048):
            raise ValueError("Wrong Omega input shape")
        return {"actions": np.zeros((50, 17), dtype=np.float32)}


class _FakeEncoder:
    def reset(self):
        pass

    def encode(self, images):
        if any(image.shape != (480, 640, 3) for image in images.values()):
            raise ValueError("Wrong native camera shape")
        return np.zeros((1, 3, 1, 17, 2048), dtype=np.float32), np.ones(1, dtype=np.bool_)


def _serve(port: int) -> None:
    backend = BaselineWR07Backend(
        _FakePolicy(),
        _FakeEncoder(),
        checkpoint=CheckpointIdentity("test://fake-baseline", "sha256:fake-baseline"),
        omega_fingerprint="sha256:fake-omega",
        stats_fingerprint="sha256:fake-stats",
    )
    WebsocketPolicyServer(backend, host="127.0.0.1", port=port).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "data/real_robot/wr07_lerobot_v3"
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=ROOT / "data/annotation_json_20260921/wr07_real_robot_v3",
    )
    args = parser.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        "wr07-real-v3",
        root=args.dataset_root,
        return_uint8=True,
        video_backend="pyav",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = multiprocessing.Process(target=_serve, args=(port,), daemon=True)
    process.start()
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Fake baseline server did not start")
                time.sleep(0.02)
        replay_args = argparse.Namespace(
            dataset_root=args.dataset_root,
            stride=10,
            max_frames=1,
            prompt_source="annotation",
            annotation_root=args.annotation_root,
            prompt=None,
            expected_config="pi05_real_robot_wr07_v3",
        )
        with RemotePolicyClient(f"ws://127.0.0.1:{port}", timeout_s=30) as client:
            result = evaluate_dataset(client=client, dataset=dataset, args=replay_args)
        record = result["records"][0]
        print(
            f"frames={result['frames']} episode={record['episode_index']} "
            f"frame={record['frame_index']} prompt={record['prompt']!r} "
            f"action_dim={len(record['prediction'])}"
        )
    finally:
        process.terminate()
        process.join(timeout=2)


if __name__ == "__main__":
    main()
