from __future__ import annotations

import argparse
import multiprocessing
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from bench.evaluation.client import RemotePolicyClient
from bench.policy.backends.baseline_wr07 import BaselineWR07Backend, CurrentOmegaEncoder
from bench.policy.protocol import (
    CanonicalObservation,
    CheckpointIdentity,
    InferenceRequest,
    validate_contract,
)
from bench.policy.server import WebsocketPolicyServer
from bench.real_robot.wr07_prompts import SUBGOAL_PROMPTS
from scripts.eval.real.run_baseline_wr07_client import run as run_robot_client


class _Policy:
    def __init__(self) -> None:
        self.inputs = []

    def infer(self, observation):
        self.inputs.append(observation)
        return {"actions": np.zeros((50, 17), dtype=np.float32)}


class _Encoder:
    def __init__(self) -> None:
        self.images = None

    def reset(self):
        self.images = None

    def encode(self, images):
        self.images = images
        return np.zeros((1, 3, 1, 17, 2048), dtype=np.float32), np.ones(1, dtype=np.bool_)


def _request(prompt: str, *, with_raw: bool = True, step: int = 0) -> InferenceRequest:
    raw = {alias: np.zeros((480, 640, 3), dtype=np.uint8) for alias in ("front", "left", "right")}
    resized = {alias: np.zeros((224, 224, 3), dtype=np.uint8) for alias in raw}
    return InferenceRequest(
        request_id="request-1",
        episode_id="episode-1",
        step=step,
        observation=CanonicalObservation(
            task_id="WR-07",
            state=np.zeros(17, dtype=np.float32),
            images=resized,
            prompt=prompt,
            extra_modalities={"raw_images": raw} if with_raw else {},
        ),
    )


def test_baseline_wr07_backend_forwards_omega_and_three_cameras() -> None:
    policy = _Policy()
    encoder = _Encoder()
    backend = BaselineWR07Backend(
        policy,
        encoder,
        checkpoint=CheckpointIdentity("test://baseline", "sha256:baseline"),
        omega_fingerprint="sha256:omega",
        stats_fingerprint="sha256:stats",
    )
    validate_contract(backend.contract)
    backend.reset("episode-1")

    result = backend.infer(_request(SUBGOAL_PROMPTS["brown"]))

    assert result.actions.shape == (50, 17)
    assert set(encoder.images) == {"front", "left", "right"}
    assert encoder.images["front"].shape == (480, 640, 3)
    assert set(policy.inputs[0]["images"]) == {"agentview_left", "agentview_right", "eye_in_hand"}
    assert policy.inputs[0]["images"]["agentview_left"].shape == (480, 640, 3)
    assert policy.inputs[0]["omega_memory"].shape == (1, 3, 1, 17, 2048)
    assert policy.inputs[0]["omega_memory_mask"].tolist() == [True]
    assert set(result.timing) == {"omega_ms", "model_ms"}


def test_baseline_wr07_backend_rejects_missing_raw_images_and_prompt_change() -> None:
    backend = BaselineWR07Backend(
        _Policy(),
        _Encoder(),
        checkpoint=CheckpointIdentity("test://baseline", "sha256:baseline"),
        omega_fingerprint="sha256:omega",
        stats_fingerprint="sha256:stats",
    )
    backend.reset("episode-1")
    with pytest.raises(ValueError, match="raw_images"):
        backend.infer(_request(SUBGOAL_PROMPTS["brown"], with_raw=False))
    backend.infer(_request(SUBGOAL_PROMPTS["brown"]))
    with pytest.raises(ValueError, match="cannot change"):
        backend.infer(_request(SUBGOAL_PROMPTS["white"]))
    with pytest.raises(ValueError, match="expected decision frame 10"):
        backend.infer(_request(SUBGOAL_PROMPTS["brown"], step=20))


def test_online_omega_window_matches_training_order_and_resets() -> None:
    torch = pytest.importorskip("torch")

    class _Model:
        def __call__(self, batch):
            self.batch = batch
            return {"camera_and_register_tokens": torch.zeros(1, 18, 17, 2048)}

    encoder = CurrentOmegaEncoder.__new__(CurrentOmegaEncoder)
    encoder.device = "cpu"
    encoder.model = _Model()
    encoder.preprocess = lambda image: torch.full((3, 16, 16), float(image[0, 0, 0]))
    encoder.reset()
    for frame in range(6):
        images = {
            alias: np.full((480, 640, 3), frame, dtype=np.uint8)
            for alias in ("front", "left", "right")
        }
        memory, mask = encoder.encode(images)
        assert memory.shape == (1, 3, 1, 17, 2048)
        assert mask.tolist() == [True]
        assert encoder.model.batch.shape == (1, 18, 3, 16, 16)
        assert encoder.model.batch[0, ::3, 0, 0, 0].tolist() == list(range(frame, -1, -1)) + [0] * (
            5 - frame
        )
    encoder.reset()
    encoder.encode(
        {alias: np.full((480, 640, 3), 9, dtype=np.uint8) for alias in ("front", "left", "right")}
    )
    assert encoder.model.batch[0, ::3, 0, 0, 0].tolist() == [9] * 6


class _EchoPolicy(_Policy):
    def infer(self, observation):
        value = observation["omega_memory"][0, 0, 0, 0, 0]
        return {"actions": np.full((50, 17), value, dtype=np.float32)}


class _EchoEncoder(_Encoder):
    def encode(self, images):
        memory, mask = super().encode(images)
        memory[0, 0, 0, 0, 0] = images["front"][0, 0, 0]
        return memory, mask


def _serve_echo(port: int) -> None:
    backend = BaselineWR07Backend(
        _EchoPolicy(),
        _EchoEncoder(),
        checkpoint=CheckpointIdentity("test://baseline", "sha256:baseline"),
        omega_fingerprint="sha256:omega",
        stats_fingerprint="sha256:stats",
    )
    WebsocketPolicyServer(backend, host="127.0.0.1", port=port).serve_forever()


def test_wr07_baseline_websocket_roundtrip_preserves_raw_images() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = multiprocessing.Process(target=_serve_echo, args=(port,), daemon=True)
    process.start()
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    pytest.fail("WR07 test server did not start")
                time.sleep(0.02)
        with RemotePolicyClient(f"ws://127.0.0.1:{port}") as client:
            client.reset("episode-1")
            observation = _request(SUBGOAL_PROMPTS["brown"]).observation
            observation.extra_modalities["raw_images"]["front"][0, 0, 0] = 37
            response = client.infer(observation, episode_id="episode-1", step=0)
            assert response.actions.shape == (50, 17)
            assert response.actions[0, 0] == 37
            client.reset("episode-2")
            response = client.infer(observation, episode_id="episode-2", step=0)
            assert response.actions[0, 0] == 37

            class _Robot:
                def observe(self):
                    return np.zeros(17, dtype=np.float32), {
                        alias: np.zeros((480, 640, 3), dtype=np.uint8)
                        for alias in ("front", "left", "right")
                    }

                def hold(self):
                    pytest.fail("shadow mode should not command the robot")

            result = run_robot_client(
                client=client,
                robot=_Robot(),
                args=argparse.Namespace(
                    color="brown",
                    episode_id="episode-3",
                    max_decisions=2,
                    replan_steps=10,
                    execute=False,
                ),
            )
            assert result["mode"] == "shadow"
            assert [record["frame"] for record in result["records"]] == [0, 10]
            assert all(len(record["actions"]) == 10 for record in result["records"])
    finally:
        process.terminate()
        process.join(timeout=2)


def test_recorded_wr07_frame_replays_through_baseline_websocket() -> None:
    if sys.version_info < (3, 12):
        pytest.skip("The local LeRobot v3 reader requires Python 3.12")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from scripts.eval.real.replay_remote_dataset import evaluate_dataset

    root = Path(__file__).resolve().parents[2]
    dataset_root = root / "data/real_robot/wr07_lerobot_v3"
    annotation_root = root / "data/annotation_json_20260921/wr07_real_robot_v3"
    if not (dataset_root / "meta/info.json").is_file() or not annotation_root.is_dir():
        pytest.skip("Local WR07 recording is unavailable")
    dataset = LeRobotDataset(
        "wr07-real-v3",
        root=dataset_root,
        return_uint8=True,
        video_backend="pyav",
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = multiprocessing.Process(target=_serve_echo, args=(port,), daemon=True)
    process.start()
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    pytest.fail("WR07 test server did not start")
                time.sleep(0.02)
        args = argparse.Namespace(
            dataset_root=dataset_root,
            stride=10,
            max_frames=1,
            prompt_source="annotation",
            annotation_root=annotation_root,
            prompt=None,
            expected_config="pi05_real_robot_wr07_v3",
        )
        with RemotePolicyClient(f"ws://127.0.0.1:{port}", timeout_s=30) as client:
            result = evaluate_dataset(client=client, dataset=dataset, args=args)
        assert result["frames"] == 1
        assert result["records"][0]["frame_index"] == 0
        assert result["records"][0]["prompt"] in SUBGOAL_PROMPTS.values()
        assert len(result["records"][0]["prediction"]) == 17
    finally:
        process.terminate()
        process.join(timeout=2)
