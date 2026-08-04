from __future__ import annotations

import multiprocessing
import socket
import time

import numpy as np
import pytest

from bench.evaluation.adapter import EpisodeRuntime
from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.client import RemotePolicyClient
from bench.evaluation.rollout import RolloutRunner
from bench.evaluation.suite import ArtifactConfig, EvalSuiteConfig, RolloutConfig
from bench.policy.protocol import CanonicalObservation
from bench.policy.backends.fake import FakePolicyBackend
from bench.policy.server import WebsocketPolicyServer


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _serve(port: int) -> None:
    WebsocketPolicyServer(
        FakePolicyBackend(
            horizon=2,
            image_size=(2, 2),
            required_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
        ),
        host="127.0.0.1",
        port=port,
    ).serve_forever()


class _Env:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, action):
        assert np.asarray(action).shape == (13,)
        self.steps += 1
        return {"step": self.steps}, 0.0, False, {"success": self.steps >= 2}

    def close(self):
        pass


class _Adapter:
    task_id = "WX-01"
    control_frequency_hz = 20.0

    def create_episode(self, *, seed, policy_contract, suite):
        del policy_contract, suite
        return EpisodeRuntime(
            episode_id=f"e2e-{seed}",
            seed=seed,
            env=_Env(),
            observation={"step": 0},
            prompt="complete the integration task",
        )

    def build_observation(self, runtime):
        return CanonicalObservation(
            task_id=self.task_id,
            state=np.zeros(37, dtype=np.float32),
            images={
                "agentview_left": np.zeros((2, 2, 3), dtype=np.uint8),
                "agentview_right": np.zeros((2, 2, 3), dtype=np.uint8),
                "eye_in_hand": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            prompt=runtime.prompt,
        )

    def check_success(self, runtime, info):
        del runtime
        return bool(info["success"])

    def diagnostics(self, runtime):
        return {"steps": runtime.env.steps}

    def close(self, runtime):
        runtime.env.close()


@pytest.mark.policy_e2e
def test_remote_client_runner_artifacts_summary_and_resume(tmp_path):
    port = _unused_port()
    process = multiprocessing.Process(target=_serve, args=(port,), daemon=True)
    process.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)
    else:
        process.terminate()
        raise AssertionError("fake policy server did not start")

    suite = EvalSuiteConfig(
        suite_version="e2e_v1",
        task_id="WX-01",
        env_split="pretrain",
        seeds=(1000, 1001),
        rollout=RolloutConfig(max_steps=4, replan_steps=2),
        artifacts=ArtifactConfig(record_video=False),
    )
    writer = ArtifactWriter(tmp_path / "run", expected_seeds=suite.seeds)
    client = RemotePolicyClient(
        f"ws://127.0.0.1:{port}",
        connect_attempts=20,
        connect_retry_delay_s=0.02,
    )
    try:
        client.connect()
        runner = RolloutRunner(client=client, suite=suite, artifacts=writer)
        results = runner.run_suite(_Adapter())
        resumed = runner.run_suite(_Adapter(), resume=True)
    finally:
        client.close()
        process.terminate()
        process.join(timeout=2)

    assert [result.status for result in results] == ["success", "success"]
    assert resumed == []
    summary = writer.write_summary()
    assert summary["counts"]["success"] == 2
    assert summary["completion_rate"] == 1.0
    assert summary["publishable"] is True
