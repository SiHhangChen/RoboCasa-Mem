from __future__ import annotations

import multiprocessing
import socket
import time

import numpy as np
import pytest

from bench.evaluation.client import RemotePolicyClient
from bench.policy.backends.fake import FakePolicyBackend
from bench.policy.protocol import CanonicalObservation
from bench.policy.server import WebsocketPolicyServer
from bench.real_robot.contracts import RUNTIME_PROFILE


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _serve(port: int) -> None:
    action = np.zeros(17, dtype=np.float32)
    action[7] = 0.25
    WebsocketPolicyServer(
        FakePolicyBackend(
            horizon=2,
            runtime_profile=RUNTIME_PROFILE,
            action=action,
        ),
        host="127.0.0.1",
        port=port,
    ).serve_forever()


@pytest.mark.policy_e2e
def test_real_robot_profile_round_trips_over_websocket() -> None:
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
        raise AssertionError("fake real-robot policy server did not start")

    client = RemotePolicyClient(
        f"ws://127.0.0.1:{port}",
        connect_attempts=20,
        connect_retry_delay_s=0.02,
    )
    try:
        contract = client.connect()
        assert contract.runtime_profile == RUNTIME_PROFILE
        assert contract.hardware_execution_semantics == "unverified"
        client.reset("wr07-real-test")
        response = client.infer(
            CanonicalObservation(
                task_id="WR-07",
                state=np.zeros(17, dtype=np.float32),
                images={
                    camera: np.zeros((224, 224, 3), dtype=np.uint8)
                    for camera in ("front", "left", "right")
                },
                prompt="move the item to the matching-color container",
            ),
            episode_id="wr07-real-test",
            step=0,
        )
    finally:
        client.close()
        process.terminate()
        process.join(timeout=2)

    assert response.actions.shape == (2, 17)
    np.testing.assert_array_equal(response.actions[:, 7], np.array([0.25, 0.25]))
