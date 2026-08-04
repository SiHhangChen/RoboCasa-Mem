from __future__ import annotations

import multiprocessing
import socket
import time

import numpy as np
import pytest

from bench.evaluation.client import RemotePolicyClient
from bench.policy.protocol import CanonicalObservation
from bench.policy.protocol.errors import ErrorCode, ProtocolError
from bench.policy.backends.fake import FakePolicyBackend
from bench.policy.server import WebsocketPolicyServer


API_KEY = "test-secret"
pytestmark = pytest.mark.policy_e2e


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _serve(port: int) -> None:
    WebsocketPolicyServer(
        FakePolicyBackend(horizon=3, image_size=(8, 8)),
        host="127.0.0.1",
        port=port,
        api_key=API_KEY,
    ).serve_forever()


@pytest.fixture
def endpoint():
    port = _unused_port()
    process = multiprocessing.Process(target=_serve, args=(port,), daemon=True)
    process.start()
    endpoint = f"ws://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)
    else:
        process.terminate()
        pytest.fail("Fake policy server did not start")
    yield endpoint
    process.terminate()
    process.join(timeout=2)


def _observation() -> CanonicalObservation:
    return CanonicalObservation(
        task_id="WX-01",
        state=np.zeros(37, dtype=np.float32),
        images={
            "agentview_left": np.zeros((8, 8, 3), dtype=np.uint8),
            "agentview_right": np.zeros((8, 8, 3), dtype=np.uint8),
            "eye_in_hand": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        prompt="return the cup",
    )


def test_authenticated_hello_reset_and_infer(endpoint):
    with RemotePolicyClient(
        endpoint,
        api_key=API_KEY,
        connect_attempts=20,
        connect_retry_delay_s=0.02,
    ) as client:
        assert client.contract.observation.state_dim == 37
        assert client.contract.action.dim == 13
        client.reset("episode-1")
        response = client.infer(_observation(), episode_id="episode-1", step=0)

    assert response.actions.shape == (3, 13)
    assert response.actions.dtype == np.float32
    assert response.timing["total_ms"] >= 0


def test_invalid_api_key_is_rejected(endpoint):
    client = RemotePolicyClient(endpoint, api_key="wrong")
    with pytest.raises(ProtocolError) as exc:
        client.connect()
    assert exc.value.code == ErrorCode.AUTH_FAILED


def test_infer_requires_matching_reset(endpoint):
    with RemotePolicyClient(endpoint, api_key=API_KEY) as client:
        with pytest.raises(ProtocolError) as exc:
            client.infer(_observation(), episode_id="episode-1", step=0)
    assert exc.value.code == ErrorCode.EPISODE_MISMATCH
