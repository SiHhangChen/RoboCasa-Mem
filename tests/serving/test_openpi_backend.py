from __future__ import annotations

import numpy as np

from bench.policy.protocol import (
    CanonicalObservation,
    CheckpointIdentity,
    InferenceRequest,
)
from bench.policy.backends.openpi import OpenPIBackend


class FakeOpenPIPolicy:
    def __init__(self) -> None:
        self.last_input = None
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def infer(self, observation):
        self.last_input = observation
        return {
            "actions": np.zeros((4, 13), dtype=np.float32),
            "policy_timing": {"infer_ms": 2.0},
        }


def test_openpi_backend_maps_canonical_observation():
    policy = FakeOpenPIPolicy()
    backend = OpenPIBackend(
        policy,
        config_name="pi05_test",
        checkpoint=CheckpointIdentity("test://pi05", "sha256:pi05"),
        action_horizon=4,
        image_size=(8, 8),
    )
    request = InferenceRequest(
        request_id="request-1",
        episode_id="episode-1",
        step=0,
        observation=CanonicalObservation(
            task_id="WX-01",
            state=np.zeros(37, dtype=np.float32),
            images={
                "agentview_left": np.zeros((8, 8, 3), dtype=np.uint8),
                "agentview_right": np.zeros((8, 8, 3), dtype=np.uint8),
                "eye_in_hand": np.zeros((8, 8, 3), dtype=np.uint8),
            },
            prompt="return cup",
        ),
    )

    backend.reset(request.episode_id)
    response = backend.infer(request)

    assert policy.reset_calls == 1
    assert set(policy.last_input["images"]) == {
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    }
    assert response.actions.shape == (4, 13)
    assert response.timing["model_ms"] == 2.0
