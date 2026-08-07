from __future__ import annotations

import numpy as np
import pytest

from bench.policy.backends.openpi import (
    _DEFAULT_POLICY_CAMERAS,
    _policy_cameras_from_config,
    OpenPIBackend,
)
from bench.policy.protocol import (
    CanonicalObservation,
    CheckpointIdentity,
    InferenceRequest,
)


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


def _make_request() -> InferenceRequest:
    return InferenceRequest(
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


def _make_backend(policy, *, policy_cameras=None) -> OpenPIBackend:
    kwargs = {} if policy_cameras is None else {"policy_cameras": policy_cameras}
    return OpenPIBackend(
        policy,
        config_name="pi05_test",
        checkpoint=CheckpointIdentity("test://pi05", "sha256:pi05"),
        action_horizon=4,
        image_size=(8, 8),
        **kwargs,
    )


def test_openpi_backend_maps_canonical_observation():
    policy = FakeOpenPIPolicy()
    backend = _make_backend(policy)
    request = _make_request()

    backend.reset(request.episode_id)
    response = backend.infer(request)

    assert policy.reset_calls == 1
    # The default policy consumes only two cameras; the wire contract still
    # carries all three, but agentview_left must not reach the policy.
    assert set(policy.last_input["images"]) == {"agentview_right", "eye_in_hand"}
    assert "agentview_left" not in policy.last_input["images"]
    assert response.actions.shape == (4, 13)
    assert response.timing["model_ms"] == 2.0


def test_openpi_backend_forwards_three_cameras_when_configured():
    policy = FakeOpenPIPolicy()
    backend = _make_backend(
        policy,
        policy_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
    )
    request = _make_request()

    backend.reset(request.episode_id)
    response = backend.infer(request)

    assert set(policy.last_input["images"]) == {
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    }
    assert response.actions.shape == (4, 13)


def test_openpi_backend_rejects_invalid_policy_cameras():
    policy = FakeOpenPIPolicy()
    # Empty set and aliases outside the wire contract's camera set are invalid.
    with pytest.raises(ValueError):
        _make_backend(policy, policy_cameras=())
    with pytest.raises(ValueError):
        _make_backend(policy, policy_cameras=("agentview_right", "not_a_camera"))


def test_policy_cameras_derived_from_repack():
    class _Transform:
        structure = {
            "images": {
                "agentview_right": "observation.images.robot0_agentview_right",
                "eye_in_hand": "observation.images.robot0_eye_in_hand",
            },
            "state": "observation.state",
        }

    class _Repack:
        inputs = [_Transform()]

    class _Data:
        repack_transforms = _Repack()

    class _Config:
        data = _Data()

    assert _policy_cameras_from_config(_Config()) == ("agentview_right", "eye_in_hand")


def test_policy_cameras_fallback_when_no_repack():
    class _Config:
        data = None

    assert _policy_cameras_from_config(_Config()) == _DEFAULT_POLICY_CAMERAS
