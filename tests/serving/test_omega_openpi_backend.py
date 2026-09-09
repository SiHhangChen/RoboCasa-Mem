from pathlib import Path

import numpy as np

from bench.policy.backends.omega_openpi import (
    _checkpoint_step,
    _fingerprint_file,
    OmegaOpenPIBackend,
)
from bench.policy.protocol import CanonicalObservation, CheckpointIdentity, InferenceRequest


class _FakeOmegaPolicy:
    def __init__(self) -> None:
        self.input = None
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def infer(self, observation):
        self.input = observation
        return {"actions": np.zeros((50, 13), dtype=np.float32)}


def test_omega_backend_forwards_all_cameras_and_state37() -> None:
    policy = _FakeOmegaPolicy()
    backend = OmegaOpenPIBackend(
        policy,
        config_name="mme_vla_omega_ts01_action_modulation",
        checkpoint=CheckpointIdentity("test://omega", "sha256:omega", step=60000),
        action_horizon=50,
        image_size=(224, 224),
        policy_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
    )
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    request = InferenceRequest(
        request_id="request-1",
        episode_id="episode-1",
        step=0,
        observation=CanonicalObservation(
            task_id="TS01",
            state=np.zeros(37, dtype=np.float32),
            images={
                "agentview_left": image,
                "agentview_right": image,
                "eye_in_hand": image,
            },
            prompt="move the tray",
        ),
    )

    backend.reset(request.episode_id)
    response = backend.infer(request)

    assert policy.reset_calls == 1
    assert set(policy.input["images"]) == {
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    }
    assert policy.input["state"].shape == (37,)
    assert response.actions.shape == (50, 13)


def test_omega_checkpoint_identity_helpers(tmp_path: Path) -> None:
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"omega-weights")

    assert _checkpoint_step(tmp_path / "60000") == 60000
    assert _checkpoint_step(tmp_path / "latest") is None
    assert _fingerprint_file(checkpoint).startswith("sha256-manifest:")
