from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bench.policy.backends.lerobot import (
    LeRobotDiffusionBackend,
    _checkpoint_step,
    inspect_lerobot_diffusion_checkpoint,
)
from bench.policy.protocol import (
    ActionContract,
    CanonicalObservation,
    CheckpointIdentity,
    InferenceRequest,
    ObservationContract,
    PolicyContract,
)
from bench.runtime.data_contract import ACTION_SCHEMA, STATE_SCHEMA

CAMERA_FEATURES = {
    "agentview_left": "observation.images.robot0_agentview_left",
    "agentview_right": "observation.images.robot0_agentview_right",
    "eye_in_hand": "observation.images.robot0_eye_in_hand",
}


class FakeDiffusionPolicy:
    def __init__(self) -> None:
        self.batch = None
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(self, batch):
        assert not torch.is_grad_enabled()
        self.batch = batch
        return torch.cat((torch.rand(1, 12), torch.tensor([[0.25]])), dim=1)


def _contract() -> PolicyContract:
    return PolicyContract(
        server_id="diffusion-test",
        policy={
            "family": "lerobot",
            "name": "diffusion",
            "camera_features": CAMERA_FEATURES,
            "n_obs_steps": 2,
            "native_action_horizon": 47,
            "seed_policy": "episode_seed_v1",
            "wire_action_mode": "select_action_v1",
        },
        checkpoint=CheckpointIdentity("test://diffusion", "sha256:test", step=40000),
        observation=ObservationContract(
            state_schema=STATE_SCHEMA,
            state_dim=37,
            required_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
            image_size=(2, 2),
        ),
        action=ActionContract(schema=ACTION_SCHEMA, dim=13, horizon=1),
        control_frequency_hz=20.0,
    )


def _backend() -> LeRobotDiffusionBackend:
    policy = FakeDiffusionPolicy()
    return LeRobotDiffusionBackend(
        policy=policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda action: action,
        contract=_contract(),
        device="cpu",
        torch_module=torch,
        seed_by_episode=True,
    )


def _request() -> InferenceRequest:
    return InferenceRequest(
        request_id="request-1",
        episode_id="wa-05-seed-001049",
        step=0,
        observation=CanonicalObservation(
            task_id="WA-05",
            state=np.arange(37, dtype=np.float32),
            images={
                alias: np.full((2, 2, 3), index + 1, dtype=np.uint8)
                for index, alias in enumerate(CAMERA_FEATURES)
            },
            prompt="complete the task",
        ),
    )


def test_diffusion_backend_returns_one_action_and_forwards_all_inputs() -> None:
    backend = _backend()
    request = _request()
    backend.reset(request.episode_id)

    response = backend.infer(request)

    assert backend.policy.reset_calls == 1
    assert response.actions.shape == (1, 13)
    assert response.actions.dtype == np.float32
    assert abs(response.actions[0, 7]) == 1.0
    np.testing.assert_array_equal(
        backend.policy.batch["observation.state"], request.observation.state
    )
    for feature in CAMERA_FEATURES.values():
        assert backend.policy.batch[feature].shape == (3, 2, 2)
        assert backend.policy.batch[feature].dtype == torch.float32
        assert float(backend.policy.batch[feature].max()) <= 1.0


def test_diffusion_seeded_reset_is_reproducible() -> None:
    backend = _backend()
    request = _request()

    backend.reset("wa-05-seed-001049")
    first = backend.infer(request).actions.copy()
    backend.reset("wa-05-seed-001050")
    other = backend.infer(request).actions.copy()
    backend.reset("wa-05-seed-001049")
    second = backend.infer(request).actions.copy()

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_diffusion_seeded_reset_requires_seed_in_episode_id() -> None:
    with pytest.raises(ValueError, match="seed-<integer>"):
        _backend().reset("custom-episode")


def _checkpoint(path: Path, *, n_action_steps: int = 47) -> Path:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"test")
    (path / "config.json").write_text(
        json.dumps(
            {
                "type": "diffusion",
                "n_obs_steps": 2,
                "horizon": 48,
                "n_action_steps": n_action_steps,
                "input_features": {
                    "observation.state": {"type": "STATE", "shape": [37]},
                    **{
                        feature: {"type": "VISUAL", "shape": [3, 224, 224]}
                        for feature in CAMERA_FEATURES.values()
                    },
                },
                "output_features": {"action": {"type": "ACTION", "shape": [13]}},
            }
        ),
        encoding="utf-8",
    )
    (path / "membench_policy.json").write_text(
        json.dumps(
            {
                "policy_type": "diffusion",
                "state_schema": STATE_SCHEMA,
                "action_schema": ACTION_SCHEMA,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_diffusion_checkpoint_contract_is_extracted(tmp_path: Path) -> None:
    inspected = inspect_lerobot_diffusion_checkpoint(_checkpoint(tmp_path / "model"))

    assert inspected["image_size"] == (224, 224)
    assert inspected["camera_features"] == CAMERA_FEATURES
    assert inspected["n_obs_steps"] == 2
    assert inspected["horizon"] == 48
    assert inspected["native_action_horizon"] == 47


def test_diffusion_checkpoint_rejects_impossible_action_horizon(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "model", n_action_steps=48)
    with pytest.raises(ValueError, match="n_action_steps"):
        inspect_lerobot_diffusion_checkpoint(checkpoint)


def test_checkpoint_step_reads_training_state(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "040000" / "pretrained_model")
    state = checkpoint.parent / "training_state"
    state.mkdir()
    (state / "training_step.json").write_text('{"step": 40000}', encoding="utf-8")
    assert _checkpoint_step(checkpoint) == 40000
