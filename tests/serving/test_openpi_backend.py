from __future__ import annotations

import numpy as np
import pytest

from bench.policy.backends.openpi import (
    _DEFAULT_POLICY_CAMERAS,
    _camera_map_from_config,
    _policy_cameras_from_config,
    OpenPIBackend,
)
from bench.policy.protocol import (
    CanonicalObservation,
    CheckpointIdentity,
    InferenceRequest,
    PolicyContract,
)
from bench.policy.protocol.profiles import COBOT_MAGIC_RUNTIME_PROFILE
from bench.real_robot.wr07_prompts import FULL_CONFIG, SUBGOAL_CONFIG, SUBGOAL_PROMPTS, prompt_policy


class FakeOpenPIPolicy:
    def __init__(self) -> None:
        self.last_input = None
        self.reset_calls = 0
        self.reseed_seeds: list[int] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def reseed(self, seed: int) -> None:
        self.reseed_seeds.append(seed)

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


def _make_backend(policy, *, policy_cameras=None, seed_by_episode=False) -> OpenPIBackend:
    kwargs = {} if policy_cameras is None else {"policy_cameras": policy_cameras}
    return OpenPIBackend(
        policy,
        config_name="pi05_test",
        checkpoint=CheckpointIdentity("test://pi05", "sha256:pi05"),
        action_horizon=4,
        image_size=(8, 8),
        seed_by_episode=seed_by_episode,
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
    with pytest.raises(ValueError):
        OpenPIBackend(
            policy, config_name="pi05_test",
            checkpoint=CheckpointIdentity("test://pi05", "sha256:pi05"),
            action_horizon=4, image_size=(8, 8), camera_map={},
        )


def test_openpi_backend_seeded_reset_reseeds_policy_from_episode_id():
    policy = FakeOpenPIPolicy()
    backend = _make_backend(policy, seed_by_episode=True)

    backend.reset("wr-07-seed-001050")
    backend.reset("wr-07-seed-001051")

    assert policy.reseed_seeds == [1050, 1051]
    assert policy.reset_calls == 2


def test_openpi_backend_seeded_reset_rejects_episode_id_without_seed():
    policy = FakeOpenPIPolicy()
    backend = _make_backend(policy, seed_by_episode=True)

    with pytest.raises(ValueError):
        backend.reset("episode-1")
    assert policy.reseed_seeds == []


def test_openpi_backend_unseeded_reset_leaves_policy_rng_untouched():
    policy = FakeOpenPIPolicy()
    backend = _make_backend(policy)

    backend.reset("episode-1")

    assert policy.reseed_seeds == []
    assert policy.reset_calls == 1


def test_openpi_backend_contract_records_seed_policy():
    policy = FakeOpenPIPolicy()
    unseeded = _make_backend(policy)
    seeded = _make_backend(policy, seed_by_episode=True)

    assert unseeded.contract.policy["seed_policy"] == "unseeded"
    assert seeded.contract.policy["seed_policy"] == "episode_seed_v1"


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


def test_real_wr07_backend_maps_cameras_and_preserves_17d_actions():
    class RealPolicy(FakeOpenPIPolicy):
        def infer(self, observation):
            self.last_input = observation
            actions = np.zeros((4, 17), dtype=np.float32)
            actions[:, 7] = 0.25
            return {"actions": actions}

    policy = RealPolicy()
    prompt = SUBGOAL_PROMPTS["pink"]
    backend = OpenPIBackend(
        policy,
        config_name=SUBGOAL_CONFIG,
        checkpoint=CheckpointIdentity("test://real", "sha256:real"),
        action_horizon=4,
        image_size=(224, 224),
        runtime_profile=COBOT_MAGIC_RUNTIME_PROFILE,
        camera_map={"agentview_left": "front", "agentview_right": "left", "eye_in_hand": "right"},
        prompt_contract=prompt_policy(SUBGOAL_CONFIG),
    )
    assert backend.contract.runtime_profile == COBOT_MAGIC_RUNTIME_PROFILE
    assert backend.contract.observation.state_dim == backend.contract.action.dim == 17
    assert backend.contract.control_frequency_hz == 30
    assert backend.contract.hardware_execution_semantics == "unverified"
    assert PolicyContract.from_hello(backend.contract.to_hello()) == backend.contract

    request = InferenceRequest(
        request_id="real-1", episode_id="episode-1", step=0,
        observation=CanonicalObservation(
            task_id="WR-07", state=np.arange(17, dtype=np.float32),
            images={name: np.full((224, 224, 3), index, dtype=np.uint8)
                    for index, name in enumerate(("front", "left", "right"), start=1)},
            prompt=prompt,
        ),
    )
    backend.reset("episode-1")
    response = backend.infer(request)
    assert response.actions.shape == (4, 17)
    assert np.all(response.actions[:, 7] == 0.25)
    assert policy.last_input["images"]["agentview_left"][0, 0, 0] == 1
    assert policy.last_input["images"]["agentview_right"][0, 0, 0] == 2
    assert policy.last_input["images"]["eye_in_hand"][0, 0, 0] == 3
    assert policy.last_input["prompt"] == prompt

    changed = InferenceRequest(
        request_id="real-2", episode_id="episode-1", step=1,
        observation=CanonicalObservation(
            task_id="WR-07", state=request.observation.state, images=request.observation.images,
            prompt=SUBGOAL_PROMPTS["brown"],
        ),
    )
    with pytest.raises(ValueError, match="cannot change within an episode"):
        backend.infer(changed)
    backend.reset("episode-2")
    next_episode = InferenceRequest(
        request_id="real-3", episode_id="episode-2", step=0, observation=changed.observation
    )
    assert backend.infer(next_episode).actions.shape == (4, 17)


def test_real_wr07_camera_mapping_matches_training_repack():
    class _Transform:
        structure = {"images": {
            "agentview_left": "observation.images.front",
            "agentview_right": "observation.images.left",
            "eye_in_hand": "observation.images.right",
        }}

    class _Config:
        data = type("Data", (), {"repack_transforms": type("Repack", (), {"inputs": [_Transform()]})()})()

    assert _camera_map_from_config(_Config(), COBOT_MAGIC_RUNTIME_PROFILE) == {
        "agentview_left": "front", "agentview_right": "left", "eye_in_hand": "right",
    }
    assert prompt_policy(FULL_CONFIG, default_prompt="full task")["default_prompt"] == "full task"
