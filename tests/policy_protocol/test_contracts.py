from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bench.policy.protocol.contracts import (
    ACTION_SCHEMA,
    ActionContract,
    CanonicalObservation,
    CheckpointIdentity,
    InferenceRequest,
    ObservationContract,
    PolicyContract,
    validate_contract,
    validate_observation,
)
from bench.policy.protocol.errors import ErrorCode, ProtocolError
from bench.runtime.data_contract import STATE_DIM, STATE_SCHEMA


CAMERAS = ("agentview_left", "agentview_right", "eye_in_hand")


def make_contract(
    *,
    state_dim: int = STATE_DIM,
    state_schema: str = STATE_SCHEMA,
    required_modalities: tuple[str, ...] = ("rgb", "state", "prompt"),
) -> PolicyContract:
    return PolicyContract(
        server_id="fake-1",
        policy={"family": "fake", "name": "constant"},
        checkpoint=CheckpointIdentity(uri="test://constant", fingerprint="sha256:test"),
        observation=ObservationContract(
            state_schema=state_schema,
            state_dim=state_dim,
            required_cameras=CAMERAS,
            image_size=(224, 224),
            required_modalities=required_modalities,
        ),
        action=ActionContract(schema=ACTION_SCHEMA, dim=13, horizon=50),
        control_frequency_hz=20,
    )


def _images() -> dict[str, np.ndarray]:
    return {camera: np.zeros((224, 224, 3), dtype=np.uint8) for camera in CAMERAS}


def test_contract_hello_round_trip():
    contract = make_contract()
    assert PolicyContract.from_hello(contract.to_hello()) == contract


def test_wrong_state_dimension_is_rejected():
    with pytest.raises(ProtocolError, match=f"state_dim={STATE_DIM}") as exc:
        PolicyContract.from_hello(make_contract(state_dim=23).to_hello())
    assert exc.value.code == ErrorCode.CONTRACT_MISMATCH


def test_unknown_state_schema_is_rejected():
    with pytest.raises(ProtocolError, match="Unsupported state schema"):
        validate_contract(make_contract(state_schema="membench.robot_state37.v2.typo"))


def test_contract_requires_core_modalities_and_fixed_cameras():
    with pytest.raises(ProtocolError, match="Missing core"):
        validate_contract(make_contract(required_modalities=("rgb", "state")))

    duplicate = replace(
        make_contract(),
        observation=ObservationContract(
            STATE_SCHEMA,
            STATE_DIM,
            ("agentview_left", "agentview_left", "eye_in_hand"),
            (224, 224),
        ),
    )
    with pytest.raises(ProtocolError, match="required_cameras must be exactly"):
        validate_contract(duplicate)


def test_inference_request_round_trip():
    request = InferenceRequest(
        request_id="request-1",
        episode_id="episode-1",
        step=10,
        observation=CanonicalObservation(
            task_id="WX-01",
            state=np.zeros(STATE_DIM, dtype=np.float32),
            images=_images(),
            prompt="return the object",
        ),
    )
    parsed = InferenceRequest.from_wire(request.to_wire())
    assert parsed.request_id == request.request_id
    assert parsed.episode_id == request.episode_id
    np.testing.assert_array_equal(parsed.observation.state, request.observation.state)


def test_extra_modalities_cannot_override_reserved_fields():
    observation = CanonicalObservation(
        task_id="WX-01",
        state=np.zeros(STATE_DIM, dtype=np.float32),
        images=_images(),
        prompt="return the object",
        extra_modalities={"state": np.ones(STATE_DIM, dtype=np.float32)},
    )
    with pytest.raises(ProtocolError, match="reserved"):
        observation.to_wire()
    with pytest.raises(ProtocolError, match="reserved"):
        validate_observation(observation, make_contract())


def test_declared_extra_modality_must_be_present():
    contract = make_contract(required_modalities=("rgb", "state", "prompt", "depth.agentview_right"))
    observation = CanonicalObservation(
        task_id="WX-01",
        state=np.zeros(STATE_DIM, dtype=np.float32),
        images=_images(),
        prompt="return the object",
    )
    with pytest.raises(ProtocolError, match="depth.agentview_right"):
        validate_observation(observation, contract)
