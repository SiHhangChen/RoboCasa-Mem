from __future__ import annotations

from dataclasses import replace

import pytest

from bench.policy.protocol import (
    ActionContract,
    CheckpointIdentity,
    ObservationContract,
    PolicyContract,
)
from bench.policy.protocol.errors import ErrorCode, ProtocolError
from bench.real_robot.contracts import (
    ACTION_DIM,
    ACTION_SCHEMA,
    CAMERA_ALIASES,
    CONTROL_FREQUENCY_HZ,
    IMAGE_SIZE,
    RUNTIME_PROFILE,
    STATE_DIM,
    STATE_SCHEMA,
)


def make_real_contract() -> PolicyContract:
    return PolicyContract(
        server_id="real-test",
        policy={"family": "fake", "name": "real"},
        checkpoint=CheckpointIdentity("test://real", "sha256:test"),
        observation=ObservationContract(
            state_schema=STATE_SCHEMA,
            state_dim=STATE_DIM,
            required_cameras=CAMERA_ALIASES,
            image_size=IMAGE_SIZE,
        ),
        action=ActionContract(schema=ACTION_SCHEMA, dim=ACTION_DIM, horizon=10),
        control_frequency_hz=CONTROL_FREQUENCY_HZ,
        runtime_profile=RUNTIME_PROFILE,
        hardware_execution_semantics="unverified",
    )


def test_real_profile_hello_round_trip() -> None:
    contract = make_real_contract()
    assert PolicyContract.from_hello(contract.to_hello()) == contract


@pytest.mark.parametrize(
    "change",
    [
        lambda c: replace(c, observation=replace(c.observation, state_dim=37)),
        lambda c: replace(c, observation=replace(c.observation, required_cameras=("front", "left"))),
        lambda c: replace(c, control_frequency_hz=20.0),
    ],
)
def test_real_profile_rejects_contract_mismatch(change) -> None:
    with pytest.raises(ProtocolError) as exc:
        PolicyContract.from_hello(change(make_real_contract()).to_hello())
    assert exc.value.code == ErrorCode.CONTRACT_MISMATCH


def test_unregistered_profile_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="Unsupported runtime profile"):
        PolicyContract.from_hello(
            replace(make_real_contract(), runtime_profile="membench.unknown.v1").to_hello()
        )


def test_real_profile_requires_unverified_hardware_semantics() -> None:
    with pytest.raises(ProtocolError, match="hardware_execution_semantics"):
        PolicyContract.from_hello(
            replace(make_real_contract(), hardware_execution_semantics=None).to_hello()
        )


def test_legacy_hello_without_runtime_profile_defaults_to_simulation() -> None:
    hello = PolicyContract(
        server_id="legacy",
        policy={"family": "fake"},
        checkpoint=CheckpointIdentity("test://legacy", "sha256:test"),
        observation=ObservationContract(
            state_schema="membench.robot_state37.v2",
            state_dim=37,
            required_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
            image_size=(224, 224),
        ),
        action=ActionContract(schema="membench.pandaomron_hybrid13.v2", dim=13, horizon=2),
        control_frequency_hz=20.0,
    ).to_hello()
    hello.pop("runtime_profile")
    assert PolicyContract.from_hello(hello).runtime_profile == "membench.robocasa_pandaomron.v2"
