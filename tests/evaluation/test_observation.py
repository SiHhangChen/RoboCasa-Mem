from __future__ import annotations

import numpy as np
import pytest

from bench.evaluation.observation import build_canonical_observation, build_state37
from bench.policy.protocol.contracts import (
    ACTION_SCHEMA,
    ActionContract,
    CheckpointIdentity,
    ObservationContract,
    PolicyContract,
)
from bench.policy.protocol.errors import ProtocolError
from bench.runtime.data_contract import STATE_SCHEMA


def _contract(*, image_size: tuple[int, int] = (224, 224)) -> PolicyContract:
    return PolicyContract(
        server_id="fake",
        policy={"family": "fake"},
        checkpoint=CheckpointIdentity("test://fake", "sha256:fake"),
        observation=ObservationContract(
            STATE_SCHEMA,
            37,
            ("agentview_left", "agentview_right", "eye_in_hand"),
            image_size,
        ),
        action=ActionContract(ACTION_SCHEMA, 13, 4),
        control_frequency_hz=20,
    )


def _raw_observation() -> dict[str, np.ndarray]:
    return {
        "robot0_arm_joint_qpos": np.arange(7, dtype=np.float64),
        "robot0_arm_joint_qvel": np.arange(7, dtype=np.float64) + 10,
        "robot0_gripper_qpos": np.asarray([0.01, 0.02]),
        "robot0_gripper_qvel": np.asarray([0.03, 0.04]),
        "robot0_base_pos": np.asarray([1, 2, 3], dtype=np.float64),
        "robot0_base_quat": np.asarray([-0.8, 0.1, 0.2, 0.3]),
        "robot0_base_joint_qvel": np.asarray([0.1, 0.2, 0.3]),
        "robot0_torso_qpos": np.asarray([0.2]),
        "robot0_torso_qvel": np.asarray([0.0]),
        "robot0_base_to_eef_pos": np.asarray([4, 5, 6]),
        "robot0_base_to_eef_quat_site": np.asarray([0.1, -0.9, 0.2, 0.3]),
        "robot0_agentview_left_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot0_agentview_right_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((224, 224, 3), dtype=np.uint8),
    }


def test_build_state37_order_dtype_and_canonical_sign():
    state = build_state37(_raw_observation())

    assert state.shape == (37,)
    assert state.dtype == np.float32
    np.testing.assert_array_equal(state[18:21], [1, 2, 3])
    expected_base_quat = np.asarray([0.8, -0.1, -0.2, -0.3])
    expected_base_quat /= np.linalg.norm(expected_base_quat)
    np.testing.assert_allclose(state[21:25], expected_base_quat)
    np.testing.assert_array_equal(state[30:33], [4, 5, 6])
    expected_eef_quat = np.asarray([-0.1, 0.9, -0.2, -0.3])
    expected_eef_quat /= np.linalg.norm(expected_eef_quat)
    np.testing.assert_allclose(state[33:37], expected_eef_quat)


def test_build_state37_rejects_wrong_field_dimension():
    observation = _raw_observation()
    observation["robot0_gripper_qpos"] = np.zeros(3)

    with pytest.raises(ValueError, match="expected state37"):
        build_state37(observation)


def test_canonical_observation_validates_rgb_shape():
    observation = _raw_observation()
    observation["robot0_eye_in_hand_image"] = np.zeros((223, 224, 3), dtype=np.uint8)

    with pytest.raises(ProtocolError, match="eye_in_hand"):
        build_canonical_observation(
            observation,
            task_id="WX-01",
            prompt="return the cup",
            contract=_contract(),
        )
