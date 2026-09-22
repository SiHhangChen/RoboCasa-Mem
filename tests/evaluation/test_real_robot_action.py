from __future__ import annotations

import numpy as np
import pytest

from bench.evaluation.action import ActionChunkQueue, validate_action_chunk
from bench.policy.protocol.errors import ProtocolError
from bench.real_robot.contracts import ACTION_DIM, ACTION_SCHEMA


def test_real_robot_action_is_not_gripper_quantized() -> None:
    action = np.zeros((2, ACTION_DIM), dtype=np.float32)
    action[:, 7] = (-0.25, 0.25)
    validated = validate_action_chunk(
        action,
        expected_dim=ACTION_DIM,
        action_schema=ACTION_SCHEMA,
    )
    np.testing.assert_array_equal(validated[:, 7], action[:, 7])


def test_real_robot_action_wrong_dimension_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="17"):
        validate_action_chunk(
            np.zeros((2, 13), dtype=np.float32),
            expected_dim=ACTION_DIM,
            action_schema=ACTION_SCHEMA,
        )


def test_real_robot_queue_preserves_17d_actions() -> None:
    queue = ActionChunkQueue(expected_dim=ACTION_DIM, action_schema=ACTION_SCHEMA)
    action = np.arange(2 * ACTION_DIM, dtype=np.float32).reshape(2, ACTION_DIM)
    assert queue.push(action) == 2
    np.testing.assert_array_equal(queue.pop(), action[0])
