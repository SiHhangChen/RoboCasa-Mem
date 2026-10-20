from __future__ import annotations

import numpy as np
import pytest

from bench.evaluation.action import ActionChunkQueue, validate_action_chunk
from bench.policy.protocol.errors import ProtocolError
from bench.runtime.data_contract import validate_action


def test_action_queue_consumes_only_replan_prefix():
    chunk = np.arange(5 * 13, dtype=np.float32).reshape(5, 13)
    chunk[:, 7] = 0.0
    queue = ActionChunkQueue()

    assert queue.push(chunk, limit=2) == 2
    np.testing.assert_array_equal(queue.pop(), chunk[0])
    np.testing.assert_array_equal(queue.pop(), chunk[1])
    assert len(queue) == 0


def test_single_action_is_promoted_to_chunk():
    action = np.zeros(13, dtype=np.float32)
    assert validate_action_chunk(action).shape == (1, 13)


def test_non_finite_action_is_rejected():
    chunk = np.zeros((2, 13), dtype=np.float32)
    chunk[0, 0] = np.nan
    with pytest.raises(ProtocolError, match="non-finite"):
        validate_action_chunk(chunk)


def test_action_chunk_discretizes_gripper_with_deadband():
    chunk = np.zeros((6, 13), dtype=np.float32)
    chunk[:, 7] = [-1.0, -0.101, -0.099, 0.099, 0.101, 1.0]

    validated = validate_action_chunk(chunk)

    np.testing.assert_array_equal(
        validated[:, 7],
        np.array([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(validated[:, :7], chunk[:, :7])
    np.testing.assert_array_equal(validated[:, 8:], chunk[:, 8:])


def test_runtime_action_boundary_discretizes_gripper_with_deadband():
    action = np.arange(13, dtype=np.float32) / 20.0
    action[7] = -0.05

    validated = validate_action(action)

    assert validated[7] == 0.0
    np.testing.assert_array_equal(validated[:7], action[:7])
