from __future__ import annotations

import numpy as np
import pytest

from bench.evaluation.action import ActionChunkQueue, validate_action_chunk
from bench.policy.protocol.errors import ProtocolError


def test_action_queue_consumes_only_replan_prefix():
    chunk = np.arange(5 * 13, dtype=np.float32).reshape(5, 13)
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
