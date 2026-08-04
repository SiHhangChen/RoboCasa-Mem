"""Validation and deterministic consumption of MobileMemBench action chunks."""

from __future__ import annotations

from collections import deque

import numpy as np

from bench.policy.protocol.errors import ErrorCode, ProtocolError


ACTION_DIM = 13


def validate_action13(action: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (ACTION_DIM,):
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"Expected action shape ({ACTION_DIM},), got {value.shape}",
        )
    if not np.all(np.isfinite(value)):
        raise ProtocolError(ErrorCode.INFERENCE_ERROR, "Action contains non-finite values")
    return np.ascontiguousarray(value)


def validate_action_chunk(
    actions: np.ndarray,
    *,
    expected_horizon: int | None = None,
) -> np.ndarray:
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != ACTION_DIM or value.shape[0] <= 0:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"Expected action chunk shape [H, {ACTION_DIM}], got {value.shape}",
        )
    if expected_horizon is not None and value.shape[0] != expected_horizon:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"Expected action horizon {expected_horizon}, got {value.shape[0]}",
        )
    if not np.all(np.isfinite(value)):
        raise ProtocolError(
            ErrorCode.INFERENCE_ERROR,
            "Action chunk contains non-finite values",
        )
    return np.ascontiguousarray(value)


class ActionChunkQueue:
    def __init__(self) -> None:
        self._actions: deque[np.ndarray] = deque()

    def __len__(self) -> int:
        return len(self._actions)

    def clear(self) -> None:
        self._actions.clear()

    def push(self, actions: np.ndarray, *, limit: int | None = None) -> int:
        chunk = validate_action_chunk(actions)
        count = chunk.shape[0] if limit is None else min(chunk.shape[0], int(limit))
        if count <= 0:
            raise ValueError("Action queue limit must be positive")
        self._actions.extend(np.array(action, copy=True) for action in chunk[:count])
        return count

    def pop(self) -> np.ndarray:
        if not self._actions:
            raise IndexError("Action chunk queue is empty")
        return self._actions.popleft()
