"""Validation and deterministic consumption of MobileMemBench action chunks."""

from __future__ import annotations

from collections import deque

import numpy as np

from bench.policy.protocol.errors import ErrorCode, ProtocolError
from bench.runtime.data_contract import ACTION_SCHEMA as SIM_ACTION_SCHEMA
from bench.runtime.data_contract import normalize_gripper_command


ACTION_DIM = 13


def validate_action13(action: np.ndarray) -> np.ndarray:
    value = validate_action_chunk(
        action,
        expected_horizon=1,
        expected_dim=ACTION_DIM,
        action_schema=SIM_ACTION_SCHEMA,
    )
    if np.asarray(action).ndim != 1:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"Expected action shape ({ACTION_DIM},), got {np.asarray(action).shape}",
        )
    return value[0]


def validate_action_chunk(
    actions: np.ndarray,
    *,
    expected_horizon: int | None = None,
    expected_dim: int = ACTION_DIM,
    action_schema: str = SIM_ACTION_SCHEMA,
) -> np.ndarray:
    if expected_dim <= 0:
        raise ValueError("expected_dim must be positive")
    value = np.asarray(actions, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[1] != expected_dim or value.shape[0] <= 0:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"Expected action chunk shape [H, {expected_dim}], got {value.shape}",
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
    value = np.array(value, copy=True, order="C")
    if action_schema == SIM_ACTION_SCHEMA:
        value[:, 7] = normalize_gripper_command(value[:, 7])
    return value


class ActionChunkQueue:
    def __init__(
        self,
        *,
        expected_dim: int = ACTION_DIM,
        action_schema: str = SIM_ACTION_SCHEMA,
    ) -> None:
        self._actions: deque[np.ndarray] = deque()
        self.expected_dim = int(expected_dim)
        self.action_schema = str(action_schema)

    def __len__(self) -> int:
        return len(self._actions)

    def clear(self) -> None:
        self._actions.clear()

    def push(self, actions: np.ndarray, *, limit: int | None = None) -> int:
        chunk = validate_action_chunk(
            actions,
            expected_dim=self.expected_dim,
            action_schema=self.action_schema,
        )
        count = chunk.shape[0] if limit is None else min(chunk.shape[0], int(limit))
        if count <= 0:
            raise ValueError("Action queue limit must be positive")
        self._actions.extend(np.array(action, copy=True) for action in chunk[:count])
        return count

    def pop(self) -> np.ndarray:
        if not self._actions:
            raise IndexError("Action chunk queue is empty")
        return self._actions.popleft()
