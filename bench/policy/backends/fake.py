"""Deterministic CPU backend for protocol and rollout integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bench.policy.protocol import (
    ActionContract,
    CheckpointIdentity,
    InferenceRequest,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
)
from bench.policy.protocol.errors import ErrorCode, ProtocolError
from bench.runtime.data_contract import ACTION_DIM, ACTION_SCHEMA, STATE_DIM, STATE_SCHEMA


def make_constant_action_chunk(
    *,
    horizon: int,
    action: np.ndarray | None = None,
) -> np.ndarray:
    value = np.zeros(ACTION_DIM, dtype=np.float32) if action is None else np.asarray(action)
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (ACTION_DIM,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Fake action must be finite float32[{ACTION_DIM}]")
    if horizon <= 0:
        raise ValueError("Fake horizon must be positive")
    return np.repeat(value[None, :], int(horizon), axis=0)


@dataclass
class FakePolicyBackend:
    horizon: int = 4
    image_size: tuple[int, int] = (224, 224)
    required_cameras: tuple[str, ...] = ("agentview_left", "agentview_right", "eye_in_hand")
    action: np.ndarray = field(default_factory=lambda: np.zeros(ACTION_DIM, dtype=np.float32))

    def __post_init__(self) -> None:
        self._active_episode: str | None = None
        self._actions = make_constant_action_chunk(
            horizon=self.horizon,
            action=self.action,
        )
        self._contract = PolicyContract(
            server_id="fake-policy",
            policy={"family": "fake", "name": "constant_action"},
            checkpoint=CheckpointIdentity(
                uri="test://fake/constant_action",
                fingerprint="sha256:fake-constant-action-v1",
            ),
            observation=ObservationContract(
                state_schema=STATE_SCHEMA,
                state_dim=STATE_DIM,
                required_cameras=self.required_cameras,
                image_size=self.image_size,
            ),
            action=ActionContract(
                schema=ACTION_SCHEMA,
                dim=ACTION_DIM,
                horizon=self.horizon,
            ),
            control_frequency_hz=20.0,
            code_revision={"membench": "test"},
        )

    @property
    def contract(self) -> PolicyContract:
        return self._contract

    def reset(self, episode_id: str) -> None:
        self._active_episode = episode_id

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if request.episode_id != self._active_episode:
            raise ProtocolError(
                ErrorCode.EPISODE_MISMATCH,
                f"Reset episode {request.episode_id!r} before infer",
                request_id=request.request_id,
            )
        return InferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            actions=np.array(self._actions, copy=True),
            timing={"preprocess_ms": 0.0, "model_ms": 0.0, "postprocess_ms": 0.0},
        )

    def close(self) -> None:
        self._active_episode = None
