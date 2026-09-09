"""Synchronous remote policy client used by local rollout workers."""

from __future__ import annotations

import time
import os
from typing import Any, Protocol
import uuid

from bench.evaluation.action import validate_action_chunk
from bench.policy.protocol import (
    CanonicalObservation,
    InferenceRequest,
    InferenceResponse,
    PolicyContract,
)
from bench.policy.protocol.codec import MAX_MESSAGE_BYTES, decode_message, encode_message
from bench.policy.protocol.errors import (
    ErrorCode,
    ProtocolError,
    raise_for_error_response,
)


class PolicyClient(Protocol):
    """Client boundary consumed by the policy-neutral rollout runner."""

    @property
    def contract(self) -> PolicyContract: ...

    def reset(self, episode_id: str) -> None: ...

    def infer(
        self,
        observation: CanonicalObservation,
        *,
        episode_id: str,
        step: int,
    ) -> InferenceResponse: ...


class RemotePolicyClient:
    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        connect_attempts: int = 1,
        connect_retry_delay_s: float = 2.0,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
    ) -> None:
        if not endpoint.startswith(("ws://", "wss://")):
            endpoint = f"ws://{endpoint}"
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.connect_attempts = int(connect_attempts)
        self.connect_retry_delay_s = float(connect_retry_delay_s)
        self.max_message_bytes = int(max_message_bytes)
        self._connection: Any | None = None
        self._contract: PolicyContract | None = None
        self._episode_id: str | None = None

    @property
    def contract(self) -> PolicyContract:
        if self._contract is None:
            raise RuntimeError("Remote policy client is not connected")
        return self._contract

    def connect(self) -> PolicyContract:
        if self._connection is not None:
            return self.contract
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "websockets>=12 is required; install the MobileMemBench project dependencies"
            ) from exc
        headers = (
            {"Authorization": f"Api-Key {self.api_key}"} if self.api_key is not None else None
        )
        last_error: Exception | None = None
        for attempt in range(max(1, self.connect_attempts)):
            try:
                self._connection = connect(
                    self.endpoint,
                    compression=None,
                    max_size=self.max_message_bytes,
                    additional_headers=headers,
                    open_timeout=float(
                        os.environ.get("MEMBENCH_WS_OPEN_TIMEOUT", str(self.timeout_s))
                    ),
                    ping_timeout=float(os.environ.get("MEMBENCH_WS_PING_TIMEOUT", "20")),
                )
                hello = self._receive()
                raise_for_error_response(hello)
                self._contract = PolicyContract.from_hello(hello)
                return self._contract
            except (OSError, TimeoutError, ProtocolError) as error:
                last_error = error
                self.close()
                if attempt + 1 < max(1, self.connect_attempts):
                    time.sleep(self.connect_retry_delay_s)
        assert last_error is not None
        raise last_error

    def reset(self, episode_id: str) -> None:
        request_id = str(uuid.uuid4())
        response = self._exchange(
            {
                "op": "reset",
                "request_id": request_id,
                "episode_id": episode_id,
            }
        )
        if (
            response.get("op") != "reset_result"
            or response.get("request_id") != request_id
            or response.get("episode_id") != episode_id
        ):
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Malformed reset response")
        self._episode_id = episode_id

    def infer(
        self,
        observation: CanonicalObservation,
        *,
        episode_id: str,
        step: int,
    ) -> InferenceResponse:
        if episode_id != self._episode_id:
            raise ProtocolError(
                ErrorCode.EPISODE_MISMATCH,
                f"Call reset({episode_id!r}) before infer",
            )
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            episode_id=episode_id,
            step=int(step),
            observation=observation,
        )
        response = InferenceResponse.from_wire(self._exchange(request.to_wire()))
        if response.request_id != request.request_id or response.episode_id != episode_id:
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Mismatched infer response identity")
        actions = validate_action_chunk(
            response.actions,
            expected_horizon=self.contract.action.horizon,
        )
        return InferenceResponse(
            request_id=response.request_id,
            episode_id=response.episode_id,
            actions=actions,
            timing=response.timing,
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._contract = None
        self._episode_id = None

    def __enter__(self) -> "RemotePolicyClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _exchange(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("Remote policy client is not connected")
        self._connection.send(encode_message(message))
        response = self._receive()
        raise_for_error_response(response)
        return response

    def _receive(self) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("Remote policy client is not connected")
        frame = self._connection.recv(timeout=self.timeout_s)
        return decode_message(frame)
