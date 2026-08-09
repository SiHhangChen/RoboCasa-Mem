"""Authenticated WebSocket server for ``membench-policy/v1``."""

from __future__ import annotations

import asyncio
import hmac
import http
import logging
import time
from typing import Any, Mapping

from bench.policy.protocol import InferenceRequest, validate_observation
from bench.policy.protocol.codec import MAX_MESSAGE_BYTES, decode_message, encode_message
from bench.policy.protocol.errors import ErrorCode, ProtocolError, make_error_response
from bench.policy.backends.base import PolicyBackend


LOGGER = logging.getLogger(__name__)


class WebsocketPolicyServer:
    def __init__(
        self,
        backend: PolicyBackend,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        api_key: str | None = None,
        inference_timeout_s: float = 30.0,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
    ) -> None:
        if inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be positive")
        self.backend = backend
        self.host = host
        self.port = int(port)
        self.api_key = api_key
        self.inference_timeout_s = float(inference_timeout_s)
        self.max_message_bytes = int(max_message_bytes)
        self._inference_lock: asyncio.Lock | None = None
        self._client_lock: asyncio.Lock | None = None
        self._inference_poisoned = False

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise RuntimeError(
                "websockets>=12 is required; install the MobileMemBench project dependencies"
            ) from exc
        # Surface the websockets library's own listening/connection logs, like the
        # original OpenPI server does.
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self._inference_lock = asyncio.Lock()
        self._client_lock = asyncio.Lock()
        LOGGER.info("Serving %s on ws://%s:%d", self.backend.contract.server_id, self.host, self.port)
        try:
            async with serve(
                self.handle_connection,
                self.host,
                self.port,
                compression=None,
                max_size=self.max_message_bytes,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        finally:
            self.backend.close()

    async def handle_connection(self, websocket: Any) -> None:
        if not self._authorized(websocket):
            await websocket.send(
                encode_message(
                    make_error_response(
                        ProtocolError(ErrorCode.AUTH_FAILED, "Invalid or missing API key")
                    )
                )
            )
            await websocket.close(code=1008, reason="Authentication failed")
            return
        assert self._client_lock is not None
        if self._client_lock.locked():
            await websocket.send(
                encode_message(
                    make_error_response(
                        ProtocolError(
                            ErrorCode.SERVER_BUSY,
                            "Another inference client is active",
                            retryable=True,
                        )
                    )
                )
            )
            await websocket.close(code=1013, reason="Server busy")
            return

        async with self._client_lock:
            LOGGER.info("Connection from %s opened", websocket.remote_address)
            await websocket.send(encode_message(self.backend.contract.to_hello()))
            active_episode: str | None = None
            async for frame in websocket:
                request_id: str | None = None
                try:
                    message = decode_message(frame)
                    request_id = _request_id(message)
                    op = message.get("op")
                    if op == "reset":
                        response, active_episode = await self.handle_reset(message)
                    elif op == "infer":
                        response = await self.handle_infer(message, active_episode)
                    else:
                        raise ProtocolError(
                            ErrorCode.INVALID_REQUEST,
                            f"Unsupported operation: {op!r}",
                            request_id=request_id,
                        )
                except ProtocolError as error:
                    if error.request_id is None:
                        error.request_id = request_id
                    response = make_error_response(error)
                except Exception:
                    LOGGER.exception("Unhandled policy server request error")
                    response = make_error_response(
                        ProtocolError(
                            ErrorCode.INFERENCE_ERROR,
                            "Internal policy server error",
                            request_id=request_id,
                        )
                    )
                await websocket.send(encode_message(response))
            LOGGER.info("Connection from %s closed", websocket.remote_address)

    async def handle_reset(
        self,
        message: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        request_id = _required_string(message, "request_id")
        episode_id = _required_string(message, "episode_id")
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.backend.reset, episode_id),
                timeout=self.inference_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ProtocolError(
                ErrorCode.INFERENCE_TIMEOUT,
                "Policy reset timed out",
                retryable=True,
                request_id=request_id,
            ) from exc
        return (
            {
                "op": "reset_result",
                "request_id": request_id,
                "episode_id": episode_id,
            },
            episode_id,
        )

    async def handle_infer(
        self,
        message: Mapping[str, Any],
        active_episode: str | None,
    ) -> dict[str, Any]:
        request = InferenceRequest.from_wire(message)
        validate_observation(request.observation, self.backend.contract)
        if self._inference_poisoned:
            raise ProtocolError(
                ErrorCode.INFERENCE_ERROR,
                "A previous inference timed out; restart the policy server",
                request_id=request.request_id,
            )
        if request.episode_id != active_episode:
            raise ProtocolError(
                ErrorCode.EPISODE_MISMATCH,
                f"Active episode is {active_episode!r}, got {request.episode_id!r}",
                request_id=request.request_id,
            )
        assert self._inference_lock is not None
        if self._inference_lock.locked():
            raise ProtocolError(
                ErrorCode.SERVER_BUSY,
                "Inference is already in progress",
                retryable=True,
                request_id=request.request_id,
            )
        start = time.monotonic()
        try:
            async with self._inference_lock:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.backend.infer, request),
                    timeout=self.inference_timeout_s,
                )
        except asyncio.TimeoutError as exc:
            self._inference_poisoned = True
            raise ProtocolError(
                ErrorCode.INFERENCE_TIMEOUT,
                "Policy inference timed out",
                retryable=False,
                request_id=request.request_id,
            ) from exc
        if response.request_id != request.request_id:
            raise ProtocolError(
                ErrorCode.INFERENCE_ERROR,
                "Backend returned a mismatched request_id",
                request_id=request.request_id,
            )
        timing = dict(response.timing)
        timing["total_ms"] = (time.monotonic() - start) * 1000.0
        return {
            **response.to_wire(),
            "timing": timing,
        }

    def _authorized(self, websocket: Any) -> bool:
        if self.api_key is None:
            return True
        request = getattr(websocket, "request", None)
        headers = getattr(request, "headers", {})
        authorization = headers.get("Authorization", "")
        prefix = "Api-Key "
        if not authorization.startswith(prefix):
            return False
        return hmac.compare_digest(authorization[len(prefix) :], self.api_key)


def _request_id(message: Mapping[str, Any]) -> str | None:
    value = message.get("request_id")
    return value if isinstance(value, str) and value else None


def _required_string(message: Mapping[str, Any], key: str) -> str:
    value = message.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"{key} must be a non-empty string")
    return value


def _health_check(connection: Any, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
