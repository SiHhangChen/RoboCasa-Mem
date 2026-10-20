"""Protocol error types and wire representations."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    AUTH_FAILED = "AUTH_FAILED"
    UNSUPPORTED_PROTOCOL = "UNSUPPORTED_PROTOCOL"
    INVALID_REQUEST = "INVALID_REQUEST"
    CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
    EPISODE_MISMATCH = "EPISODE_MISMATCH"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    SERVER_BUSY = "SERVER_BUSY"


class ProtocolError(RuntimeError):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool = False,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.request_id = request_id


def make_error_response(error: ProtocolError) -> dict[str, Any]:
    return {
        "op": "error",
        "request_id": error.request_id,
        "code": error.code.value,
        "message": str(error),
        "retryable": error.retryable,
    }


def raise_for_error_response(message: Mapping[str, Any]) -> None:
    if message.get("op") != "error":
        return
    try:
        code = ErrorCode(str(message["code"]))
    except (KeyError, ValueError) as exc:
        raise ProtocolError(ErrorCode.INVALID_REQUEST, "Malformed error response") from exc
    raise ProtocolError(
        code,
        str(message.get("message", code.value)),
        retryable=bool(message.get("retryable", False)),
        request_id=_optional_string(message.get("request_id")),
    )


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
