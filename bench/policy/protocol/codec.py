"""Safe msgpack codec with explicit NumPy array support."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import msgpack
import numpy as np

from .errors import ErrorCode, ProtocolError


MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_ARRAY_BYTES = 32 * 1024 * 1024
MAX_ARRAY_RANK = 8
_ARRAY_MARKER = "__membench_ndarray__"


def encode_ndarray(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind in {"V", "O", "c"}:
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            f"Unsupported ndarray dtype: {array.dtype}",
        )
    if array.ndim > MAX_ARRAY_RANK:
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            f"Array rank {array.ndim} exceeds limit {MAX_ARRAY_RANK}",
        )
    contiguous = np.ascontiguousarray(array)
    if contiguous.nbytes > MAX_ARRAY_BYTES:
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            f"Array has {contiguous.nbytes} bytes; limit is {MAX_ARRAY_BYTES}",
        )
    return {
        _ARRAY_MARKER: True,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def decode_ndarray(payload: Mapping[str, Any]) -> np.ndarray:
    try:
        dtype = np.dtype(payload["dtype"])
        shape = tuple(int(item) for item in payload["shape"])
        data = payload["data"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(ErrorCode.INVALID_REQUEST, "Malformed ndarray payload") from exc
    if dtype.hasobject or dtype.kind in {"V", "O", "c"}:
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"Unsupported ndarray dtype: {dtype}")
    if len(shape) > MAX_ARRAY_RANK or any(item < 0 for item in shape):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"Invalid ndarray shape: {shape}")
    if not isinstance(data, bytes):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, "ndarray data must be bytes")
    element_count = math.prod(shape)
    expected_bytes = element_count * dtype.itemsize
    if expected_bytes > MAX_ARRAY_BYTES:
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            f"Array has {expected_bytes} bytes; limit is {MAX_ARRAY_BYTES}",
        )
    if len(data) != expected_bytes:
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            f"ndarray byte length mismatch: expected {expected_bytes}, got {len(data)}",
        )
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def encode_message(message: Mapping[str, Any]) -> bytes:
    try:
        encoded = msgpack.packb(dict(message), default=_encode_value, use_bin_type=True)
    except ProtocolError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"Cannot encode message: {exc}") from exc
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            f"Message has {len(encoded)} bytes; limit is {MAX_MESSAGE_BYTES}",
        )
    return encoded


def decode_message(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, "Protocol frames must be binary")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            f"Message has {len(payload)} bytes; limit is {MAX_MESSAGE_BYTES}",
        )
    try:
        decoded = msgpack.unpackb(
            payload,
            raw=False,
            object_hook=_decode_value,
            strict_map_key=True,
        )
    except ProtocolError:
        raise
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"Cannot decode message: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, "Top-level message must be a map")
    return decoded


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return encode_ndarray(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported value type: {type(value).__name__}")


def _decode_value(value: dict[str, Any]) -> Any:
    if value.get(_ARRAY_MARKER) is True:
        return decode_ndarray(value)
    return value
