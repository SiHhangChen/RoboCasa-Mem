from __future__ import annotations

import msgpack
import numpy as np
import pytest

from bench.policy.protocol.codec import decode_message, encode_message
from bench.policy.protocol.errors import ErrorCode, ProtocolError


def test_numpy_round_trip_preserves_dtype_shape_and_values():
    image = np.arange(3 * 4 * 2, dtype=np.uint8).reshape(3, 4, 2)
    state = np.linspace(-1, 1, 37, dtype=np.float32)

    decoded = decode_message(encode_message({"image": image, "state": state}))

    np.testing.assert_array_equal(decoded["image"], image)
    np.testing.assert_array_equal(decoded["state"], state)
    assert decoded["image"].dtype == np.uint8
    assert decoded["state"].dtype == np.float32


def test_object_and_complex_arrays_are_rejected():
    for array in (
        np.asarray([object()], dtype=object),
        np.asarray([1 + 2j], dtype=np.complex64),
    ):
        with pytest.raises(ProtocolError, match="Unsupported ndarray dtype") as exc:
            encode_message({"value": array})
        assert exc.value.code == ErrorCode.INVALID_REQUEST


def test_malformed_array_byte_length_is_rejected():
    payload = msgpack.packb(
        {
            "array": {
                "__membench_ndarray__": True,
                "dtype": "<f4",
                "shape": [37],
                "data": b"too short",
            }
        },
        use_bin_type=True,
    )

    with pytest.raises(ProtocolError, match="byte length mismatch"):
        decode_message(payload)


def test_top_level_message_must_be_map():
    with pytest.raises(ProtocolError, match="Top-level"):
        decode_message(msgpack.packb([1, 2, 3]))
