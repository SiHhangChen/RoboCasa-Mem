from __future__ import annotations

import numpy as np
import pytest

from bench.execution.planning.model_alignment import (
    canonicalize_quat_xyzw_sign,
    quat_xyzw_to_mat,
)


def test_canonicalize_quaternion_maps_antipodes_to_same_value():
    quat = np.array([0.70716, 0.70704, -0.00392, -0.00218], dtype=np.float32)

    canonical = canonicalize_quat_xyzw_sign(quat)
    canonical_antipode = canonicalize_quat_xyzw_sign(-quat)

    np.testing.assert_allclose(canonical, canonical_antipode)
    assert canonical[int(np.argmax(np.abs(canonical)))] > 0


def test_canonicalize_quaternion_preserves_rotation_and_norm():
    quat = np.array([-0.70716, -0.70704, 0.00392, 0.00218], dtype=np.float64)

    canonical = canonicalize_quat_xyzw_sign(quat)

    np.testing.assert_allclose(quat_xyzw_to_mat(canonical), quat_xyzw_to_mat(quat))
    assert np.linalg.norm(canonical) == pytest.approx(np.linalg.norm(quat))


def test_canonicalize_quaternion_rejects_invalid_values():
    with pytest.raises(ValueError, match="at least 4"):
        canonicalize_quat_xyzw_sign(np.array([0.0, 1.0, 0.0]))

    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_quat_xyzw_sign(np.array([0.0, 1.0, np.nan, 0.0]))
