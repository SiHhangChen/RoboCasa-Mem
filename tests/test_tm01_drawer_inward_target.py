import numpy as np
import pytest

from bench.tasks.tm01.execution.manipulation import _drawer_inward_target


def test_inward_target_preserves_height_and_orientation():
    current = np.eye(4)
    current[:3, 3] = [0.6, 0.8, 1.1]
    base = np.eye(4)
    target = _drawer_inward_target(current, base, 0.15)
    np.testing.assert_allclose(target[:2, 3], [0.51, 0.68])
    assert target[2, 3] == current[2, 3]
    np.testing.assert_array_equal(target[:3, :3], current[:3, :3])
    np.testing.assert_array_equal(current[:3, 3], [0.6, 0.8, 1.1])


def test_inward_target_does_not_cross_base():
    current = np.eye(4)
    current[0, 3] = 0.05
    target = _drawer_inward_target(current, np.eye(4), 0.15)
    np.testing.assert_allclose(target[:2, 3], [0, 0])


@pytest.mark.parametrize("distance", [-1, float("nan"), float("inf")])
def test_inward_target_rejects_invalid_distance(distance):
    with pytest.raises(ValueError):
        _drawer_inward_target(np.eye(4), np.eye(4), distance)
