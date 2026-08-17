"""Unit tests for recorded-action replay validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from bench.recording.replay_validation import ActionReplayValidation, load_recorded_actions


def _write_trajectory(tmp_path, actions: np.ndarray) -> object:
    import h5py as _h5py

    path = tmp_path / "trajectory.h5"
    with _h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_1")
        demo.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
    return path


def test_load_recorded_actions_reads_native_stream(tmp_path) -> None:
    actions = np.zeros((3, 13), dtype=np.float32)
    path = _write_trajectory(tmp_path, actions)
    loaded = load_recorded_actions(path)
    np.testing.assert_array_equal(loaded, actions)


def test_load_recorded_actions_rejects_bad_shape(tmp_path) -> None:
    path = _write_trajectory(tmp_path, np.zeros((2, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="invalid recorded action shape"):
        load_recorded_actions(path)


def test_load_recorded_actions_rejects_non_finite(tmp_path) -> None:
    actions = np.zeros((2, 13), dtype=np.float32)
    actions[0, 3] = np.nan
    path = _write_trajectory(tmp_path, actions)
    with pytest.raises(ValueError, match="non-finite"):
        load_recorded_actions(path)


def test_replay_validation_qualified_property() -> None:
    assert ActionReplayValidation(success=True, recorded_steps=1, consumed_steps=1).qualified is True
    assert ActionReplayValidation(success=False, recorded_steps=1, consumed_steps=1).qualified is False
