from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("mujoco")

from scripts.planning.calibrate_anchor_pose import (
    initial_marker_pose,
    load_pose_world_matrix,
)


def test_anchor_pose_loader_accepts_position_and_xyzw_quaternion(tmp_path):
    pose_path = tmp_path / "anchor_pose_pq.json"
    pose_path.write_text(
        json.dumps(
            {
                "config_kind": "anchor_pose",
                "pose": {
                    "world": {
                        "p": [1.0, 2.0, 3.0],
                        "q_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    matrix = load_pose_world_matrix(pose_path)

    assert matrix == pytest.approx(np.eye(4) + np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ))


def test_reanchored_anchor_pose_preserves_rotation_and_updates_translation(tmp_path):
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    saved_matrix = np.eye(4, dtype=float)
    saved_matrix[:3, :3] = rotation
    saved_matrix[:3, 3] = [9.0, 8.0, 7.0]
    pose_path = tmp_path / "anchor_pose.json"
    pose_path.write_text(
        json.dumps(
            {
                "config_kind": "anchor_pose",
                "anchor": {"selector": "site:door_handle"},
                "pose": {
                    "world": {"matrix": saved_matrix.tolist()},
                    "offset_from_anchor_world": [0.1, -0.2, 0.3],
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        initial_pose_file=pose_path,
        reanchor_initial_pose=True,
        anchor="site:door_handle",
        initial_orientation_grasp_file=None,
    )

    position, resolved_rotation, metadata = initial_marker_pose(
        None,
        args,
        np.array([1.0, 2.0, 3.0], dtype=float),
    )

    assert position == pytest.approx((1.1, 1.8, 3.3))
    assert resolved_rotation == pytest.approx(rotation)
    assert metadata["mode"] == "reanchored_pose_file"
    assert metadata["offset_from_anchor_world"] == pytest.approx((0.1, -0.2, 0.3))
