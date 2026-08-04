from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import h5py
import numpy as np

from bench.recording.writers import EpisodeRecorder
from bench.runtime.data_contract import ACTION_SCHEMA, NATIVE_EPISODE_SCHEMA, STATE_DIM


CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


class _Model:
    stat = SimpleNamespace(extent=1.0)
    vis = SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=10.0))

    def get_joint_qpos_addr(self, name):
        return {
            "mobilebase0_joint_mobile_torso_height": 21,
        }[name]

    def get_joint_qvel_addr(self, name):
        return {
            "mobilebase0_joint_mobile_forward": 21,
            "mobilebase0_joint_mobile_side": 22,
            "mobilebase0_joint_mobile_yaw": 23,
            "mobilebase0_joint_mobile_torso_height": 24,
        }[name]


class _Robot:
    _ref_arm_joint_pos_indexes = list(range(7))
    _ref_arm_joint_vel_indexes = list(range(7))
    _ref_gripper_joint_pos_indexes = {"right": [7, 8]}
    _ref_gripper_joint_vel_indexes = {"right": [7, 8]}
    robot_model = SimpleNamespace(
        base_joints=(
            "mobilebase0_joint_mobile_forward",
            "mobilebase0_joint_mobile_side",
            "mobilebase0_joint_mobile_yaw",
        ),
        torso_joints=("mobilebase0_joint_mobile_torso_height",),
    )


class _Env:
    def __init__(self) -> None:
        self.steps = 0
        self.robots = [_Robot()]
        self.sim = SimpleNamespace(
            model=_Model(),
            data=SimpleNamespace(
                qpos=np.zeros(25),
                qvel=np.zeros(25),
                time=0.0,
            ),
        )
        self._obs = self._observation()

    def _observation(self):
        result = {
            "robot0_base_pos": np.zeros(3, dtype=np.float32),
            "robot0_base_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
            "robot0_base_to_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_base_to_eef_quat_site": np.asarray([0, 0, 0, 1], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        }
        for camera in CAMERAS:
            result[f"{camera}_image"] = np.zeros((2, 2, 3), dtype=np.uint8)
            result[f"{camera}_depth"] = np.full((2, 2), 0.5, dtype=np.float32)
        return result

    def _get_observations(self, force_update=False):
        del force_update
        return self._obs

    def step(self, action):
        assert np.asarray(action).shape == (13,)
        self.steps += 1
        self.sim.data.time += 0.05
        return self._obs, 0.0, False, {}


def _patch_camera_utils(monkeypatch):
    camera_utils = types.ModuleType("robosuite.utils.camera_utils")
    camera_utils.get_camera_extrinsic_matrix = lambda sim, camera: np.eye(4)
    camera_utils.get_camera_intrinsic_matrix = (
        lambda sim, camera, camera_height, camera_width: np.eye(3)
    )
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", camera_utils)


def test_native_hdf5_uses_one_action_vector_and_v2_contract(tmp_path: Path, monkeypatch):
    _patch_camera_utils(monkeypatch)
    recorder = EpisodeRecorder(
        _Env(),
        output_dir=tmp_path,
        cameras=CAMERAS,
        camera_width=2,
        camera_height=2,
    )
    recorder.install()
    recorder.record_initial_observation(recorder.env._obs)
    recorder.env.step(np.zeros(13, dtype=np.float32))
    recorder.env.step(np.zeros(13, dtype=np.float32))
    recorder.finalize_terminal(terminated=True, truncated=False, success=True, reason="task_success")
    recorder.close()
    episode = recorder.write_episode(metadata={"task_id": "FAKE"}, planner_summary={"pass": True}, accepted=True)

    assert episode["schema_version"] == NATIVE_EPISODE_SCHEMA
    with h5py.File(tmp_path / "trajectory.h5", "r") as handle:
        demo = handle["data/demo_1"]
        assert demo["actions"].shape == (2, 13)
        assert demo["obs/robot_state"].shape == (2, STATE_DIM)
        assert np.asarray(demo["terminated"]).tolist() == [False, True]
        assert np.asarray(demo["dones"]).tolist() == [False, True]
        assert json.loads(handle["data"].attrs["action_schema_json"])["schema"] == ACTION_SCHEMA
        assert "oracle" in demo and "diagnostics" in demo


def test_native_writer_rejects_non_three_camera_contract(tmp_path: Path):
    try:
        EpisodeRecorder(
            _Env(),
            output_dir=tmp_path,
            cameras=("robot0_agentview_right",),
            camera_width=2,
            camera_height=2,
        )
    except ValueError as exc:
        assert "exactly three cameras" in str(exc)
    else:
        raise AssertionError("expected the native recorder to reject one camera")
