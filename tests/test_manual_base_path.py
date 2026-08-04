from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import bench.execution.navigation.manual_base_path as manual_path
from bench.execution.navigation.manual_base_path import ManualBasePathExecutor, ManualBasePathSample


def _executor() -> ManualBasePathExecutor:
    executor = ManualBasePathExecutor.__new__(ManualBasePathExecutor)
    executor.env = SimpleNamespace()
    executor.xy_tolerance = 0.03
    executor.yaw_tolerance = 0.06
    executor.segment_yaw_threshold = 0.25
    executor.max_rotate_steps = 10
    executor.max_translate_steps = 10
    executor.xy_gain = 3.0
    executor.yaw_gain = 2.0
    executor.velocity_limits = None
    executor._control_dt = 0.05
    executor._zero_base_action = lambda: np.zeros(3, dtype=float)
    return executor


def test_rotate_to_sample_holds_rotation_start_xy(monkeypatch):
    executor = _executor()
    actions: list[np.ndarray] = []

    monkeypatch.setattr(manual_path, "base_yaw", lambda obs: float(obs["yaw"]))

    def fake_step(env, *, arm_qpos, base_action, base_mode, gripper_action):
        del env, arm_qpos, base_mode, gripper_action
        actions.append(np.array(base_action, dtype=float))
        return {
            "robot0_base_pos": np.array([0.0, 0.0, 0.7], dtype=float),
            "yaw": np.pi / 2,
        }

    monkeypatch.setattr(manual_path, "step_base_and_hold_arm", fake_step)
    obs = {
        "robot0_base_pos": np.array([0.0, 0.0, 0.7], dtype=float),
        "yaw": 0.0,
    }
    target = ManualBasePathSample(
        index=1,
        xy=np.array([1.0, -0.5], dtype=float),
        yaw=np.pi / 2,
    )

    _, result = executor._rotate_to_sample(
        obs,
        target=target,
        arm_qpos=np.zeros(7, dtype=float),
        gripper_action=1.0,
    )

    assert actions
    np.testing.assert_allclose(actions[0][:2], [0.0, 0.0])
    assert result["xy_drift"] == 0.0
    assert result["final_dist"] > 1.0
    assert result["pass"] is True


def test_rotate_success_depends_only_on_yaw(monkeypatch):
    executor = _executor()
    executor.rotate_xy_tolerance = 0.10

    monkeypatch.setattr(manual_path, "base_yaw", lambda obs: float(obs["yaw"]))

    def fake_step(env, *, arm_qpos, base_action, base_mode, gripper_action):
        del env, arm_qpos, base_action, base_mode, gripper_action
        return {
            "robot0_base_pos": np.array([0.42, 0.0, 0.7], dtype=float),
            "yaw": np.pi / 2,
        }

    monkeypatch.setattr(manual_path, "step_base_and_hold_arm", fake_step)
    obs = {
        "robot0_base_pos": np.array([0.0, 0.0, 0.7], dtype=float),
        "yaw": 0.0,
    }
    target = ManualBasePathSample(
        index=1,
        xy=np.array([0.0, 0.0], dtype=float),
        yaw=np.pi / 2,
    )

    _, result = executor._rotate_to_sample(
        obs,
        target=target,
        arm_qpos=np.zeros(7, dtype=float),
        gripper_action=1.0,
    )

    assert result["xy_drift"] == 0.42
    assert result["xy_motion_ok"] is False
    assert result["checks"] == {"final_yaw": True}
    assert result["pass"] is True


def test_translate_success_depends_only_on_xy(monkeypatch):
    executor = _executor()
    monkeypatch.setattr(manual_path, "base_yaw", lambda obs: float(obs["yaw"]))
    monkeypatch.setattr(manual_path, "controller_translation_yaw", lambda env: 0.0)
    obs = {
        "robot0_base_pos": np.array([1.0, 2.0, 0.7], dtype=float),
        "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        "yaw": 0.0,
    }
    target = ManualBasePathSample(
        index=1,
        xy=np.array([1.0, 2.0], dtype=float),
        yaw=np.pi,
    )

    _, result = executor._translate_to_sample(
        obs,
        target=target,
        arm_qpos=np.zeros(7, dtype=float),
        gripper_action=1.0,
        pass_tolerance=executor.xy_tolerance,
    )

    assert result["final_dist"] == 0.0
    assert result["checks"] == {"final_xy": True}
    assert result["pass"] is True


def test_rotate_then_translate_continues_after_large_rotation_xy_drift(monkeypatch):
    executor = _executor()
    translate_called = False
    monkeypatch.setattr(manual_path, "base_yaw", lambda obs: float(obs["yaw"]))

    def rotate(obs, **kwargs):
        del obs, kwargs
        return {
            "robot0_base_pos": np.array([0.42, 0.0, 0.7], dtype=float),
            "yaw": np.pi,
        }, {
            "primitive": "rotate",
            "pass": True,
            "steps": 8,
            "final_yaw_error": 0.0,
            "xy_drift": 0.42,
            "xy_motion_ok": False,
        }

    def translate(obs, **kwargs):
        nonlocal translate_called
        del obs, kwargs
        translate_called = True
        return {
            "robot0_base_pos": np.array([1.0, 0.0, 0.7], dtype=float),
            "yaw": np.pi,
        }, {"primitive": "translate", "pass": True, "steps": 5}

    executor._rotate_to_sample = rotate
    executor._translate_to_sample = translate
    obs = {
        "robot0_base_pos": np.array([0.0, 0.0, 0.7], dtype=float),
        "yaw": 0.0,
    }
    source = ManualBasePathSample(index=0, xy=np.array([0.0, 0.0]), yaw=0.0)
    target = ManualBasePathSample(index=1, xy=np.array([1.0, 0.0]), yaw=np.pi)

    _, result = executor._move_segment(
        obs,
        source=source,
        target=target,
        arm_qpos=np.zeros(7, dtype=float),
        gripper_action=1.0,
    )

    assert translate_called is True
    assert result["pre_rotate"]["xy_motion_ok"] is False
    assert result["post_translate"]["pass"] is True
    assert result["pass"] is True


def test_rotate_then_translate_stops_when_rotation_fails():
    executor = _executor()
    translate_called = False

    def failed_rotate(obs, **kwargs):
        del kwargs
        return obs, {
            "primitive": "rotate",
            "pass": False,
            "steps": 4,
            "final_yaw_error": 0.2,
            "xy_drift": 0.12,
        }

    def translate(obs, **kwargs):
        nonlocal translate_called
        del kwargs
        translate_called = True
        return obs, {"primitive": "translate", "pass": True}

    executor._rotate_to_sample = failed_rotate
    executor._translate_to_sample = translate
    obs = {
        "robot0_base_pos": np.array([0.0, 0.0, 0.7], dtype=float),
        "yaw": 0.0,
    }
    source = ManualBasePathSample(index=0, xy=np.array([0.0, 0.0]), yaw=0.0)
    target = ManualBasePathSample(index=1, xy=np.array([1.0, 0.0]), yaw=np.pi)

    _, result = executor._move_segment(
        obs,
        source=source,
        target=target,
        arm_qpos=np.zeros(7, dtype=float),
        gripper_action=1.0,
    )

    assert translate_called is False
    assert result["pass"] is False
    assert result["post_translate"]["skipped"] is True
    assert result["post_translate"]["reason"] == "pre_rotate_failed"


def test_settle_checks_full_pose_after_rotation_and_translation_corrections(monkeypatch):
    executor = _executor()
    calls: list[str] = []
    monkeypatch.setattr(manual_path, "base_yaw", lambda obs: float(obs["yaw"]))

    def translate(obs, **kwargs):
        del obs, kwargs
        calls.append("translate")
        return {
            "robot0_base_pos": np.array([1.0, 0.0, 0.7], dtype=float),
            "yaw": 0.0 if len(calls) == 1 else 1.0,
        }, {"primitive": "translate", "pass": True, "steps": 2}

    def rotate(obs, **kwargs):
        del obs, kwargs
        calls.append("rotate")
        return {
            "robot0_base_pos": np.array([0.6, 0.0, 0.7], dtype=float),
            "yaw": 1.0,
        }, {"primitive": "rotate", "pass": True, "steps": 3}

    executor._translate_to_sample = translate
    executor._rotate_to_sample = rotate
    obs = {
        "robot0_base_pos": np.array([0.0, 0.0, 0.7], dtype=float),
        "yaw": 0.0,
    }
    target = ManualBasePathSample(index=1, xy=np.array([1.0, 0.0]), yaw=1.0)

    _, result = executor._settle_at_sample(
        obs,
        target=target,
        arm_qpos=np.zeros(7, dtype=float),
        gripper_action=1.0,
    )

    assert calls == ["translate", "rotate", "translate"]
    assert result["checks"] == {
        "pre_translate": True,
        "rotate": True,
        "post_translate": True,
        "final_xy": True,
        "final_yaw": True,
    }
    assert result["steps"] == 7
    assert result["pass"] is True


def test_settle_rejects_full_pose_when_post_translation_breaks_yaw(monkeypatch):
    executor = _executor()
    translate_calls = 0
    monkeypatch.setattr(manual_path, "base_yaw", lambda obs: float(obs["yaw"]))

    def translate(obs, **kwargs):
        nonlocal translate_calls
        del obs, kwargs
        translate_calls += 1
        return {
            "robot0_base_pos": np.array([1.0, 0.0, 0.7], dtype=float),
            "yaw": 0.0,
        }, {"primitive": "translate", "pass": True, "steps": 2}

    def rotate(obs, **kwargs):
        del obs, kwargs
        return {
            "robot0_base_pos": np.array([0.6, 0.0, 0.7], dtype=float),
            "yaw": 1.0,
        }, {"primitive": "rotate", "pass": True, "steps": 3}

    executor._translate_to_sample = translate
    executor._rotate_to_sample = rotate
    obs = {
        "robot0_base_pos": np.array([0.0, 0.0, 0.7], dtype=float),
        "yaw": 0.0,
    }
    target = ManualBasePathSample(index=1, xy=np.array([1.0, 0.0]), yaw=1.0)

    _, result = executor._settle_at_sample(
        obs,
        target=target,
        arm_qpos=np.zeros(7, dtype=float),
        gripper_action=1.0,
    )

    assert translate_calls == 2
    assert result["checks"]["final_xy"] is True
    assert result["checks"]["final_yaw"] is False
    assert result["pass"] is False


def test_spatial_graph_routes_directly_between_station_vertices():
    executor = _executor()
    executor.route_mode = "spatial_graph"

    assert executor._route_indices(1, 7) == [1, 7]
    assert executor._route_indices(11, 6) == [11, 6]
    assert executor._route_indices(3, 11) == [3, 11]
    assert executor._route_indices(5, 5) == [5]
