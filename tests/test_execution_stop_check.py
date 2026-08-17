"""Focused tests for the generic execution-layer capabilities used by WR-05/WR-06.

Covers the place-descent ``trajectory_stop_check`` (early trajectory stop) and
the navigation ``xy_tolerance`` override on direct base navigation.  These are
task-agnostic: they contain no bottle / plate / task-id semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from bench.execution.manipulation.drivers import execute_arm_trajectory
from bench.execution.navigation.controller import BaseDriverAdapter


class _FakeRightController:
    qpos_index = 0


class _FakeComposite:
    _action_split_indexes = {"right": (0, 1), "right_gripper": (1, 2)}
    part_controllers = {"right": _FakeRightController()}


class _FakeRobot:
    composite_controller = _FakeComposite()

    def create_action_vector(self, action_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [np.asarray(action_dict[key], dtype=float) for key in ("right", "right_gripper")]
        )


class _FakeSimData:
    def __init__(self) -> None:
        self.qpos = np.zeros(1, dtype=float)


class _FakeSim:
    def __init__(self) -> None:
        self.data = _FakeSimData()


class _FakeEnv:
    def __init__(self) -> None:
        self.robots = [_FakeRobot()]
        self.sim = _FakeSim()
        self.steps = 0

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.sim.data.qpos[0] = float(np.asarray(action, dtype=float)[0])
        self.steps += 1
        return self._get_observations(), 0.0, False, {}

    def _get_observations(self, force_update: bool = False) -> dict[str, Any]:
        del force_update
        return {"robot0_base_pos": np.zeros(3), "robot0_base_quat": np.asarray([0, 0, 0, 1])}


def _trajectory(n: int = 3) -> list[list[float]]:
    return [[float(i)] for i in range(1, n + 1)]


def test_execute_arm_trajectory_runs_full_trajectory_without_stop_check() -> None:
    env = _FakeEnv()
    result = execute_arm_trajectory(
        env,
        _trajectory(3),
        max_joint_step=1.0,
        settle_steps=4,
    )
    assert result["early_stop"] is None
    assert result["aborted_on_collision"] is False
    # 3 waypoints (1 substep each) + 4 settle steps
    assert result["executed_steps"] == 7


def test_execute_arm_trajectory_stops_early_and_skips_settle() -> None:
    env = _FakeEnv()
    trips = {"count": 0}

    def stop_check() -> dict[str, Any]:
        trips["count"] += 1
        return {"stop": True, "reason": "support_guard_reached"} if trips["count"] >= 2 else {"stop": False}

    result = execute_arm_trajectory(
        env,
        _trajectory(3),
        max_joint_step=1.0,
        settle_steps=8,
        trajectory_stop_check=stop_check,
    )
    assert result["early_stop"] is not None
    assert result["early_stop"]["stop"] is True
    assert result["early_stop"]["reason"] == "support_guard_reached"
    assert result["early_stop"]["phase"] in {"trajectory", "settle"}
    # Trajectory was truncated before completing all waypoints/settle steps.
    assert result["executed_steps"] < 11


def test_trajectory_stop_check_accepts_plain_bool() -> None:
    env = _FakeEnv()
    result = execute_arm_trajectory(
        env,
        _trajectory(2),
        max_joint_step=1.0,
        settle_steps=2,
        trajectory_stop_check=lambda: True,
    )
    assert result["early_stop"] is not None
    assert result["early_stop"]["stop"] is True
    assert result["executed_steps"] == 1


def test_trajectory_stop_check_error_is_recorded_not_raised() -> None:
    env = _FakeEnv()

    def boom() -> dict[str, Any]:
        raise RuntimeError("sensor failed")

    result = execute_arm_trajectory(
        env,
        _trajectory(2),
        max_joint_step=1.0,
        settle_steps=2,
        trajectory_stop_check=boom,
    )
    assert result["early_stop"] is not None
    assert result["early_stop"]["stop"] is True
    assert result["early_stop"]["reason"].startswith("trajectory_stop_check_error:")


def test_direct_navigation_xy_tolerance_override_maps_to_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _FakeEnv()
    obs = env._get_observations()
    driver = BaseDriverAdapter(env, obs)

    captured: dict[str, Any] = {}

    def fake_move_base_to_pose(
        env: Any,
        obs: dict[str, Any],
        *,
        target_xy: np.ndarray,
        target_yaw: float | None,
        gripper_action: float,
        arm_qpos: np.ndarray,
        translation_action_sign: float,
        path_monitor: Any,
        holding_object_name: str | None,
        velocity_limits: Any,
        post_step_callback: Any,
        **kwargs: Any,
    ):
        del env, target_xy, target_yaw, gripper_action, arm_qpos, translation_action_sign
        del path_monitor, holding_object_name, velocity_limits, post_step_callback
        captured.update(kwargs)
        return obs, {"pass": True, "reason": "fake"}

    monkeypatch.setattr(
        "bench.execution.navigation.base_motion.move_base_to_pose",
        fake_move_base_to_pose,
    )

    result = driver.plan_and_move(
        np.asarray([1.0, 2.0]),
        None,
        xy_tolerance=0.02,
        yaw_tolerance=0.01,
    )

    assert result.success
    assert captured["tolerance"] == 0.02
    assert captured["yaw_tolerance"] == 0.01
    assert result.details["xy_tolerance_override"] == 0.02
    assert result.details["yaw_tolerance_override"] == 0.01


def test_direct_navigation_without_overrides_keeps_nav_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _FakeEnv()
    obs = env._get_observations()
    driver = BaseDriverAdapter(env, obs)

    captured: dict[str, Any] = {}

    def fake_move_base_to_pose(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return obs, {"pass": True, "reason": "fake"}

    monkeypatch.setattr(
        "bench.execution.navigation.base_motion.move_base_to_pose",
        fake_move_base_to_pose,
    )

    result = driver.plan_and_move(np.asarray([1.0, 2.0]), None)
    assert result.success
    assert "tolerance" in captured  # nav default, not overridden
    assert "xy_tolerance_override" not in result.details
    assert "yaw_tolerance_override" not in result.details
