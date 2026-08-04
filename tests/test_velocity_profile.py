"""velocity_profile 纯数学单元测试（理想质点 x += v*dt 闭环仿真）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bench.execution.navigation.velocity_profile import (
    DEFAULT_DT,
    XY_ACTION_SPEED,
    YAW_ACTION_SPEED,
    AxialProfileParams,
    AxialVelocityProfile,
    BaseVelocityLimits,
    BaseVelocityPlanner,
    PlanarVelocityProfile,
)

DT = DEFAULT_DT
EPS = 1e-9


def default_params(**overrides) -> AxialProfileParams:
    base = dict(v_max=1.0, a_max=2.5, a_dec=1.5, v_min=0.30, action_speed=1.0)
    base.update(overrides)
    return AxialProfileParams(**base)


def simulate_to_target(
    profile: AxialVelocityProfile,
    dist0: float,
    *,
    tolerance: float,
    max_steps: int = 2000,
) -> tuple[list[float], list[float]]:
    """返回 (每步剩余距离序列, 每步命令速度序列)。"""
    dist = float(dist0)
    dists, vels = [], []
    for _ in range(max_steps):
        if dist <= tolerance:
            break
        v = profile.step(dist, tolerance=tolerance)
        dist -= v * DT
        dists.append(dist)
        vels.append(v)
    return dists, vels


class TestAccelerationLimit:
    def test_ramp_and_plateau(self):
        p = default_params(v_min=0.0)
        profile = AxialVelocityProfile(p, DT)
        profile.reset()
        vels = [profile.step(5.0) for _ in range(50)]
        for prev, cur in zip([0.0] + vels[:-1], vels):
            assert cur - prev <= p.a_max * DT + EPS
        # 平台段恒为 v_max
        assert all(abs(v - p.v_max) < EPS for v in vels[20:])


class TestNoOvershoot:
    @pytest.mark.parametrize("dist0", [0.05, 0.3, 1.0, 3.0])
    def test_converges_without_crossing(self, dist0):
        tol = 0.06
        profile = AxialVelocityProfile(default_params(), DT)
        profile.reset()
        dists, _ = simulate_to_target(profile, dist0, tolerance=tol)
        if dist0 <= tol:
            assert dists == []  # 初始即在容差内，不应产生运动
            return
        # 全程不穿越目标（剩余距离不为负超过容差）
        assert min(dists) >= -tol
        assert dists[-1] <= tol


class TestBrakingCurve:
    def test_decel_respects_stopping_distance(self):
        p = default_params(v_min=0.0)
        profile = AxialVelocityProfile(p, DT)
        profile.reset()
        dist = 2.0
        for _ in range(500):
            if dist <= 1e-3:
                break
            v = profile.step(dist)
            # 任意时刻速度不超过该距离能刹住的速度（加一步余量）
            assert v * v <= 2.0 * p.a_dec * dist + p.a_max * DT * p.v_max + 1e-6
            dist -= v * DT


class TestMinVelocityFloor:
    def test_floor_active_outside_tolerance(self):
        p = default_params()
        profile = AxialVelocityProfile(p, DT)
        profile.reset()
        tol = 0.06
        dist = tol * 1.5
        v = profile.step(dist, tolerance=tol)
        assert v >= min(p.v_min, dist / DT) - EPS

    def test_zero_inside_tolerance(self):
        profile = AxialVelocityProfile(default_params(), DT)
        profile.reset(v_prev=0.0)
        assert profile.step(0.01, tolerance=0.06) == 0.0


class TestSingleStepClamp:
    def test_small_distance_clamped(self):
        profile = AxialVelocityProfile(default_params(v_min=0.0), DT)
        profile.reset(v_prev=1.0)
        v = profile.step(0.01)
        assert v <= 0.01 / DT + EPS  # <= 0.2


class TestSignedYaw:
    def test_symmetry(self):
        p = default_params(v_max=1.5, a_max=4.0, a_dec=3.0, v_min=0.25, action_speed=YAW_ACTION_SPEED)
        pos = AxialVelocityProfile(p, DT)
        neg = AxialVelocityProfile(p, DT)
        pos.reset()
        neg.reset()
        for e in [1.0, 0.8, 0.5, 0.2]:
            assert pos.step_signed(e) == pytest.approx(-neg.step_signed(-e))

    def test_sign_flip_resets(self):
        p = default_params(v_min=0.0)
        profile = AxialVelocityProfile(p, DT)
        profile.reset()
        for _ in range(20):
            profile.step_signed(2.0)
        v_after_flip = profile.step_signed(-2.0)
        assert abs(v_after_flip) <= p.a_max * DT + EPS


class TestPhaseReset:
    def test_planner_phase_switch_restarts_ramp(self):
        limits = BaseVelocityLimits()
        planner = BaseVelocityPlanner(limits, DT)
        planner.begin()
        for _ in range(20):
            planner.xy_action(np.array([3.0, 0.0]), tolerance=0.06, phase="a")
        first = planner.xy_action(np.array([3.0, 0.0]), tolerance=0.06, phase="b")
        assert float(np.linalg.norm(first)) <= (limits.a_max * DT + limits.v_min) / XY_ACTION_SPEED + EPS

    def test_single_axis_switch_resets(self):
        profile = PlanarVelocityProfile(default_params(v_min=0.0), DT)
        profile.reset()
        for _ in range(20):
            profile.step_action_single_axis(np.array([3.0, 0.1]))
        action, axis = profile.step_action_single_axis(np.array([0.1, 3.0]))
        assert axis == 1
        assert abs(action[1]) <= default_params().a_max * DT + EPS
        assert action[0] == 0.0


class TestConvergenceSteps:
    def test_one_meter_regression(self):
        tol = 0.06
        profile = AxialVelocityProfile(default_params(), DT)
        profile.reset()
        _, vels = simulate_to_target(profile, 1.0, tolerance=tol)
        p = default_params()
        analytic_t = 1.0 / p.v_max + p.v_max / (2 * p.a_max) + p.v_max / (2 * p.a_dec)
        analytic_steps = analytic_t / DT
        assert len(vels) <= analytic_steps * 1.3

    def test_beats_legacy_p_controller(self):
        tol = 0.06
        gain = 3.0
        # legacy: v = clip(dist * gain, 0, 1)
        dist = 1.0
        legacy_steps = 0
        for _ in range(2000):
            if dist <= tol:
                break
            v = min(dist * gain, 1.0)
            dist -= v * DT
            legacy_steps += 1
        profile = AxialVelocityProfile(default_params(), DT)
        profile.reset()
        _, vels = simulate_to_target(profile, 1.0, tolerance=tol)
        assert len(vels) <= legacy_steps


class TestActionScaling:
    def test_xy_scale(self):
        profile = AxialVelocityProfile(default_params(action_speed=XY_ACTION_SPEED), DT)
        profile.reset(v_prev=0.5)
        # v_target≈sqrt(2*1.5*0.1)≈0.548 但被斜率限幅在 0.5±a*dt 内
        a = profile.step_action(0.1)
        assert 0.0 <= a <= 1.0

    def test_yaw_scale(self):
        p = default_params(v_max=1.5, action_speed=YAW_ACTION_SPEED, v_min=0.0)
        profile = AxialVelocityProfile(p, DT)
        profile.reset(v_prev=1.5)
        a = profile.step_action(10.0)
        assert a == pytest.approx(1.0)

    def test_action_in_unit_range(self):
        profile = AxialVelocityProfile(default_params(), DT)
        profile.reset()
        for d in [0.01, 0.1, 1.0, 10.0]:
            assert 0.0 <= profile.step_action(d) <= 1.0


class TestLimitsRoundTrip:
    def test_from_navigation_config_defaults(self):
        class FakeConfig:
            base_velocity_profile_enabled = True
            base_profile_v_max = 0.8
            base_profile_a_max = 2.0
            base_profile_a_dec = 1.2
            base_profile_v_min = 0.28
            base_profile_w_max = 1.2
            base_profile_alpha_max = 3.5
            base_profile_alpha_dec = 2.5
            base_profile_w_min = 0.2
            base_profile_dock_v_max = 0.6
            base_profile_dock_v_min = 0.3

        limits = BaseVelocityLimits.from_navigation_config(FakeConfig())
        assert limits.v_max == 0.8
        assert limits.w_min == 0.2
        assert limits.dock_v_min == 0.3
        assert limits.enabled is True

    def test_none_config_gives_defaults(self):
        limits = BaseVelocityLimits.from_navigation_config(None)
        assert limits == BaseVelocityLimits()

    def test_params_action_speeds(self):
        limits = BaseVelocityLimits()
        assert limits.xy_params().action_speed == XY_ACTION_SPEED
        assert limits.yaw_params().action_speed == YAW_ACTION_SPEED
        assert limits.dock_params().v_max == limits.dock_v_max
        assert limits.dock_params().v_min == limits.dock_v_min
