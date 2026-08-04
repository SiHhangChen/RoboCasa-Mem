"""底盘移动的梯形速度剖面规划（纯数学，不依赖 env）。

将"位置误差 -> 归一化 action"的饱和 P 控制替换为：
- 减速段按制动距离规划: v = min(v_max, sqrt(2 * a_dec * dist))
- 加速段对命令速度做斜率限幅（a_max * dt），消除速度阶跃
- 末端保底命令速度 v_min 克服执行器静摩擦（仅 dist > tolerance 时生效）

物理换算（omron 底盘 + MobileBaseJointVelocityController）：
action 1.0 -> xy 1.0 m/s；action 1.0 -> yaw 1.5 rad/s（ctrlrange ±1.5 半幅缩放）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

XY_ACTION_SPEED = 1.0
YAW_ACTION_SPEED = 1.5
DEFAULT_DT = 0.05

__all__ = [
    "XY_ACTION_SPEED",
    "YAW_ACTION_SPEED",
    "DEFAULT_DT",
    "AxialProfileParams",
    "AxialVelocityProfile",
    "PlanarVelocityProfile",
    "BaseVelocityLimits",
    "BaseVelocityPlanner",
    "resolve_control_dt",
    "planner_for_env",
]


@dataclass(frozen=True)
class AxialProfileParams:
    """1D 剖面参数（物理单位：m 或 rad）。"""

    v_max: float
    a_max: float
    a_dec: float
    v_min: float = 0.0
    action_speed: float = 1.0


class AxialVelocityProfile:
    """有状态的 1D 梯形速度剖面，记忆上一步命令速度。"""

    def __init__(self, params: AxialProfileParams, dt: float = DEFAULT_DT) -> None:
        if params.v_max <= 0.0 or params.a_max <= 0.0 or params.a_dec <= 0.0:
            raise ValueError("v_max, a_max, a_dec must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.params = params
        self.dt = float(dt)
        self._v_prev = 0.0
        self._sign_prev = 0.0

    def reset(self, v_prev: float = 0.0) -> None:
        self._v_prev = float(abs(v_prev))
        self._sign_prev = 0.0

    @property
    def last_speed(self) -> float:
        return self._v_prev

    def step(self, dist: float, *, tolerance: float = 0.0) -> float:
        """输入非负距离，返回非负命令速度（物理单位）。"""
        p = self.params
        dist = max(0.0, float(dist))
        tolerance = max(0.0, float(tolerance))
        if dist <= tolerance:
            v_target = 0.0
        else:
            # 梯形剖面减速公式 + 小距离平滑
            v_brake = math.sqrt(2.0 * p.a_dec * dist)
            # 小于 tolerance*2 时，额外降速 20% 提升末端稳定性
            if dist < tolerance * 2.0:
                smoothing = 0.8 + 0.2 * (dist / max(tolerance * 2.0, 1e-6))
                v_brake *= smoothing
            v_target = min(p.v_max, v_brake)
        v_cmd = float(
            np.clip(
                v_target,
                self._v_prev - p.a_dec * self.dt,
                self._v_prev + p.a_max * self.dt,
            )
        )
        v_cmd = max(0.0, min(v_cmd, dist / self.dt))
        if dist > tolerance and p.v_min > 0.0:
            v_cmd = max(v_cmd, min(p.v_min, dist / self.dt))
        self._v_prev = v_cmd
        return v_cmd

    def step_signed(self, error: float, *, tolerance: float = 0.0) -> float:
        """带符号误差（yaw 用）；误差符号翻转时重置速度记忆。"""
        sign = math.copysign(1.0, error) if abs(error) > 1e-12 else 0.0
        if sign != 0.0 and self._sign_prev != 0.0 and sign != self._sign_prev:
            self.reset()
        if sign != 0.0:
            self._sign_prev = sign
        return sign * self.step(abs(error), tolerance=tolerance)

    def step_action(self, dist: float, *, tolerance: float = 0.0) -> float:
        """返回 [0, 1] 内的 action 幅值。"""
        v = self.step(dist, tolerance=tolerance)
        return float(np.clip(v / self.params.action_speed, 0.0, 1.0))

    def step_action_signed(self, error: float, *, tolerance: float = 0.0) -> float:
        v = self.step_signed(error, tolerance=tolerance)
        return float(np.clip(v / self.params.action_speed, -1.0, 1.0))


class PlanarVelocityProfile:
    """2D 剖面：对误差向量范数跑 1D 剖面，输出沿单位方向的 action 向量。"""

    def __init__(self, params: AxialProfileParams, dt: float = DEFAULT_DT) -> None:
        self._axial = AxialVelocityProfile(params, dt)
        self._last_axis: int | None = None

    def reset(self) -> None:
        self._axial.reset()
        self._last_axis = None

    @property
    def last_speed(self) -> float:
        return self._axial.last_speed

    def step_action(self, err_xy: np.ndarray, *, tolerance: float = 0.0) -> np.ndarray:
        err_xy = np.array(err_xy, dtype=float)[:2]
        dist = float(np.linalg.norm(err_xy))
        action_mag = self._axial.step_action(dist, tolerance=tolerance)
        if dist <= 1e-12 or action_mag <= 0.0:
            return np.zeros(2, dtype=float)
        return err_xy / dist * action_mag

    def step_action_single_axis(
        self, err_xy: np.ndarray, *, tolerance: float = 0.0
    ) -> tuple[np.ndarray, int]:
        """single_axis_tracking：只沿误差最大的轴输出，轴切换时重置剖面。"""
        err_xy = np.array(err_xy, dtype=float)[:2]
        axis = int(np.argmax(np.abs(err_xy)))
        if self._last_axis is not None and axis != self._last_axis:
            self._axial.reset()
        self._last_axis = axis
        action = np.zeros(2, dtype=float)
        action[axis] = self._axial.step_action_signed(
            float(err_xy[axis]), tolerance=tolerance
        )
        return action, axis


@dataclass(frozen=True)
class BaseVelocityLimits:
    """xy + yaw 全套剖面参数；NavigationConfig -> 控制循环的传输载体。"""

    v_max: float = 0.65      # 配合双轴运动的默认值
    a_max: float = 1.6
    a_dec: float = 1.2
    v_min: float = 0.22
    w_max: float = 1.2
    alpha_max: float = 3.0
    alpha_dec: float = 2.5
    w_min: float = 0.20
    dock_v_max: float = 0.50
    dock_v_min: float = 0.28
    enabled: bool = True

    @classmethod
    def from_navigation_config(cls, nav_config: Any) -> "BaseVelocityLimits":
        if nav_config is None:
            return cls()
        return cls(
            enabled=bool(getattr(nav_config, "base_velocity_profile_enabled", True)),
            v_max=float(getattr(nav_config, "base_profile_v_max", cls.v_max)),
            a_max=float(getattr(nav_config, "base_profile_a_max", cls.a_max)),
            a_dec=float(getattr(nav_config, "base_profile_a_dec", cls.a_dec)),
            v_min=float(getattr(nav_config, "base_profile_v_min", cls.v_min)),
            w_max=float(getattr(nav_config, "base_profile_w_max", cls.w_max)),
            alpha_max=float(getattr(nav_config, "base_profile_alpha_max", cls.alpha_max)),
            alpha_dec=float(getattr(nav_config, "base_profile_alpha_dec", cls.alpha_dec)),
            w_min=float(getattr(nav_config, "base_profile_w_min", cls.w_min)),
            dock_v_max=float(getattr(nav_config, "base_profile_dock_v_max", cls.dock_v_max)),
            dock_v_min=float(getattr(nav_config, "base_profile_dock_v_min", cls.dock_v_min)),
        )

    def xy_params(self) -> AxialProfileParams:
        return AxialProfileParams(
            v_max=self.v_max,
            a_max=self.a_max,
            a_dec=self.a_dec,
            v_min=self.v_min,
            action_speed=XY_ACTION_SPEED,
        )

    def yaw_params(self) -> AxialProfileParams:
        return AxialProfileParams(
            v_max=self.w_max,
            a_max=self.alpha_max,
            a_dec=self.alpha_dec,
            v_min=self.w_min,
            action_speed=YAW_ACTION_SPEED,
        )

    def dock_params(self) -> AxialProfileParams:
        return AxialProfileParams(
            v_max=self.dock_v_max,
            a_max=self.a_max,
            a_dec=self.a_dec,
            v_min=self.dock_v_min,
            action_speed=XY_ACTION_SPEED,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "v_max": float(self.v_max),
            "a_max": float(self.a_max),
            "a_dec": float(self.a_dec),
            "v_min": float(self.v_min),
            "w_max": float(self.w_max),
            "alpha_max": float(self.alpha_max),
            "alpha_dec": float(self.alpha_dec),
            "w_min": float(self.w_min),
            "dock_v_max": float(self.dock_v_max),
            "dock_v_min": float(self.dock_v_min),
        }


class BaseVelocityPlanner:
    """一个控制循环持有一个实例；phase 字符串变化时自动 reset 对应剖面。"""

    def __init__(self, limits: BaseVelocityLimits, dt: float = DEFAULT_DT) -> None:
        self.limits = limits
        self.dt = float(dt)
        self._xy = PlanarVelocityProfile(limits.xy_params(), dt)
        self._yaw = AxialVelocityProfile(limits.yaw_params(), dt)
        self._dock = AxialVelocityProfile(limits.dock_params(), dt)
        self._xy_phase: str | None = None
        self._yaw_phase: str | None = None
        self._dock_phase: str | None = None

    def begin(self) -> None:
        self._xy.reset()
        self._yaw.reset()
        self._dock.reset()
        self._xy_phase = None
        self._yaw_phase = None
        self._dock_phase = None

    def xy_action(
        self,
        err_xy: np.ndarray,
        *,
        tolerance: float,
        phase: str,
        single_axis: bool = False,
    ) -> np.ndarray:
        if phase != self._xy_phase:
            self._xy.reset()
            self._xy_phase = phase
        if single_axis:
            action, _ = self._xy.step_action_single_axis(err_xy, tolerance=tolerance)
            return action
        return self._xy.step_action(err_xy, tolerance=tolerance)

    def yaw_action(self, yaw_err: float, *, tolerance: float, phase: str) -> float:
        if phase != self._yaw_phase:
            self._yaw.reset()
            self._yaw_phase = phase
        return self._yaw.step_action_signed(float(yaw_err), tolerance=tolerance)

    def dock_axis_action(
        self, signed_err: float, *, tolerance: float, phase: str
    ) -> float:
        if phase != self._dock_phase:
            self._dock.reset()
            self._dock_phase = phase
        return self._dock.step_action_signed(float(signed_err), tolerance=tolerance)

    def last_speeds(self) -> dict[str, float]:
        return {
            "v_cmd_xy": float(self._xy.last_speed),
            "v_cmd_yaw": float(self._yaw.last_speed),
            "v_cmd_dock": float(self._dock.last_speed),
        }


def resolve_control_dt(env: Any) -> float:
    dt = getattr(env, "control_timestep", None)
    if dt:
        return float(dt)
    freq = getattr(env, "control_freq", None)
    if freq:
        return 1.0 / float(freq)
    return DEFAULT_DT


def planner_for_env(
    env: Any, limits: BaseVelocityLimits | None
) -> BaseVelocityPlanner | None:
    if limits is None or not limits.enabled:
        return None
    return BaseVelocityPlanner(limits, dt=resolve_control_dt(env))
