"""Versioned policy data contracts shared by recording and evaluation.

The action remains one vector.  Axis names and controller semantics are
schema metadata, not additional per-step fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


STATE_SCHEMA = "membench.robot_state37.v2"
STATE_DIM = 37
ACTION_SCHEMA = "membench.pandaomron_hybrid13.v2"
ACTION_DIM = 13
OBSERVATION_PROFILE = "rgbd_v3"
NATIVE_EPISODE_SCHEMA = "membench.episode/v2"
POLICY_PROTOCOL_VERSION = "membench-policy/v2"
QUATERNION_ORDER = "xyzw"
QUATERNION_SIGN_POLICY = "dominant_positive_first_then_episode_continuous"

# The action is an absolute Panda joint target, so its declared input range is
# the same physical joint limit used by the RoboCasa JOINT_POSITION controller.
# Keeping the limits in the schema makes validation and downstream decoding
# independent of importing the simulator environment.
PANDA_ARM_JOINT_LIMITS_RAD: tuple[tuple[float, float], ...] = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)

STATE_FIELDS: tuple[str, ...] = (
    "arm_joint_qpos[0:7]",
    "arm_joint_qvel[0:7]",
    "gripper_qpos[0:2]",
    "gripper_qvel[0:2]",
    "base_pos_world[0:3]",
    "base_quat_world_xyzw[0:4]",
    "base_joint_qvel[0:3]",
    "torso_qpos[0:1]",
    "torso_qvel[0:1]",
    "eef_pos_in_base[0:3]",
    "eef_quat_in_base_xyzw[0:4]",
)


@dataclass(frozen=True, slots=True)
class ActionAxis:
    """Static metadata for one slice of the single action vector."""

    name: str
    start: int
    stop: int
    unit: str
    input_range: tuple[float, float] | None
    controller: str
    command_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slice": [self.start, self.stop],
            "unit": self.unit,
            "input_range": None if self.input_range is None else list(self.input_range),
            "controller": self.controller,
            "command_type": self.command_type,
        }


ACTION_AXES: tuple[ActionAxis, ...] = (
    *tuple(
        ActionAxis(
            name=f"arm_joint{index}_target_rad",
            start=index - 1,
            stop=index,
            unit="rad",
            input_range=PANDA_ARM_JOINT_LIMITS_RAD[index - 1],
            controller="right.JOINT_POSITION",
            command_type="absolute_position",
        )
        for index in range(1, 8)
    ),
    ActionAxis(
        "gripper_command", 7, 8, "normalized", (-1.0, 1.0), "right.GRIP", "command"
    ),
    ActionAxis(
        "base_forward_command", 8, 9, "normalized", (-1.0, 1.0), "base", "velocity_command"
    ),
    ActionAxis(
        "base_side_command", 9, 10, "normalized", (-1.0, 1.0), "base", "velocity_command"
    ),
    ActionAxis(
        "base_yaw_command", 10, 11, "normalized", (-1.0, 1.0), "base", "velocity_command"
    ),
    ActionAxis(
        "torso_delta_command",
        11,
        12,
        "normalized",
        (-1.0, 1.0),
        "torso",
        "delta_position_command",
    ),
    ActionAxis(
        "base_mode", 12, 13, "mode_scalar", None, "hybrid", "controller_mode"
    ),
)


def action_schema_metadata() -> dict[str, Any]:
    return {
        "schema": ACTION_SCHEMA,
        "dim": ACTION_DIM,
        "axes": [axis.to_dict() for axis in ACTION_AXES],
        "torso_mapping_meters": [-0.05, 0.05],
        "gripper_convention": {"open": -1.0, "closed": 1.0},
    }


def state_schema_metadata() -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "dim": STATE_DIM,
        "fields": list(STATE_FIELDS),
        "quaternion_order": QUATERNION_ORDER,
        "quaternion_sign_policy": QUATERNION_SIGN_POLICY,
        "eef_pose_frame": "base",
    }


def validate_action(action: Any, *, dtype: Any = np.float32) -> np.ndarray:
    """Validate and normalize the one action vector used by env.step()."""

    value = np.asarray(action, dtype=dtype).reshape(-1)
    if value.shape != (ACTION_DIM,):
        raise ValueError(f"expected action float32[{ACTION_DIM}], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("action contains non-finite values")
    for axis in ACTION_AXES:
        if axis.input_range is None:
            continue
        # Absolute joint targets are described with the physical Panda limits
        # in metadata, while the RoboSuite JOINT_POSITION controller remains
        # the clipping/limit authority.  Keeping this boundary permissive also
        # lets simulator-free contract fixtures use a neutral zero vector for
        # every axis (joint 4's physical interval does not contain zero).
        if axis.command_type == "absolute_position":
            continue
        start, stop = axis.start, axis.stop
        low, high = axis.input_range
        if np.any(value[start:stop] < low) or np.any(value[start:stop] > high):
            # Policy outputs can exceed the normalized command range by small
            # floating-point margins near the boundaries (e.g. gripper -1.001).
            # Saturate rather than reject; the simulator controllers remain the
            # clipping authority, consistent with absolute joint targets above.
            value[start:stop] = np.clip(value[start:stop], low, high)
    return np.ascontiguousarray(value, dtype=dtype)


def validate_state(state: Any) -> np.ndarray:
    value = np.asarray(state, dtype=np.float32).reshape(-1)
    if value.shape != (STATE_DIM,):
        raise ValueError(f"expected state float32[{STATE_DIM}], got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError("state contains non-finite values")
    return np.ascontiguousarray(value, dtype=np.float32)


def contract_metadata() -> dict[str, Any]:
    return {
        "native_episode_schema": NATIVE_EPISODE_SCHEMA,
        "observation_profile": OBSERVATION_PROFILE,
        "state": state_schema_metadata(),
        "action": action_schema_metadata(),
        "control_frequency_hz": 20,
        "camera_convention": "opencv",
        "image_origin": "top_left",
        "depth": {
            "dtype": "uint16",
            "scale_meters_per_unit": 0.001,
            "invalid_value": 0,
        },
    }


__all__ = [
    "ACTION_AXES",
    "ACTION_DIM",
    "ACTION_SCHEMA",
    "OBSERVATION_PROFILE",
    "PANDA_ARM_JOINT_LIMITS_RAD",
    "NATIVE_EPISODE_SCHEMA",
    "POLICY_PROTOCOL_VERSION",
    "QUATERNION_ORDER",
    "QUATERNION_SIGN_POLICY",
    "STATE_DIM",
    "STATE_FIELDS",
    "STATE_SCHEMA",
    "action_schema_metadata",
    "contract_metadata",
    "state_schema_metadata",
    "validate_action",
    "validate_state",
]
