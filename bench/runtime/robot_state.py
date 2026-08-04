"""Shared RoboCasa robot-state extraction for policy and recording paths."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.execution.planning.model_alignment import align_quat_xyzw_sign
from bench.runtime.data_contract import STATE_DIM, validate_state


class RobotStateExtractor:
    """Read the measured PandaOmron state in the frozen state37 order.

    The extractor owns quaternion references so each runtime/episode starts a
    fresh sign-continuity sequence.  It intentionally reads qpos/qvel from
    MuJoCo named-joint indices instead of relying on task-specific observation
    dictionaries for values that RoboCasa does not expose uniformly.
    """

    def __init__(self) -> None:
        self._references: dict[str, np.ndarray] = {}

    @property
    def quaternion_references(self) -> Mapping[str, np.ndarray]:
        return self._references

    def reset(self) -> None:
        self._references.clear()

    def extract(self, env: Any, observation: Mapping[str, Any] | None = None) -> np.ndarray:
        sim = getattr(env, "sim", None)
        robots = getattr(env, "robots", None) or ()
        if sim is None or not robots:
            raise RuntimeError("state37 extraction requires env.sim and one robot")
        robot = robots[0]
        data = getattr(sim, "data", None)
        model = getattr(sim, "model", None)
        if data is None or model is None:
            raise RuntimeError("state37 extraction requires MuJoCo sim.data and sim.model")
        obs = observation if isinstance(observation, Mapping) else {}

        arm_qpos = self._indexed(data.qpos, getattr(robot, "_ref_arm_joint_pos_indexes", None), "arm qpos", 7)
        arm_qvel = self._indexed(data.qvel, getattr(robot, "_ref_arm_joint_vel_indexes", None), "arm qvel", 7)

        gripper_pos_indexes, gripper_vel_indexes = self._right_gripper_indexes(robot)
        gripper_qpos = self._indexed(data.qpos, gripper_pos_indexes, "gripper qpos", 2)
        gripper_qvel = self._indexed(data.qvel, gripper_vel_indexes, "gripper qvel", 2)

        base_joint_names = self._joint_names(robot, "base_joints", expected=3)
        base_joint_qvel = self._named_joint_values(
            model, data.qvel, base_joint_names, "base qvel", velocity=True
        )
        torso_joint_names = self._joint_names(robot, "torso_joints", expected=1)
        torso_qpos = self._named_joint_values(
            model, data.qpos, torso_joint_names, "torso qpos", velocity=False
        )
        torso_qvel = self._named_joint_values(
            model, data.qvel, torso_joint_names, "torso qvel", velocity=True
        )

        base_pos = self._obs_vector(obs, "robot0_base_pos", 3)
        base_quat = self._aligned_quaternion(
            "base_quat_world", self._obs_vector(obs, "robot0_base_quat", 4)
        )
        eef_pos = self._obs_vector(obs, "robot0_base_to_eef_pos", 3)
        eef_quat = self._aligned_quaternion(
            "eef_quat_in_base", self._obs_vector(obs, "robot0_base_to_eef_quat_site", 4)
        )

        state = np.concatenate(
            (
                arm_qpos,
                arm_qvel,
                gripper_qpos,
                gripper_qvel,
                base_pos,
                base_quat,
                base_joint_qvel,
                torso_qpos,
                torso_qvel,
                eef_pos,
                eef_quat,
            )
        )
        if state.size != STATE_DIM:
            raise RuntimeError(f"state37 assembly produced {state.size} values")
        return validate_state(state)

    def _aligned_quaternion(self, name: str, value: np.ndarray) -> np.ndarray:
        aligned = align_quat_xyzw_sign(value, self._references.get(name))
        self._references[name] = np.asarray(aligned, dtype=np.float32).copy()
        return np.asarray(aligned, dtype=np.float32)

    @staticmethod
    def _obs_vector(obs: Mapping[str, Any], key: str, size: int) -> np.ndarray:
        if key not in obs:
            raise KeyError(f"state37 requires observation key {key!r}")
        value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must be finite shape ({size},), got {value.shape}")
        return value

    @staticmethod
    def _indexed(array: Any, indexes: Any, label: str, expected: int) -> np.ndarray:
        if indexes is None:
            raise RuntimeError(f"robot does not expose indexes for {label}")
        values = np.asarray(array[indexes], dtype=np.float64).reshape(-1)
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{label} must have {expected} finite values, got {values.shape}")
        return values

    @staticmethod
    def _joint_names(robot: Any, attribute: str, *, expected: int) -> tuple[str, ...]:
        robot_model = getattr(robot, "robot_model", None)
        names = getattr(robot_model, attribute, None)
        if names is None:
            names = getattr(robot_model, f"_{attribute}", None)
        values = tuple(str(name) for name in (names or ()))
        if len(values) != expected:
            raise RuntimeError(f"robot.{attribute} must contain {expected} joints, got {values}")
        return values

    @staticmethod
    def _named_joint_values(
        model: Any,
        array: Any,
        names: tuple[str, ...],
        label: str,
        *,
        velocity: bool,
    ) -> np.ndarray:
        values: list[float] = []
        getter = model.get_joint_qvel_addr if velocity else model.get_joint_qpos_addr
        for name in names:
            try:
                address = getter(name)
                item = np.asarray(array[address], dtype=np.float64).reshape(-1)
            except Exception as exc:
                raise RuntimeError(f"could not read {label} joint {name!r}") from exc
            if item.size != 1:
                raise RuntimeError(f"{label} joint {name!r} is not scalar")
            values.append(float(item[0]))
        result = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise RuntimeError(f"{label} contains non-finite values")
        return result

    @staticmethod
    def _right_gripper_indexes(robot: Any) -> tuple[Any, Any]:
        positions = getattr(robot, "_ref_gripper_joint_pos_indexes", {}) or {}
        velocities = getattr(robot, "_ref_gripper_joint_vel_indexes", {}) or {}
        if not isinstance(positions, Mapping) or not isinstance(velocities, Mapping):
            raise RuntimeError("robot does not expose gripper joint indexes")
        arm = "right" if "right" in positions else (next(iter(positions), None))
        if arm is None or arm not in velocities:
            raise RuntimeError("right gripper joint indexes are unavailable")
        return positions[arm], velocities[arm]


__all__ = ["RobotStateExtractor"]
