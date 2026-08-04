"""Canonical policy observation construction for the v2 contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.execution.planning.model_alignment import align_quat_xyzw_sign
from bench.policy.protocol import CanonicalObservation, PolicyContract, validate_observation
from bench.policy.protocol.errors import ErrorCode, ProtocolError
from bench.runtime.data_contract import STATE_DIM, STATE_SCHEMA, validate_state
from bench.runtime.robot_state import RobotStateExtractor


DEFAULT_CAMERA_ALIASES = {
    "agentview_right": "robot0_agentview_right_image",
    "agentview_left": "robot0_agentview_left_image",
    "eye_in_hand": "robot0_eye_in_hand_image",
}


def build_state37(
    obs: Mapping[str, Any],
    *,
    env: Any | None = None,
    robot_state_extractor: RobotStateExtractor | None = None,
) -> np.ndarray:
    """Build the single measured state vector used by policy and recording.

    Live RoboCasa evaluation must pass ``env`` so qpos/qvel are read from the
    named MuJoCo joints.  A mapping-only fallback is intentionally limited to
    already assembled state37 fields for simulator-free contract tooling.
    """

    if env is not None:
        extractor = robot_state_extractor or RobotStateExtractor()
        return extractor.extract(env, obs)

    keys = (
        "robot0_arm_joint_qpos",
        "robot0_arm_joint_qvel",
        "robot0_gripper_qpos",
        "robot0_gripper_qvel",
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_joint_qvel",
        "robot0_torso_qpos",
        "robot0_torso_qvel",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat_site",
    )
    missing = [key for key in keys if key not in obs]
    if missing:
        raise KeyError(
            "mapping-only state37 construction requires assembled fields; "
            f"missing={missing}. Pass env= for live RoboCasa observations."
        )
    parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in keys]
    value = np.concatenate(parts)
    if value.size != STATE_DIM:
        raise ValueError(f"expected state37, got {value.shape}")
    value[21:25] = align_quat_xyzw_sign(value[21:25]).astype(np.float32)
    value[33:37] = align_quat_xyzw_sign(value[33:37]).astype(np.float32)
    return validate_state(value)


def build_canonical_observation(
    obs: Mapping[str, Any],
    *,
    task_id: str,
    prompt: str,
    contract: PolicyContract,
    camera_aliases: Mapping[str, str] = DEFAULT_CAMERA_ALIASES,
    env: Any | None = None,
    robot_state_extractor: RobotStateExtractor | None = None,
    extra_modalities: Mapping[str, Any] | None = None,
) -> CanonicalObservation:
    images: dict[str, np.ndarray] = {}
    for alias in contract.observation.required_cameras:
        source_key = camera_aliases.get(alias)
        if source_key is None:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"No runtime camera mapping for required alias {alias!r}",
            )
        if source_key not in obs:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"Missing required camera observation {source_key!r}",
            )
        image = np.asarray(obs[source_key])
        # rgbd_v3 is already OpenCV/top-left.  There is deliberately no
        # per-suite vertical-flip switch.
        images[alias] = np.ascontiguousarray(image)

    observation = CanonicalObservation(
        task_id=task_id,
        state=build_state37(
            obs,
            env=env,
            robot_state_extractor=robot_state_extractor,
        ),
        images=images,
        prompt=prompt,
        extra_modalities=dict(extra_modalities or {}),
    )
    validate_observation_against_contract(observation, contract)
    return observation


def validate_observation_against_contract(
    observation: CanonicalObservation,
    contract: PolicyContract,
) -> None:
    validate_observation(observation, contract)


def quaternion_policy_from_schema(state_schema: str) -> str:
    """Validate the only supported state schema for callers doing preflight."""

    if state_schema != STATE_SCHEMA:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"Unsupported state schema: {state_schema}",
        )
    return "episode_continuous_dominant_positive"


__all__ = [
    "DEFAULT_CAMERA_ALIASES",
    "build_canonical_observation",
    "build_state37",
    "quaternion_policy_from_schema",
    "validate_observation_against_contract",
]
