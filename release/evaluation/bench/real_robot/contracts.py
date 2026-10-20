"""Frozen dataset-facing contract for the Cobot Magic WR07 platform.

The field order comes from ``data/real_robot/wr07_lerobot_v3``.  It does not
authorize sending dataset actions to hardware: command units, actuator mapping,
limits, hold behavior, and emergency-stop behavior still require validation
against the robot SDK.
"""

from __future__ import annotations

from typing import Any

RUNTIME_PROFILE = "membench.cobot_magic_wr07.v1"
STATE_SCHEMA = "membench.cobot_magic_state17.v1"
ACTION_SCHEMA = "membench.cobot_magic_jointbase17.v1"
STATE_DIM = 17
ACTION_DIM = 17
CAMERA_ALIASES = ("front", "left", "right")
IMAGE_SIZE = (224, 224)
CONTROL_FREQUENCY_HZ = 30.0

STATE_FIELDS: tuple[str, ...] = (
    "arm.jointStatePosition.puppetLeft.joint0",
    "arm.jointStatePosition.puppetLeft.joint1",
    "arm.jointStatePosition.puppetLeft.joint2",
    "arm.jointStatePosition.puppetLeft.joint3",
    "arm.jointStatePosition.puppetLeft.joint4",
    "arm.jointStatePosition.puppetLeft.joint5",
    "arm.jointStatePosition.puppetLeft.joint6",
    "arm.jointStatePosition.puppetRight.joint0",
    "arm.jointStatePosition.puppetRight.joint1",
    "arm.jointStatePosition.puppetRight.joint2",
    "arm.jointStatePosition.puppetRight.joint3",
    "arm.jointStatePosition.puppetRight.joint4",
    "arm.jointStatePosition.puppetRight.joint5",
    "arm.jointStatePosition.puppetRight.joint6",
    "robotBase.vel.chassis.linear_x",
    "robotBase.vel.chassis.linear_y",
    "robotBase.vel.chassis.angular_z",
)

ACTION_FIELDS: tuple[str, ...] = (
    "arm.jointStatePosition.masterLeft.joint0",
    "arm.jointStatePosition.masterLeft.joint1",
    "arm.jointStatePosition.masterLeft.joint2",
    "arm.jointStatePosition.masterLeft.joint3",
    "arm.jointStatePosition.masterLeft.joint4",
    "arm.jointStatePosition.masterLeft.joint5",
    "arm.jointStatePosition.masterLeft.joint6",
    "arm.jointStatePosition.masterRight.joint0",
    "arm.jointStatePosition.masterRight.joint1",
    "arm.jointStatePosition.masterRight.joint2",
    "arm.jointStatePosition.masterRight.joint3",
    "arm.jointStatePosition.masterRight.joint4",
    "arm.jointStatePosition.masterRight.joint5",
    "arm.jointStatePosition.masterRight.joint6",
    "robotBase.vel.chassis.linear_x",
    "robotBase.vel.chassis.linear_y",
    "robotBase.vel.chassis.angular_z",
)


def contract_metadata() -> dict[str, Any]:
    return {
        "runtime_profile": RUNTIME_PROFILE,
        "state_schema": STATE_SCHEMA,
        "state_dim": STATE_DIM,
        "state_fields": list(STATE_FIELDS),
        "action_schema": ACTION_SCHEMA,
        "action_dim": ACTION_DIM,
        "action_fields": list(ACTION_FIELDS),
        "camera_aliases": list(CAMERA_ALIASES),
        "image_size": list(IMAGE_SIZE),
        "control_frequency_hz": CONTROL_FREQUENCY_HZ,
        "hardware_execution_semantics": "unverified",
    }


__all__ = [
    "ACTION_DIM",
    "ACTION_FIELDS",
    "ACTION_SCHEMA",
    "CAMERA_ALIASES",
    "CONTROL_FREQUENCY_HZ",
    "IMAGE_SIZE",
    "RUNTIME_PROFILE",
    "STATE_DIM",
    "STATE_FIELDS",
    "STATE_SCHEMA",
    "contract_metadata",
]
