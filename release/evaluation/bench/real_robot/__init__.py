"""Real-robot contracts and runtime boundaries."""

from .contracts import (
    ACTION_DIM,
    ACTION_FIELDS,
    ACTION_SCHEMA,
    CAMERA_ALIASES,
    CONTROL_FREQUENCY_HZ,
    IMAGE_SIZE,
    RUNTIME_PROFILE,
    STATE_DIM,
    STATE_FIELDS,
    STATE_SCHEMA,
    contract_metadata,
)

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
