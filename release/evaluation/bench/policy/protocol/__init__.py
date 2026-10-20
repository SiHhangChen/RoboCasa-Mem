"""Shared wire contract for MobileMemBench policy clients and servers."""

from .contracts import (
    PROTOCOL_VERSION,
    ActionContract,
    CanonicalObservation,
    CheckpointIdentity,
    InferenceRequest,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
    validate_contract,
    validate_observation,
)
from .errors import ErrorCode, ProtocolError
from .profiles import (
    COBOT_MAGIC_RUNTIME_PROFILE,
    SIM_RUNTIME_PROFILE,
    RuntimeProfile,
    get_runtime_profile,
    runtime_profile_names,
)

__all__ = [
    "COBOT_MAGIC_RUNTIME_PROFILE",
    "PROTOCOL_VERSION",
    "SIM_RUNTIME_PROFILE",
    "ActionContract",
    "CanonicalObservation",
    "CheckpointIdentity",
    "ErrorCode",
    "InferenceRequest",
    "InferenceResponse",
    "ObservationContract",
    "PolicyContract",
    "ProtocolError",
    "RuntimeProfile",
    "get_runtime_profile",
    "runtime_profile_names",
    "validate_contract",
    "validate_observation",
]
