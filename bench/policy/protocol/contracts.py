"""Versioned policy protocol data models and validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ErrorCode, ProtocolError
from bench.runtime.data_contract import (
    ACTION_DIM,
    ACTION_SCHEMA,
    POLICY_PROTOCOL_VERSION,
    STATE_DIM,
    STATE_SCHEMA,
)

REQUIRED_CAMERA_ALIASES = ("agentview_left", "agentview_right", "eye_in_hand")


PROTOCOL_VERSION = POLICY_PROTOCOL_VERSION
STATE_SCHEMA_PREFIX = STATE_SCHEMA
STATE_SCHEMAS = frozenset({STATE_SCHEMA})
CORE_OBSERVATION_MODALITIES = frozenset({"rgb", "state", "prompt"})
OBSERVATION_RESERVED_FIELDS = frozenset({"task_id", "state", "images", "prompt"})


@dataclass(frozen=True)
class CheckpointIdentity:
    uri: str
    fingerprint: str
    step: int | None = None

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "CheckpointIdentity":
        return cls(
            uri=_required_string(payload, "uri", fallback_key="path"),
            fingerprint=_required_string(payload, "fingerprint"),
            step=_optional_int(payload.get("step")),
        )

    def to_wire(self) -> dict[str, Any]:
        return {"uri": self.uri, "step": self.step, "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class ObservationContract:
    state_schema: str
    state_dim: int
    required_cameras: tuple[str, ...]
    image_size: tuple[int, int]
    required_modalities: tuple[str, ...] = ("rgb", "state", "prompt")

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ObservationContract":
        image_size = _int_tuple(payload.get("image_size"), "image_size", length=2)
        return cls(
            state_schema=_required_string(payload, "state_schema"),
            state_dim=_required_int(payload, "state_dim"),
            required_cameras=_string_tuple(payload.get("required_cameras"), "required_cameras"),
            image_size=image_size,
            required_modalities=_string_tuple(
                payload.get("required_modalities", ("rgb", "state", "prompt")),
                "required_modalities",
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "state_schema": self.state_schema,
            "state_dim": self.state_dim,
            "required_cameras": list(self.required_cameras),
            "image_size": list(self.image_size),
            "required_modalities": list(self.required_modalities),
        }


@dataclass(frozen=True)
class ActionContract:
    schema: str
    dim: int
    horizon: int

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "ActionContract":
        return cls(
            schema=_required_string(payload, "schema"),
            dim=_required_int(payload, "dim"),
            horizon=_required_int(payload, "horizon"),
        )

    def to_wire(self) -> dict[str, Any]:
        return {"schema": self.schema, "dim": self.dim, "horizon": self.horizon}


@dataclass(frozen=True)
class PolicyContract:
    server_id: str
    policy: Mapping[str, Any]
    checkpoint: CheckpointIdentity
    observation: ObservationContract
    action: ActionContract
    control_frequency_hz: float
    code_revision: Mapping[str, str] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def from_hello(cls, payload: Mapping[str, Any]) -> "PolicyContract":
        if payload.get("op") != "hello":
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Expected server hello message")
        version = _required_string(payload, "protocol_version")
        if version != PROTOCOL_VERSION:
            raise ProtocolError(
                ErrorCode.UNSUPPORTED_PROTOCOL,
                f"Expected {PROTOCOL_VERSION}, got {version}",
            )
        contract = cls(
            server_id=_required_string(payload, "server_id"),
            policy=_required_mapping(payload, "policy"),
            checkpoint=CheckpointIdentity.from_wire(_required_mapping(payload, "checkpoint")),
            observation=ObservationContract.from_wire(
                _required_mapping(payload, "observation")
            ),
            action=ActionContract.from_wire(_required_mapping(payload, "action")),
            control_frequency_hz=_required_float(payload, "control_frequency_hz"),
            code_revision={
                str(key): str(value)
                for key, value in _required_mapping(
                    payload, "code_revision", default={}
                ).items()
            },
            protocol_version=version,
        )
        validate_contract(contract)
        return contract

    def to_hello(self) -> dict[str, Any]:
        return {
            "op": "hello",
            "protocol_version": self.protocol_version,
            "server_id": self.server_id,
            "policy": dict(self.policy),
            "checkpoint": self.checkpoint.to_wire(),
            "observation": self.observation.to_wire(),
            "action": self.action.to_wire(),
            "control_frequency_hz": self.control_frequency_hz,
            "code_revision": dict(self.code_revision),
        }


@dataclass(frozen=True)
class CanonicalObservation:
    task_id: str
    state: np.ndarray
    images: Mapping[str, np.ndarray]
    prompt: str
    extra_modalities: Mapping[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        overlap = OBSERVATION_RESERVED_FIELDS.intersection(self.extra_modalities)
        if overlap:
            raise ProtocolError(
                ErrorCode.INVALID_REQUEST,
                "extra_modalities cannot override reserved fields: "
                + ", ".join(sorted(overlap)),
            )
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "state": self.state,
            "images": dict(self.images),
            "prompt": self.prompt,
        }
        payload.update(self.extra_modalities)
        return payload


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    episode_id: str
    step: int
    observation: CanonicalObservation

    def to_wire(self) -> dict[str, Any]:
        return {
            "op": "infer",
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "step": self.step,
            "task_id": self.observation.task_id,
            "observation": self.observation.to_wire(),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "InferenceRequest":
        if payload.get("op") != "infer":
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Expected infer request")
        observation = _required_mapping(payload, "observation")
        task_id = _required_string(payload, "task_id")
        observation_task_id = str(observation.get("task_id", task_id))
        if observation_task_id != task_id:
            raise ProtocolError(
                ErrorCode.INVALID_REQUEST,
                "task_id differs between request and observation",
            )
        images = _required_mapping(observation, "images")
        state = observation.get("state")
        if not isinstance(state, np.ndarray):
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "observation.state must be ndarray")
        known = {"task_id", "state", "images", "prompt"}
        return cls(
            request_id=_required_string(payload, "request_id"),
            episode_id=_required_string(payload, "episode_id"),
            step=_required_int(payload, "step"),
            observation=CanonicalObservation(
                task_id=task_id,
                state=state,
                images={
                    str(name): _require_ndarray(value, f"images.{name}")
                    for name, value in images.items()
                },
                prompt=_required_string(observation, "prompt"),
                extra_modalities={
                    str(key): value for key, value in observation.items() if key not in known
                },
            ),
        )


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    episode_id: str
    actions: np.ndarray
    timing: Mapping[str, float] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        return {
            "op": "infer_result",
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "actions": self.actions,
            "timing": dict(self.timing),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "InferenceResponse":
        if payload.get("op") != "infer_result":
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Expected infer_result response")
        actions = _require_ndarray(payload.get("actions"), "actions")
        timing = _required_mapping(payload, "timing", default={})
        return cls(
            request_id=_required_string(payload, "request_id"),
            episode_id=_required_string(payload, "episode_id"),
            actions=actions,
            timing={str(key): float(value) for key, value in timing.items()},
        )


def validate_contract(contract: PolicyContract) -> None:
    if contract.protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(ErrorCode.UNSUPPORTED_PROTOCOL, contract.protocol_version)
    if contract.observation.state_dim != STATE_DIM:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"MobileMemBench requires state_dim={STATE_DIM}, got {contract.observation.state_dim}",
        )
    if contract.observation.state_schema not in STATE_SCHEMAS:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"Unsupported state schema: {contract.observation.state_schema}",
        )
    if contract.action.dim != ACTION_DIM or contract.action.schema != ACTION_SCHEMA:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"MobileMemBench requires {ACTION_SCHEMA} with dim={ACTION_DIM}",
        )
    if contract.action.horizon <= 0:
        raise ProtocolError(ErrorCode.CONTRACT_MISMATCH, "Action horizon must be positive")
    if any(size <= 0 for size in contract.observation.image_size):
        raise ProtocolError(ErrorCode.CONTRACT_MISMATCH, "Image size must be positive")
    if tuple(contract.observation.required_cameras) != REQUIRED_CAMERA_ALIASES:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"required_cameras must be exactly {list(REQUIRED_CAMERA_ALIASES)}",
        )
    if tuple(contract.observation.image_size) != (224, 224):
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            "MobileMemBench v2 requires image_size=[224,224]",
        )
    modalities = contract.observation.required_modalities
    if len(set(modalities)) != len(modalities):
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            "required_modalities must be unique",
        )
    missing_modalities = CORE_OBSERVATION_MODALITIES - set(modalities)
    if missing_modalities:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            "Missing core observation modalities: "
            + ", ".join(sorted(missing_modalities)),
        )
    if contract.control_frequency_hz <= 0:
        raise ProtocolError(ErrorCode.CONTRACT_MISMATCH, "Control frequency must be positive")


def validate_observation(
    observation: CanonicalObservation,
    contract: PolicyContract,
) -> None:
    overlap = OBSERVATION_RESERVED_FIELDS.intersection(observation.extra_modalities)
    if overlap:
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            "extra_modalities cannot override reserved fields: "
            + ", ".join(sorted(overlap)),
        )
    state = np.asarray(observation.state)
    if state.shape != (contract.observation.state_dim,) or state.dtype != np.float32:
        raise ProtocolError(
            ErrorCode.CONTRACT_MISMATCH,
            f"Expected state float32[{contract.observation.state_dim}], "
            f"got {state.dtype}{list(state.shape)}",
        )
    if not np.all(np.isfinite(state)):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, "State contains non-finite values")
    expected_height, expected_width = contract.observation.image_size
    for camera in contract.observation.required_cameras:
        if camera not in observation.images:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"Missing required camera {camera!r}",
            )
        image = np.asarray(observation.images[camera])
        expected_shape = (expected_height, expected_width, 3)
        if image.shape != expected_shape or image.dtype != np.uint8:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"Expected {camera} uint8{list(expected_shape)}, "
                f"got {image.dtype}{list(image.shape)}",
            )
    if not observation.prompt.strip():
        raise ProtocolError(ErrorCode.INVALID_REQUEST, "Prompt must be non-empty")
    for modality in contract.observation.required_modalities:
        if modality in CORE_OBSERVATION_MODALITIES:
            continue
        root_name = modality.split(".", 1)[0]
        if root_name not in observation.extra_modalities:
            raise ProtocolError(
                ErrorCode.CONTRACT_MISMATCH,
                f"Missing required observation modality {modality!r}",
            )


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    value = payload.get(key, default)
    if not isinstance(value, Mapping):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"{key} must be a map")
    return value


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
) -> str:
    value = payload.get(key)
    if value is None and fallback_key is not None:
        value = payload.get(fallback_key)
    if not isinstance(value, str) or not value:
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"{key} must be a non-empty string")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"{key} must be an integer")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"{key} must be a number")
    value = float(value)
    if not np.isfinite(value):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"{key} must be finite")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, "step must be an integer or null")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"{key} must be a string list")
    return tuple(value)


def _int_tuple(value: Any, key: str, *, length: int) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ProtocolError(
            ErrorCode.INVALID_REQUEST,
            f"{key} must contain {length} integers",
        )
    return tuple(value)


def _require_ndarray(value: Any, key: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ProtocolError(ErrorCode.INVALID_REQUEST, f"{key} must be ndarray")
    return value
