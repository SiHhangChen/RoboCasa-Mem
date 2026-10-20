"""Versioned observation-profile loading and validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from bench.core.identity import file_sha256
from bench.runtime.contracts import ObservationProfile, SensorRequirements


DEFAULT_OBSERVATION_PROFILE_FILE = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmark"
    / "observation_profiles"
    / "rgbd_v3.yaml"
)


def load_observation_profile(
    path: str | Path = DEFAULT_OBSERVATION_PROFILE_FILE,
) -> ObservationProfile:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"observation profile root must be a map: {source}")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "profile_id",
            "description",
            "cameras",
            "image",
            "modalities",
            "renderer",
        },
        "observation profile",
    )
    cameras = payload.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("observation profile cameras must be a non-empty list")
    aliases: list[str] = []
    names: list[str] = []
    for index, value in enumerate(cameras):
        if not isinstance(value, Mapping):
            raise ValueError(f"cameras[{index}] must be a map")
        _reject_unknown(value, {"alias", "name"}, f"cameras[{index}]")
        aliases.append(_non_empty(value.get("alias"), f"cameras[{index}].alias"))
        names.append(_non_empty(value.get("name"), f"cameras[{index}].name"))
    if len(set(aliases)) != len(aliases):
        raise ValueError("observation profile camera aliases must be unique")
    if len(set(names)) != len(names):
        raise ValueError("observation profile physical camera names must be unique")
    image = _mapping(payload, "image")
    modalities = _mapping(payload, "modalities")
    renderer = _mapping(payload, "renderer")
    _reject_unknown(image, {"width", "height"}, "image")
    _reject_unknown(modalities, {"rgb", "depth"}, "modalities")
    _reject_unknown(
        renderer,
        {"has_renderer", "has_offscreen_renderer", "use_camera_observations"},
        "renderer",
    )
    if modalities.get("rgb") is not True:
        raise ValueError("v1 observation profiles require modalities.rgb=true")
    return ObservationProfile(
        profile_id=_non_empty(payload.get("profile_id"), "profile_id"),
        schema_version=_positive_int(payload.get("schema_version"), "schema_version"),
        camera_aliases=tuple(aliases),
        camera_names=tuple(names),
        image_size=(
            _positive_int(image.get("height"), "image.height"),
            _positive_int(image.get("width"), "image.width"),
        ),
        camera_depths=_boolean(modalities.get("depth", False), "modalities.depth"),
        has_renderer=_boolean(renderer.get("has_renderer", False), "renderer.has_renderer"),
        has_offscreen_renderer=_boolean(
            renderer.get("has_offscreen_renderer", True),
            "renderer.has_offscreen_renderer",
        ),
        use_camera_observations=_boolean(
            renderer.get("use_camera_observations", True),
            "renderer.use_camera_observations",
        ),
        source_path=source,
        content_hash=file_sha256(source),
    )


def validate_sensor_capabilities(
    profile: ObservationProfile,
    task_sensors: Mapping[str, Any],
) -> None:
    raw_cameras = task_sensors.get("cameras", ())
    if isinstance(raw_cameras, (str, bytes)):
        raise ValueError("task sensors.cameras must be a list")
    available = {str(value) for value in raw_cameras}
    wrist = task_sensors.get("wrist_rgb_camera")
    if wrist is not None:
        available.add(str(wrist))
    missing = sorted(set(profile.camera_names) - available)
    if missing:
        raise ValueError(
            f"task sensor capabilities do not provide profile "
            f"{profile.profile_id!r} cameras: {missing}"
        )


def validate_sensor_requirements(
    profile: ObservationProfile,
    requirements: SensorRequirements,
) -> None:
    unknown = sorted(
        set(requirements.required_camera_aliases) - set(profile.camera_aliases)
    )
    if unknown:
        raise ValueError(
            f"observation profile {profile.profile_id!r} has no requested aliases: "
            f"{unknown}"
        )
    if (
        requirements.required_image_size is not None
        and tuple(requirements.required_image_size) != profile.image_size
    ):
        raise ValueError(
            f"observation profile {profile.profile_id!r} "
            f"image_size={profile.image_size}, consumer requires "
            f"{tuple(requirements.required_image_size)}"
        )
    if requirements.require_depth and not profile.camera_depths:
        raise ValueError(
            f"observation profile {profile.profile_id!r} does not enable depth"
        )


def require_observation_keys(observation: Mapping[str, Any], keys: tuple[str, ...] | list[str]) -> None:
    missing = [str(key) for key in keys if str(key) not in observation]
    if missing:
        raise KeyError(f"observation is missing required keys: {missing}")


def normalize_image(image: Any, *, dtype: Any = None):
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected HxWxC image, got shape={array.shape}")
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"observation profile {key} must be a map")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(unknown)}")


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


__all__ = [
    "DEFAULT_OBSERVATION_PROFILE_FILE",
    "load_observation_profile",
    "normalize_image",
    "require_observation_keys",
    "validate_sensor_capabilities",
    "validate_sensor_requirements",
]
