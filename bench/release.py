"""Canonical benchmark manifests and immutable release locks.

Release identity is derived from the same :class:`ResolvedEpisode` and
``RuntimeDescriptor`` consumed by recording and evaluation.  This module does
not construct a simulator; descriptor creation is explicitly dry-run only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from bench.core.identity import canonical_hash
from bench.runtime.contracts import RuntimeDescriptor
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode


_ROOT = Path(__file__).resolve().parents[1]
RELEASE_LOCK_SCHEMA = "membench.release_lock/v2"


def episode_public_manifest(
    episode: ResolvedEpisode,
    descriptor: RuntimeDescriptor | None = None,
) -> dict[str, Any]:
    """Return policy-safe identity and public scene information."""

    payload: dict[str, Any] = {
        "schema_version": RELEASE_LOCK_SCHEMA,
        "task_id": episode.task_id,
        "task_version": episode.task_version,
        "seed": episode.seed,
        "variant": episode.variant,
        "split": episode.split,
        "env_split": episode.env_split,
        "episode_identity": episode.identity,
        "max_horizon": episode.max_horizon,
        "config_hash": episode.config_hash,
        "resource_hash": episode.resource_hash,
        "observation_profile": episode.observation_profile,
        "action_schema": episode.action_schema,
        "public": _jsonable(dict(episode.public)),
    }
    if descriptor is not None:
        payload.update(
            {
                "control_frequency_hz": descriptor.control_frequency_hz,
                "observation": {
                    "profile_id": descriptor.sensors.profile_id,
                    "profile_hash": descriptor.sensors.content_hash,
                    "camera_aliases": list(descriptor.sensors.camera_aliases),
                    "camera_names": list(descriptor.sensors.camera_names),
                    "image_size": list(descriptor.sensors.image_size),
                    "depth": bool(descriptor.sensors.camera_depths),
                },
                "scene": {
                    "env_name": descriptor.environment.env_name,
                    "env_split": descriptor.environment.env_split,
                    "layout_id": descriptor.environment.layout_id,
                    "style_id": descriptor.environment.style_id,
                    "scene_seed": descriptor.environment.scene_seed,
                },
                "prompt": {
                    "initial": str(getattr(descriptor.prompt, "initial_prompt", descriptor.prompt)),
                    "policy": str(getattr(descriptor.prompt, "policy_prompt", descriptor.prompt)),
                },
                "initialization": _jsonable(dict(getattr(descriptor, "release_facts", {}) or {})),
            }
        )
    return payload


def episode_diagnostic_manifest(
    episode: ResolvedEpisode,
    descriptor: RuntimeDescriptor | None = None,
) -> dict[str, Any]:
    """Return the complete reproducibility manifest for release tooling."""

    payload = episode_public_manifest(episode, descriptor)
    payload.update(
        {
            "task_config_file": _portable_path(episode.task_config_file),
            "observation_profile_file": _portable_path(episode.observation_profile_file),
            "resources": {
                name: _portable_path(path)
                for name, path in sorted(episode.resources.items())
            },
            "facts": _jsonable(dict(episode.facts)),
            "options": _jsonable(dict(episode.options)),
            "allow_partial_calibration": bool(episode.allow_partial_calibration),
            "release_facts": _jsonable(
                dict(descriptor.release_facts) if descriptor is not None else {}
            ),
        }
    )
    return payload


def resolve_release_entry(
    task_id: str,
    *,
    seed: int = 0,
    variant: str = "memory",
    split: str = "evaluation",
    env_split: str = "pretrain",
    options: Mapping[str, Any] | None = None,
) -> tuple[ResolvedEpisode, RuntimeDescriptor]:
    """Resolve one release entry and its dry-run runtime descriptor."""

    episode = TaskCatalog.resolve(
        EpisodeRequest(
            task_id=task_id,
            seed=int(seed),
            variant=str(variant),
            split=str(split),
            env_split=str(env_split),
            options=dict(options or {}),
            allow_partial_calibration=False,
        )
    )
    api = resolve_builtin_task_plugin(episode.task_id).load_runtime_api()
    descriptor = api.build_descriptor(episode, False)
    if not isinstance(descriptor, RuntimeDescriptor):
        raise TypeError("canonical runtime API returned an invalid descriptor")
    return episode, descriptor


def build_release_lock(
    entries: Iterable[tuple[ResolvedEpisode, RuntimeDescriptor]],
    *,
    release_id: str,
) -> dict[str, Any]:
    """Build a deterministic, tamper-evident release lock."""

    items = [episode_diagnostic_manifest(episode, descriptor) for episode, descriptor in entries]
    items.sort(key=lambda item: (str(item["task_id"]), int(item["seed"])))
    payload: dict[str, Any] = {
        "schema_version": RELEASE_LOCK_SCHEMA,
        "release_id": str(release_id),
        "entries": items,
    }
    payload["lock_hash"] = canonical_hash(payload)
    return payload


def write_release_lock(
    path: str | Path,
    entries: Iterable[tuple[ResolvedEpisode, RuntimeDescriptor]],
    *,
    release_id: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_release_lock(entries, release_id=release_id), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return target


def verify_release_lock(payload: Mapping[str, Any]) -> None:
    value = dict(payload)
    expected = value.pop("lock_hash", None)
    if not isinstance(expected, str) or not expected:
        raise ValueError("release lock is missing lock_hash")
    actual = canonical_hash(value)
    if actual != expected:
        raise ValueError(f"release lock hash mismatch: expected={expected}, actual={actual}")
    if value.get("schema_version") != RELEASE_LOCK_SCHEMA:
        raise ValueError(
            f"unsupported release lock schema: {value.get('schema_version')!r}"
        )
    if not isinstance(value.get("entries"), list):
        raise ValueError("release lock entries must be a list")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return repr(value)


def _portable_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(_ROOT).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "RELEASE_LOCK_SCHEMA",
    "build_release_lock",
    "episode_diagnostic_manifest",
    "episode_public_manifest",
    "resolve_release_entry",
    "verify_release_lock",
    "write_release_lock",
]
