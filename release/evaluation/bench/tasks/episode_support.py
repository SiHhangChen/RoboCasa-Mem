"""Shared mechanics for simulator-free canonical episode construction.

Leaf task modules still own configuration validation, random decisions and
prompt resolution.  This module only normalizes files and identities into the
common :class:`ResolvedEpisode` contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from bench.core.identity import canonical_hash, file_sha256
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode


DEFAULT_OBSERVATION_PROFILE_FILE = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "benchmark"
    / "observation_profiles"
    / "rgbd_v3.yaml"
)


def resolve_canonical_episode(
    request: EpisodeRequest,
    *,
    task_id: str,
    task_version: str,
    default_task_config_file: str | Path,
    load_task_config: Callable[[str | Path], Any],
    resolve_scene: Callable[[Any, EpisodeRequest], Any],
    resolve_plan: Callable[[Any, EpisodeRequest], Any],
    resolve_prompt: Callable[[Any, EpisodeRequest], Any],
    resolve_task_options: Callable[[Mapping[str, Any] | None], Mapping[str, Any]] | None = None,
    task_validated_resource_names: Iterable[str] = (),
) -> ResolvedEpisode:
    """Build one immutable episode without unsupported context or simulator imports."""

    if request.task_id != task_id:
        raise ValueError(f"{task_id} resolver cannot resolve task {request.task_id!r}")
    config_path = Path(request.task_config_file or default_task_config_file).expanduser().resolve()
    profile_path = Path(
        request.observation_profile_file or DEFAULT_OBSERVATION_PROFILE_FILE
    ).expanduser().resolve()
    options = _resolve_options(request, resolve_task_options)
    normalized = EpisodeRequest(
        task_id=task_id,
        seed=request.seed,
        variant=request.variant,
        split=request.split,
        env_split=request.env_split,
        options=options,
        task_config_file=config_path,
        observation_profile_file=profile_path,
        allow_partial_calibration=request.allow_partial_calibration,
    )
    config = load_task_config(config_path)
    resources = resolve_resource_files(
        config,
        task_validated_resource_names=task_validated_resource_names,
    )
    scene = resolve_scene(config, normalized)
    plan = resolve_plan(config, normalized)
    prompt = resolve_prompt(plan, normalized)
    public_prompt = str(getattr(prompt, "policy_prompt", prompt))
    profile_id = _observation_profile_id(profile_path)
    file_hashes = {
        "task_config:0": file_sha256(config_path),
        "observation_profile": file_sha256(profile_path),
        **{
            f"resource:{name}": file_sha256(path) if path.is_file() else "missing"
            for name, path in sorted(resources.items())
        },
    }
    config_identity = canonical_hash(
        {
            "schema_version": "membench.resolved_episode_inputs/v1",
            "task_id": task_id,
            "seed": normalized.seed,
            "env_split": normalized.env_split,
            "variant": normalized.variant,
            "task_options": dict(options),
            "allow_partial_calibration": normalized.allow_partial_calibration,
            "control_frequency_hz": 20,
            "input_hashes": file_hashes,
        }
    )
    facts = _jsonable(plan)
    if not isinstance(facts, Mapping):
        facts = {"episode": facts}
    resource_hash = canonical_hash(
        {
            name: file_hashes[f"resource:{name}"]
            for name in sorted(resources)
        }
    )
    return ResolvedEpisode(
        task_id=task_id,
        task_version=task_version,
        seed=normalized.seed,
        variant=normalized.variant,
        split=normalized.split,
        env_split=normalized.env_split,
        max_horizon=int(getattr(plan, "max_horizon")),
        facts=facts,
        public={
            "prompt": public_prompt,
            "env_name": str(getattr(scene, "env_name")),
            "layout_id": getattr(scene, "layout_id", None),
            "style_id": getattr(scene, "style_id", None),
            "scene_seed": getattr(scene, "scene_seed", normalized.seed),
        },
        resources={name: str(path) for name, path in resources.items()},
        config_hash=config_identity,
        # Resource identity is content based.  A path-only hash would allow a
        # calibrated pose or scene asset to change without invalidating an
        # episode/release identity.
        resource_hash=resource_hash,
        observation_profile=profile_id,
        action_schema="membench.pandaomron_hybrid13.v2",
        task_config_file=str(config_path),
        observation_profile_file=str(profile_path),
        options=dict(options),
        allow_partial_calibration=bool(normalized.allow_partial_calibration),
    )


def resolve_resource_files(
    config: Any,
    *,
    task_validated_resource_names: Iterable[str] = (),
) -> dict[str, Path]:
    task_validated = frozenset(str(name) for name in task_validated_resource_names)
    result: dict[str, Path] = {}
    for name, value in dict(getattr(config, "resources", {}) or {}).items():
        path = config.path(value)
        if path is None:
            raise FileNotFoundError(f"missing task resource {name!r}: {path}")
        resolved = path.expanduser().resolve()
        if not resolved.is_file() and str(name) not in task_validated:
            raise FileNotFoundError(f"missing task resource {name!r}: {resolved}")
        result[str(name)] = resolved
    return result


def _resolve_options(
    request: EpisodeRequest,
    resolver: Callable[[Mapping[str, Any] | None], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if resolver is None:
        if request.options:
            raise ValueError(f"{request.task_id} task options are YAML-owned")
        return {}
    values = resolver(request.options)
    if not isinstance(values, Mapping):
        raise TypeError("resolve_task_options() must return a mapping")
    return dict(values)


def _observation_profile_id(path: Path) -> str:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"observation profile root must be a mapping: {path}")
    value = str(payload.get("profile_id", "")).strip()
    if not value:
        raise ValueError(f"observation profile must declare profile_id: {path}")
    return value


def _jsonable(value: Any) -> Any:
    # ``str``-backed enums are also instances of ``str``.  Normalize enum-like
    # values before primitive checks so episode facts contain JSON primitives,
    # not Enum instances that only happen to serialize like strings.
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
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"episode fact is not JSON serializable: {type(value).__name__}")


__all__ = ["DEFAULT_OBSERVATION_PROFILE_FILE", "resolve_canonical_episode", "resolve_resource_files"]
