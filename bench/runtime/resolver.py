"""Episode and task-local runtime input resolution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from bench.runtime.sensors import load_observation_profile, validate_sensor_capabilities
from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode, TaskDefinition


def resolve_episode(request: EpisodeRequest) -> ResolvedEpisode:
    """Resolve one episode through the catalog without constructing RoboCasa."""

    if not isinstance(request, EpisodeRequest):
        raise TypeError("request must be an EpisodeRequest")
    return TaskCatalog.resolve(request)


def resolve_task_definition(task_id: str) -> TaskDefinition:
    return TaskCatalog.get(task_id)


def resolve_runtime_inputs(
    episode: ResolvedEpisode,
    *,
    load_task_config: Callable[[str | Path], Any],
    resolve_task_options: Callable[[Mapping[str, Any] | None], Mapping[str, Any]]
    | None = None,
    task_validated_resource_names: Iterable[str] = (),
) -> tuple[Any, Any, dict[str, Path], dict[str, Any]]:
    """Load the exact files frozen in ``ResolvedEpisode`` for runtime creation."""

    if episode.task_config_file is None or episode.observation_profile_file is None:
        raise ValueError("ResolvedEpisode must pin task and observation profile files")
    config_path = Path(episode.task_config_file).expanduser().resolve()
    profile_path = Path(episode.observation_profile_file).expanduser().resolve()
    config = load_task_config(config_path)
    if resolve_task_options is None:
        if episode.options:
            raise ValueError(f"{episode.task_id} task options are YAML-owned")
        options: Mapping[str, Any] = {}
    else:
        options = resolve_task_options(episode.options)
        if not isinstance(options, Mapping):
            raise TypeError("resolve_task_options() must return a mapping")
    profile = load_observation_profile(profile_path)
    validate_sensor_capabilities(profile, dict(getattr(config, "sensors", {}) or {}))
    task_validated = frozenset(str(name) for name in task_validated_resource_names)
    resources = {str(name): Path(value).expanduser().resolve() for name, value in episode.resources.items()}
    configured_names = set(dict(getattr(config, "resources", {}) or {}))
    if set(resources) != configured_names:
        raise ValueError(
            f"ResolvedEpisode resource names differ from task config: "
            f"episode={sorted(resources)}, config={sorted(configured_names)}"
        )
    for name, path in resources.items():
        configured = config.path(config.resources[name])
        if configured is None or configured.expanduser().resolve() != path:
            raise ValueError(f"ResolvedEpisode resource {name!r} does not match task config")
        if not path.is_file() and name not in task_validated:
            raise FileNotFoundError(f"missing task resource {name!r}: {path}")
    return config, profile, resources, dict(options)


def resolve_rollout_max_steps(
    requested_max_steps: int | None,
    episode: ResolvedEpisode,
) -> int:
    resolved = episode.max_horizon if requested_max_steps is None else int(requested_max_steps)
    if resolved <= 0:
        raise ValueError("rollout.max_steps must be positive")
    if resolved > episode.max_horizon:
        raise ValueError(
            f"Suite max_steps={resolved} exceeds task max_horizon={episode.max_horizon}"
        )
    return resolved


__all__ = [
    "resolve_episode",
    "resolve_rollout_max_steps",
    "resolve_runtime_inputs",
    "resolve_task_definition",
]
