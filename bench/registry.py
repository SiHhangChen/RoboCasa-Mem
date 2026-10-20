"""Lazy task-plugin descriptors used by the canonical task catalog."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


_TASK_PLUGINS: dict[str, "TaskPlugin"] = {}
_TASK_PLUGIN_ALIASES: dict[str, str] = {}


def _normalize_task_plugin_name(value: str) -> str:
    """Return a punctuation- and case-insensitive task lookup key."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("task plugin name must be a non-empty string")
    normalized = "".join(character for character in value.casefold() if character.isalnum())
    if not normalized:
        raise ValueError("task plugin name must contain at least one letter or digit")
    return normalized


def _load_symbol(reference: str) -> Any:
    """Load ``module:qualified.name`` without importing it at registration time."""
    module_name, separator, qualified_name = reference.partition(":")
    if not separator or not module_name or not qualified_name:
        raise ValueError(
            "lazy symbol references must use the 'module:qualified.name' format"
        )
    value: Any = importlib.import_module(module_name)
    for attribute in qualified_name.split("."):
        value = getattr(value, attribute)
    return value


@dataclass(frozen=True, slots=True)
class TaskPlugin:
    """Lazy descriptor for one benchmark task.

    The descriptor contains only stable metadata and import paths.  Constructing
    or registering it therefore does not import recording or simulator modules.
    """

    canonical_id: str
    aliases: tuple[str, ...]
    env_name: str
    default_task_config: Path
    environment_module: str
    runtime_api_symbol: str
    expert_plan_symbol: str | None = None

    def __post_init__(self) -> None:
        canonical_id = str(self.canonical_id).strip()
        env_name = str(self.env_name).strip()
        environment_module = str(self.environment_module).strip()
        runtime_api_symbol = str(self.runtime_api_symbol).strip()
        expert_plan_symbol = (
            None if self.expert_plan_symbol is None else str(self.expert_plan_symbol).strip()
        )
        aliases = tuple(str(alias).strip() for alias in self.aliases)

        _normalize_task_plugin_name(canonical_id)
        if not env_name:
            raise ValueError("task plugin env_name must be non-empty")
        if not environment_module:
            raise ValueError("task plugin environment_module must be non-empty")
        for field_name, symbol in {
            "runtime_api_symbol": runtime_api_symbol,
        }.items():
            if not symbol or ":" not in symbol:
                raise ValueError(
                    f"{field_name} must use the 'module:qualified.name' format"
                )
        if expert_plan_symbol is not None and (
            not expert_plan_symbol or ":" not in expert_plan_symbol
        ):
            raise ValueError("expert_plan_symbol must use the 'module:qualified.name' format")
        if any(not alias for alias in aliases):
            raise ValueError("task plugin aliases must be non-empty strings")
        for alias in aliases:
            _normalize_task_plugin_name(alias)

        object.__setattr__(self, "canonical_id", canonical_id)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "env_name", env_name)
        object.__setattr__(self, "default_task_config", Path(self.default_task_config))
        object.__setattr__(self, "environment_module", environment_module)
        object.__setattr__(self, "runtime_api_symbol", runtime_api_symbol)
        object.__setattr__(self, "expert_plan_symbol", expert_plan_symbol)

    @property
    def task_id(self) -> str:
        """Compatibility-friendly spelling for :attr:`canonical_id`."""
        return self.canonical_id

    @property
    def default_task_config_file(self) -> Path:
        """Compatibility-friendly spelling for :attr:`default_task_config`."""
        return self.default_task_config

    def load_environment_module(self) -> ModuleType:
        """Import this task's simulator environment module."""
        return importlib.import_module(self.environment_module)

    def load_runtime_api(self) -> Any:
        """Load the canonical task-local runtime API."""
        return _load_symbol(self.runtime_api_symbol)

    def load_expert_plan_factory(self) -> Any:
        """Load the task-local ExpertPlan factory."""
        if self.expert_plan_symbol is None:
            raise RuntimeError(f"task {self.canonical_id} has no expert policy in this distribution")
        return _load_symbol(self.expert_plan_symbol)


def register_task_plugin(plugin: TaskPlugin) -> TaskPlugin:
    """Register a task descriptor and all of its normalized aliases.

    Re-registering an equal descriptor is intentionally idempotent so builtin
    registration can be called safely from multiple entrypoints.
    """
    if not isinstance(plugin, TaskPlugin):
        raise TypeError("plugin must be a TaskPlugin")

    existing = _TASK_PLUGINS.get(plugin.canonical_id)
    if existing is not None:
        if existing != plugin:
            raise KeyError(f"task plugin {plugin.canonical_id!r} is already registered")
        return existing

    lookup_names = (plugin.canonical_id, *plugin.aliases)
    normalized_names = {_normalize_task_plugin_name(name) for name in lookup_names}
    for normalized in normalized_names:
        owner = _TASK_PLUGIN_ALIASES.get(normalized)
        if owner is not None and owner != plugin.canonical_id:
            raise KeyError(
                f"task plugin alias {normalized!r} is already registered for {owner!r}"
            )

    _TASK_PLUGINS[plugin.canonical_id] = plugin
    for normalized in normalized_names:
        _TASK_PLUGIN_ALIASES[normalized] = plugin.canonical_id
    return plugin


def get_task_plugin(canonical_id: str) -> TaskPlugin:
    """Look up a plugin by its exact canonical task id."""
    try:
        return _TASK_PLUGINS[canonical_id]
    except KeyError as exc:
        raise KeyError(f"task plugin {canonical_id!r} not found in registry") from exc


def resolve_task_plugin(name: str) -> TaskPlugin:
    """Resolve canonical ids and aliases independent of case and punctuation."""
    normalized = _normalize_task_plugin_name(name)
    try:
        canonical_id = _TASK_PLUGIN_ALIASES[normalized]
    except KeyError as exc:
        raise KeyError(f"task plugin {name!r} not found in registry") from exc
    return _TASK_PLUGINS[canonical_id]


def list_task_plugins() -> list[TaskPlugin]:
    """Return registered task descriptors ordered by canonical id."""
    return [_TASK_PLUGINS[name] for name in sorted(_TASK_PLUGINS)]


def list_task_plugin_ids() -> list[str]:
    """Return registered canonical task ids."""
    return sorted(_TASK_PLUGINS)


def load_task_plugin(
    name: str,
    *,
    runtime: bool = False,
    expert: bool = False,
    environment: bool = False,
    evaluation: bool = False,
) -> TaskPlugin:
    """Resolve a plugin and import only the explicitly requested components."""
    plugin = resolve_task_plugin(name)
    if runtime:
        plugin.load_runtime_api()
    if expert:
        plugin.load_expert_plan_factory()
    if environment:
        plugin.load_environment_module()
    if evaluation:
        # Evaluation always uses the shared canonical adapter.  The flag is
        # retained in the loader signature only as an explicit load request.
        importlib.import_module("bench.evaluation.canonical_adapter")
    return plugin
