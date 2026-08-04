"""The task-side contracts shared by recording and evaluation.

This module contains facts and ports only.  It does not import RoboCasa or a
particular policy implementation; simulator construction lives in
``bench.runtime`` and task discovery lives in ``bench.tasks.catalog``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bench.core.identity import canonical_hash


TASK_CONTRACT_VERSION = "membench.task_contract/v1"


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Policy-visible language resolved without constructing a simulator."""

    initial_prompt: str
    policy_prompt: str
    variant: str = "memory"


@dataclass(frozen=True, slots=True)
class EpisodeRequest:
    """Inputs used to resolve one deterministic episode."""

    task_id: str
    seed: int = 0
    variant: str = "memory"
    split: str = "train"
    env_split: str = "pretrain"
    options: Mapping[str, Any] = field(default_factory=dict)
    task_config_file: Path | None = None
    observation_profile_file: Path | None = None
    # Pure resolution is simulator-free and may describe an episode before all
    # physical calibration is present. Evaluation/recording callers explicitly
    # set this to False for production gates.
    allow_partial_calibration: bool = True


@dataclass(frozen=True, slots=True)
class ResolvedEpisode:
    """Immutable, serializable facts for one episode before env creation."""

    task_id: str
    task_version: str
    seed: int
    variant: str
    split: str
    env_split: str
    max_horizon: int
    facts: Mapping[str, Any] = field(default_factory=dict)
    public: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, str] = field(default_factory=dict)
    config_hash: str = ""
    resource_hash: str = ""
    observation_profile: str = ""
    action_schema: str = ""
    task_config_file: str | None = None
    observation_profile_file: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    allow_partial_calibration: bool = True
    identity: str = ""

    def __post_init__(self) -> None:
        if not str(self.task_id).strip():
            raise ValueError("ResolvedEpisode.task_id must be non-empty")
        if not str(self.task_version).strip():
            raise ValueError("ResolvedEpisode.task_version must be non-empty")
        if int(self.max_horizon) <= 0:
            raise ValueError("ResolvedEpisode.max_horizon must be positive")
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(self, "task_version", str(self.task_version))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "max_horizon", int(self.max_horizon))
        if not self.identity:
            object.__setattr__(self, "identity", canonical_hash(self.identity_payload()))

    def identity_payload(self) -> dict[str, Any]:
        """Return only deterministic facts that define this episode."""

        return {
            "schema_version": TASK_CONTRACT_VERSION,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "seed": self.seed,
            "variant": self.variant,
            "split": self.split,
            "env_split": self.env_split,
            "max_horizon": self.max_horizon,
            "facts": dict(self.facts),
            "public": dict(self.public),
            "resources": dict(self.resources),
            "config_hash": self.config_hash,
            "resource_hash": self.resource_hash,
            "observation_profile": self.observation_profile,
            "action_schema": self.action_schema,
            "task_config_file": self.task_config_file,
            "observation_profile_file": self.observation_profile_file,
            "options": dict(self.options),
            "allow_partial_calibration": bool(self.allow_partial_calibration),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["identity"] = self.identity
        return payload


@dataclass(frozen=True, slots=True)
class ExpertPlanInputs:
    """All immutable, already-resolved inputs required by an ExpertPlan.

    Expert plans must consume the same typed values that will be used by the
    runtime.  In particular, ``action_config`` is deliberately mandatory:
    task leaves must not reopen task YAML and reconstruct a second, subtly
    different configuration path.
    """

    episode: ResolvedEpisode
    scene: Any
    episode_plan: Any
    action_config: Any
    execute: bool

    def __post_init__(self) -> None:
        if not isinstance(self.episode, ResolvedEpisode):
            raise TypeError("ExpertPlanInputs.episode must be ResolvedEpisode")
        if self.scene is None:
            raise TypeError("ExpertPlanInputs.scene must be resolved")
        if self.episode_plan is None:
            raise TypeError("ExpertPlanInputs.episode_plan must be resolved")
        if self.action_config is None:
            raise TypeError(
                "ExpertPlanInputs.action_config must be the resolved ActionConfig"
            )
        object.__setattr__(self, "execute", bool(self.execute))

    @classmethod
    def from_descriptor(cls, descriptor: Any) -> "ExpertPlanInputs":
        """Build plan inputs from one canonical ``RuntimeDescriptor``."""

        if descriptor is None:
            raise TypeError("cannot build ExpertPlanInputs from None")
        return cls(
            episode=descriptor.episode,
            scene=descriptor.scene,
            episode_plan=descriptor.episode_plan,
            action_config=descriptor.actions,
            execute=descriptor.execute,
        )


class EpisodeResolver(Protocol):
    def __call__(self, request: EpisodeRequest) -> ResolvedEpisode: ...


class ExpertPlanFactory(Protocol):
    def __call__(self, inputs: ExpertPlanInputs) -> Any: ...


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """Discoverable identity and factories for one benchmark task."""

    task_id: str
    version: str
    display_name: str
    env_name: str
    package: str
    resource_root: Path
    resolve_episode: EpisodeResolver
    build_expert_plan: ExpertPlanFactory
    success_conditions: tuple[str, ...] = ()
    default_horizon: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.task_id).strip():
            raise ValueError("task_id must be non-empty")
        if not str(self.version).strip():
            raise ValueError("task version must be non-empty")
        if int(self.default_horizon) <= 0:
            raise ValueError("default_horizon must be positive")
        if not callable(self.resolve_episode) or not callable(self.build_expert_plan):
            raise TypeError("TaskDefinition factories must be callable")
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(self, "version", str(self.version))
        object.__setattr__(self, "resource_root", Path(self.resource_root).resolve())

    @property
    def task_version(self) -> str:
        return self.version

    def resolve(self, request: EpisodeRequest | None = None, **kwargs: Any) -> ResolvedEpisode:
        if request is None:
            request = EpisodeRequest(task_id=self.task_id, **kwargs)
        normalize = lambda value: "".join(ch for ch in str(value).casefold() if ch.isalnum())
        if normalize(request.task_id) != normalize(self.task_id):
            raise ValueError(
                f"task definition {self.task_id!r} cannot resolve {request.task_id!r}"
            )
        episode = self.resolve_episode(request)
        if not isinstance(episode, ResolvedEpisode):
            raise TypeError("TaskDefinition.resolve_episode() must return ResolvedEpisode")
        return episode


__all__ = [
    "EpisodeRequest",
    "ExpertPlanInputs",
    "ExpertPlanFactory",
    "PromptSpec",
    "ResolvedEpisode",
    "TASK_CONTRACT_VERSION",
    "TaskDefinition",
]
