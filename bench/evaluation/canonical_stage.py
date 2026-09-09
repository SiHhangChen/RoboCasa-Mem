"""Bridges from the canonical evaluation runtime to stage evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np

from bench.evaluation.adapter import EpisodeRuntime, EvaluationAdapter
from bench.evaluation.client import PolicyClient
from bench.evaluation.stage_rollout import StageSpec
from bench.evaluation.suite import EvalSuiteConfig
from bench.policy.protocol import CanonicalObservation, PolicyContract
from bench.runtime.data_contract import ACTION_DIM, validate_action
from bench.tasks import TaskCatalog
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode


PrivilegedStateBuilder = Callable[[EpisodeRuntime], Any]


@dataclass(frozen=True, slots=True)
class CanonicalStageTransition:
    """Environment transition carrying the policy's canonical next observation."""

    observation: CanonicalObservation
    raw_transition: Any
    terminal: bool
    terminated: bool
    truncated: bool
    timestamp: float | None


def resolve_suite_episode(suite: EvalSuiteConfig, seed: int) -> ResolvedEpisode:
    """Resolve the same immutable episode used by ``CanonicalEvaluationAdapter``."""

    raw = dict(suite.task_options)
    task_options = raw.get("task_options", {})
    if not isinstance(task_options, Mapping):
        raise ValueError("task.options.task_options must be a map")
    return TaskCatalog.resolve(
        EpisodeRequest(
            task_id=suite.task_id,
            seed=int(seed),
            env_split=suite.env_split,
            split="evaluation",
            variant=str(raw.get("variant", "memory")),
            options=dict(task_options),
            task_config_file=_option_path(raw.get("task_config_file")),
            observation_profile_file=_option_path(raw.get("observation_profile_file")),
            allow_partial_calibration=False,
        )
    )


@dataclass(slots=True)
class CanonicalStageTask:
    """Expose one canonical task episode through the ``StageTask`` protocol."""

    task_id: str
    stage_specs: tuple[StageSpec, ...]
    adapter: EvaluationAdapter
    suite: EvalSuiteConfig
    policy_contract: PolicyContract
    privileged_state_builder: PrivilegedStateBuilder
    expected_episode_identity: str | None = None
    runtime: EpisodeRuntime | None = field(init=False, default=None)
    environment_steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.stage_specs = tuple(self.stage_specs)
        if self.adapter.task_id != self.task_id:
            raise ValueError(
                f"stage task {self.task_id!r} does not match adapter {self.adapter.task_id!r}"
            )
        if self.suite.task_id != self.task_id:
            raise ValueError(
                f"stage task {self.task_id!r} does not match suite {self.suite.task_id!r}"
            )

    @property
    def instruction(self) -> str:
        return str(self._require_runtime().prompt)

    @property
    def episode_id(self) -> str:
        return str(self._require_runtime().episode_id)

    def reset(self, seed: int) -> CanonicalObservation:
        self.close()
        self.runtime = self.adapter.create_episode(
            seed=int(seed),
            policy_contract=self.policy_contract,
            suite=self.suite,
        )
        resolved = self.runtime.metadata.get("resolved_episode")
        identity = getattr(resolved, "identity", None)
        if (
            self.expected_episode_identity is not None
            and str(identity) != str(self.expected_episode_identity)
        ):
            self.close()
            raise ValueError(
                "expert prefix episode identity does not match the canonical stage episode"
            )
        self.environment_steps = 0
        return self.adapter.build_observation(self.runtime)

    def observe(self) -> CanonicalObservation:
        return self.adapter.build_observation(self._require_runtime())

    def get_privileged_state(self) -> Any:
        return self.privileged_state_builder(self._require_runtime())

    def step(self, action: Any) -> Any:
        runtime = self._require_runtime()
        shared = runtime.metadata.get("shared_runtime")
        if shared is None or not callable(getattr(shared, "step", None)):
            raise TypeError("canonical stage task requires runtime.metadata['shared_runtime']")
        raw_transition = shared.step(validate_action(action))
        runtime.observation = raw_transition.observation
        self.environment_steps += 1
        terminated = bool(getattr(raw_transition, "terminated", False))
        truncated = bool(getattr(raw_transition, "truncated", False))
        terminal = bool(
            getattr(raw_transition, "terminal", False)
            or getattr(raw_transition, "done", False)
            or terminated
            or truncated
        )
        timestamp = getattr(raw_transition, "timestamp", None)
        if timestamp is None:
            timestamp = getattr(raw_transition, "observation_timestamp", None)
        return CanonicalStageTransition(
            observation=self.adapter.build_observation(runtime),
            raw_transition=raw_transition,
            terminal=terminal,
            terminated=terminated,
            truncated=truncated,
            timestamp=None if timestamp is None else float(timestamp),
        )

    def artifact_frames(self, cameras: tuple[str, ...]) -> Mapping[str, np.ndarray]:
        callback = getattr(self.adapter, "artifact_frames", None)
        if not callable(callback):
            raise TypeError("evaluation adapter does not expose artifact_frames(runtime, cameras)")
        return callback(self._require_runtime(), cameras)

    def close(self) -> None:
        if self.runtime is None:
            return
        runtime, self.runtime = self.runtime, None
        self.adapter.close(runtime)

    def _require_runtime(self) -> EpisodeRuntime:
        if self.runtime is None:
            raise RuntimeError("canonical stage task has not been reset")
        return self.runtime


@dataclass(slots=True)
class RemotePolicyModelAdapter:
    """Keep the existing policy session alive across an expert-prefix handoff."""

    client: PolicyClient
    task: CanonicalStageTask
    request_metrics: list[dict[str, Any]] = field(default_factory=list)
    reset_calls: int = field(init=False, default=0)

    def reset_episode(self, instruction: str) -> None:
        if str(instruction) != self.task.instruction:
            raise ValueError("stage evaluator instruction differs from canonical task prompt")
        self.client.reset(self.task.episode_id)
        self.reset_calls += 1
        self.request_metrics.clear()

    def predict(self, observation: CanonicalObservation) -> np.ndarray:
        started = time.monotonic()
        response = self.client.infer(
            observation,
            episode_id=self.task.episode_id,
            step=self.task.environment_steps,
        )
        self.request_metrics.append(
            {
                "request_id": response.request_id,
                "environment_step": self.task.environment_steps,
                "client_latency_ms": (time.monotonic() - started) * 1000.0,
                "returned_action_count": int(response.actions.shape[0]),
                "server_timing": dict(response.timing),
            }
        )
        return np.asarray(response.actions, dtype=np.float32)


class RecordedExpertController:
    """Replay one frozen low-level expert action stream."""

    def __init__(self, actions: Sequence[Any]) -> None:
        value = np.asarray(actions, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != ACTION_DIM:
            raise ValueError(
                f"recorded expert actions must have shape (N, {ACTION_DIM}), got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("recorded expert actions contain non-finite values")
        self.actions = np.ascontiguousarray(value)
        self.index = 0

    @property
    def consumed_steps(self) -> int:
        return self.index

    @property
    def remaining_steps(self) -> int:
        return int(self.actions.shape[0]) - self.index

    def predict(self, privileged_state: Any) -> np.ndarray:
        del privileged_state
        if self.index >= int(self.actions.shape[0]):
            raise RuntimeError("recorded expert prefix is exhausted")
        action = self.actions[self.index].copy()
        self.index += 1
        return action


def _option_path(value: Any) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    root = Path(__file__).resolve().parents[2]
    return (path if path.is_absolute() else root / path).resolve()


__all__ = [
    "CanonicalStageTransition",
    "CanonicalStageTask",
    "PrivilegedStateBuilder",
    "RecordedExpertController",
    "RemotePolicyModelAdapter",
    "resolve_suite_episode",
]
