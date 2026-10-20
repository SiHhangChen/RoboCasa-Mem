"""Task-owned evaluation adapter interface and registry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from bench.evaluation.suite import EvalSuiteConfig
from bench.policy.protocol import CanonicalObservation, PolicyContract


@dataclass
class EpisodeRuntime:
    episode_id: str
    seed: int
    env: Any
    observation: Mapping[str, Any]
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    result_metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationAdapter(Protocol):
    task_id: str
    control_frequency_hz: float

    def create_episode(
        self,
        *,
        seed: int,
        policy_contract: PolicyContract,
        suite: EvalSuiteConfig,
    ) -> EpisodeRuntime: ...

    def build_observation(self, runtime: EpisodeRuntime) -> CanonicalObservation: ...

    def check_success(
        self,
        runtime: EpisodeRuntime,
        info: Mapping[str, Any],
    ) -> bool: ...

    def diagnostics(self, runtime: EpisodeRuntime) -> Mapping[str, Any]: ...

    @property
    def report_stage_profile_id(self) -> str | None: ...

    def report_stages(
        self,
        runtime: EpisodeRuntime,
    ) -> Mapping[str, Any] | None: ...

    def create_subtask_scheduler(
        self,
        runtime: EpisodeRuntime,
        suite: EvalSuiteConfig,
    ) -> Any | None: ...

    def close(self, runtime: EpisodeRuntime) -> None: ...


CORE_MODALITIES = frozenset({"rgb", "state", "prompt"})


_ADAPTERS: dict[str, type[EvaluationAdapter]] = {}


def register_adapter(task_id: str, adapter_type: type[EvaluationAdapter]) -> None:
    if not task_id:
        raise ValueError("task_id must be non-empty")
    existing = _ADAPTERS.get(task_id)
    if existing is not None and existing is not adapter_type:
        raise ValueError(f"Evaluation adapter already registered for {task_id}")
    _ADAPTERS[task_id] = adapter_type


def get_adapter(task_id: str) -> EvaluationAdapter:
    adapter_type = _ADAPTERS.get(task_id)
    if adapter_type is not None:
        return adapter_type()

    from bench.tasks import (
        get_builtin_task_plugin,
        list_builtin_task_plugins,
        register_builtin_task_plugins,
    )

    register_builtin_task_plugins()
    try:
        plugin = get_builtin_task_plugin(task_id)
    except KeyError as exc:
        available = [item.canonical_id for item in list_builtin_task_plugins()]
        raise KeyError(
            f"No evaluation task registered for {task_id!r}; available={available}"
        ) from exc
    from bench.evaluation.canonical_adapter import CanonicalEvaluationAdapter

    return CanonicalEvaluationAdapter(plugin.canonical_id)


def preflight_adapter(
    adapter: EvaluationAdapter,
    *,
    suite: EvalSuiteConfig,
    policy_contract: PolicyContract,
) -> dict[str, Any]:
    """Validate every task-policy boundary that does not require an environment."""

    if adapter.task_id != suite.task_id:
        raise ValueError(
            f"Evaluation adapter task {adapter.task_id!r} does not match "
            f"suite task {suite.task_id!r}"
        )

    # This performs exact schema selection.  There is one supported state
    # representation in the v2 runtime; no raw/canonical compatibility mode is
    # negotiated here.
    from bench.evaluation.observation import quaternion_policy_from_schema

    quaternion_policy_from_schema(policy_contract.observation.state_schema)

    modalities = set(policy_contract.observation.required_modalities)
    missing_core = CORE_MODALITIES - modalities
    unsupported = modalities - CORE_MODALITIES
    if missing_core:
        raise ValueError(
            "Policy contract is missing core modalities: "
            + ", ".join(sorted(missing_core))
        )
    if unsupported:
        raise ValueError(
            "Policy requests modalities not implemented by the v2 rollout: "
            + ", ".join(sorted(unsupported))
        )

    policy_cameras = policy_contract.observation.required_cameras
    if len(set(policy_cameras)) != len(policy_cameras):
        raise ValueError("Policy required_cameras must be unique")
    artifact_cameras = suite.artifacts.video_cameras
    if len(set(artifact_cameras)) != len(artifact_cameras):
        raise ValueError("Artifact video_cameras must be unique")

    expected_frequency = float(getattr(adapter, "control_frequency_hz", 20.0))
    if not np.isclose(
        policy_contract.control_frequency_hz,
        expected_frequency,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            f"Task {suite.task_id} requires control_frequency_hz={expected_frequency:g}, "
            f"policy declares {policy_contract.control_frequency_hz:g}"
        )

    validate_suite = getattr(adapter, "validate_suite", None)
    task_report: Mapping[str, Any] = {}
    if callable(validate_suite):
        task_report = validate_suite(suite=suite, policy_contract=policy_contract) or {}
        if not isinstance(task_report, Mapping):
            raise TypeError("adapter.validate_suite() must return a mapping or None")
    return {
        "task_id": suite.task_id,
        "state_schema": policy_contract.observation.state_schema,
        "action_schema": policy_contract.action.schema,
        "control_frequency_hz": expected_frequency,
        "policy_cameras": list(policy_cameras),
        "artifact_cameras": list(artifact_cameras),
        "modalities": sorted(modalities),
        "task": dict(task_report),
    }
