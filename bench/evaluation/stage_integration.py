"""Task-owned integrations for canonical stage evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np

from bench.evaluation.expert_prefix import ExpertPrefixArtifact
from bench.evaluation.stage_rollout import StageSpec
from bench.runtime.data_contract import ACTION_DIM


StageSpecBuilder = Callable[
    [Mapping[str, int] | None, str | None],
    Sequence[StageSpec],
]
PrivilegedStateBuilder = Callable[[Any], Any]
StageCompletionChecker = Callable[[Any, str], bool]
ExpertStageBoundaryBuilder = Callable[
    [Any, Any, Mapping[str, int]],
    Mapping[str, int],
]


@dataclass(frozen=True, slots=True)
class StageEvaluationIntegration:
    """Describe how one task connects to the task-neutral stage engine."""

    task_id: str
    target_stage_ids: tuple[str, ...]
    capture_stage_ids: tuple[str, ...]
    prefix_stage_by_target: Mapping[str, str | None]
    completion_stage_by_target: Mapping[str, str]
    spec_builder: StageSpecBuilder
    privileged_state_builder: PrivilegedStateBuilder
    stage_completion_checker: StageCompletionChecker
    expert_stage_boundary_builder: ExpertStageBoundaryBuilder | None = None

    def __post_init__(self) -> None:
        task_id = str(self.task_id).strip()
        targets = tuple(str(value) for value in self.target_stage_ids)
        capture = tuple(str(value) for value in self.capture_stage_ids)
        prefixes = {
            str(target): None if stage is None else str(stage)
            for target, stage in self.prefix_stage_by_target.items()
        }
        completions = {
            str(target): str(stage)
            for target, stage in self.completion_stage_by_target.items()
        }
        if not task_id:
            raise ValueError("stage integration task_id must be non-empty")
        if not targets or len(targets) != len(set(targets)):
            raise ValueError("stage integration target stages must be non-empty and unique")
        if not capture or len(capture) != len(set(capture)):
            raise ValueError("stage integration capture stages must be non-empty and unique")
        if set(prefixes) != set(targets):
            raise ValueError("stage integration must define one prefix boundary per target")
        if set(completions) != set(targets):
            raise ValueError("stage integration must define one completion boundary per target")
        if prefixes[targets[0]] is not None:
            raise ValueError("the first stage target must start without an expert prefix")
        unknown_boundaries = sorted(
            {
                stage
                for stage in (*prefixes.values(), *completions.values())
                if stage is not None and stage not in capture
            }
        )
        if unknown_boundaries:
            raise ValueError(
                f"stage integration prefix boundaries are not captured: {unknown_boundaries}"
            )
        for callback_name in (
            "spec_builder",
            "privileged_state_builder",
            "stage_completion_checker",
        ):
            if not callable(getattr(self, callback_name)):
                raise TypeError(f"stage integration {callback_name} must be callable")
        if (
            self.expert_stage_boundary_builder is not None
            and not callable(self.expert_stage_boundary_builder)
        ):
            raise TypeError(
                "stage integration expert_stage_boundary_builder must be callable"
            )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "target_stage_ids", targets)
        object.__setattr__(self, "capture_stage_ids", capture)
        object.__setattr__(self, "prefix_stage_by_target", prefixes)
        object.__setattr__(self, "completion_stage_by_target", completions)

    @property
    def budget_segments(self) -> dict[str, tuple[str | None, str]]:
        return {
            target: (
                self.prefix_stage_by_target[target],
                self.completion_stage_by_target[target],
            )
            for target in self.target_stage_ids
        }

    def validate_stage_suite(self, stage_ids: Sequence[str]) -> None:
        configured = tuple(str(value) for value in stage_ids)
        if configured != self.target_stage_ids:
            raise ValueError(
                f"stage suite for {self.task_id} must declare {self.target_stage_ids}, "
                f"got {configured}"
            )

    def build_stage_specs(
        self,
        budgets: Mapping[str, int] | None,
        target_stage: str,
    ) -> tuple[StageSpec, ...]:
        target = self._target(target_stage)
        specs = tuple(self.spec_builder(budgets, target))
        if not specs:
            raise ValueError(f"stage integration returned no specs for {target!r}")
        ids = tuple(spec.id for spec in specs)
        if ids[-1] != target:
            raise ValueError(
                f"stage integration path for {target!r} must end at the target, got {ids}"
            )
        return specs

    def prefix_actions(
        self,
        artifact: ExpertPrefixArtifact,
        target_stage: str,
    ) -> np.ndarray:
        target = self._target(target_stage)
        boundary_stage = self.prefix_stage_by_target[target]
        if boundary_stage is None:
            return np.empty((0, ACTION_DIM), dtype=np.float32)
        try:
            boundary = int(artifact.stage_boundary_steps[boundary_stage])
        except KeyError as exc:
            raise ValueError(
                f"expert artifact has no boundary for required prefix stage "
                f"{boundary_stage!r}"
            ) from exc
        if boundary < 0 or boundary > int(artifact.actions.shape[0]):
            raise ValueError(
                f"invalid expert boundary {boundary_stage}={boundary} for target {target!r}"
            )
        return np.ascontiguousarray(artifact.actions[:boundary], dtype=np.float32)

    def _target(self, value: str) -> str:
        target = str(value)
        if target not in self.target_stage_ids:
            raise KeyError(
                f"unknown {self.task_id} target stage {target!r}; "
                f"expected one of {self.target_stage_ids}"
            )
        return target


_BUILTIN_INTEGRATIONS = {
    "TM02": (
        "bench.tasks.tm02.stage_evaluation:STAGE_EVALUATION",
        "bench.tasks.tm02.stage_evaluation:B1_DIAGNOSTIC_STAGE_EVALUATION",
    ),
    "TS02": ("bench.tasks.ts02.stage_evaluation:STAGE_EVALUATION",),
    "WA-01": ("bench.tasks.wa01.stage_evaluation:STAGE_EVALUATION",),
    "WA-02": ("bench.tasks.wa02.stage_evaluation:STAGE_EVALUATION",),
    "WA-03": ("bench.tasks.wa03.stage_evaluation:STAGE_EVALUATION",),
    "WA-05": ("bench.tasks.wa05.stage_evaluation:STAGE_EVALUATION",),
    "WR-01": ("bench.tasks.wr01.stage_evaluation:STAGE_EVALUATION",),
    "WR-03": ("bench.tasks.wr03.stage_evaluation:STAGE_EVALUATION",),
}


def get_stage_evaluation_integration(
    task_id: str,
    *,
    stage_ids: Sequence[str] | None = None,
) -> StageEvaluationIntegration:
    canonical = str(task_id)
    try:
        references = _BUILTIN_INTEGRATIONS[canonical]
    except KeyError as exc:
        raise KeyError(
            f"no stage evaluation integration for {canonical!r}; "
            f"available={tuple(_BUILTIN_INTEGRATIONS)}"
        ) from exc
    integrations = tuple(_load_integration(canonical, reference) for reference in references)
    if stage_ids is None:
        return integrations[0]
    requested = tuple(str(value) for value in stage_ids)
    for integration in integrations:
        if integration.target_stage_ids == requested:
            return integration
    available = tuple(integration.target_stage_ids for integration in integrations)
    raise ValueError(
        f"no stage evaluation integration for {canonical!r} stages {requested}; "
        f"available={available}"
    )


def _load_integration(
    task_id: str,
    reference: str,
) -> StageEvaluationIntegration:
    module_name, attribute = reference.split(":", 1)
    integration = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(integration, StageEvaluationIntegration):
        raise TypeError(f"invalid stage evaluation integration at {reference}")
    if integration.task_id != task_id:
        raise ValueError(
            f"stage integration {reference} declares task {integration.task_id!r}, "
            f"expected {task_id!r}"
        )
    return integration


__all__ = [
    "StageEvaluationIntegration",
    "get_stage_evaluation_integration",
]
