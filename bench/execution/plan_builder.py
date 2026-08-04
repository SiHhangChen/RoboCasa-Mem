"""Small, task-agnostic constructors for canonical expert plans.

Leaf task packages own the operation names and stage order.  This module only
turns already-resolved descriptors into the framework contracts; it does not
know any task, object, simulator, or task-specific workflow type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bench.execution.contracts import ExpertPlan, NavigateStage, OperateStage
from bench.execution.navigation.contracts import StationTarget
from bench.tasks.contracts import ResolvedEpisode


def navigate(
    station_id: str,
    *,
    name: str = "",
    purpose: str = "",
    holding: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    skip_if_reachable: bool = False,
    fallback_if_unreachable: bool = True,
) -> NavigateStage:
    target = StationTarget(
        station_id=str(station_id),
        purpose=str(purpose),
        holding=None if holding is None else str(holding),
        metadata=dict(metadata or {}),
    )
    return NavigateStage(
        target=target,
        name=str(name),
        skip_if_reachable=bool(skip_if_reachable),
        fallback_if_unreachable=bool(fallback_if_unreachable),
        metadata=dict(metadata or {}),
    )


def operate(
    operation: str,
    target: str | None = None,
    *,
    name: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> OperateStage:
    return OperateStage(
        operation=str(operation),
        target=None if target is None else str(target),
        name=str(name),
        metadata=dict(metadata or {}),
    )


def finish_plan(
    episode: ResolvedEpisode,
    stages: Sequence[NavigateStage | OperateStage],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ExpertPlan:
    if not stages:
        raise ValueError(f"{episode.task_id} expert plan cannot be empty")
    merged = {
        "episode_identity": episode.identity,
        "seed": episode.seed,
        "variant": episode.variant,
        "split": episode.split,
        "env_split": episode.env_split,
        "max_horizon": episode.max_horizon,
        **dict(metadata or {}),
    }
    return ExpertPlan(task_id=episode.task_id, stages=tuple(stages), metadata=merged)


__all__ = ["finish_plan", "navigate", "operate"]
