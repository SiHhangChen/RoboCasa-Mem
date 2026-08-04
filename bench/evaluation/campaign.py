"""Policy-neutral campaign orchestration across task suites."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from bench.evaluation.rollout import EpisodeResult, RolloutRunner
from bench.evaluation.suite import EvalSuiteConfig
from bench.tasks import TaskCatalog


@dataclass(frozen=True, slots=True)
class EvaluationCampaign:
    """A deterministic collection of suites; no policy implementation leaks in."""

    suites: tuple[EvalSuiteConfig, ...]
    release_id: str = "unreleased"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_suites(cls, suites: Iterable[EvalSuiteConfig], *, release_id: str = "unreleased"):
        values = tuple(suites)
        if not values:
            raise ValueError("EvaluationCampaign requires at least one suite")
        unknown: list[str] = []
        for suite in values:
            try:
                TaskCatalog.canonical_id(suite.task_id)
            except KeyError:
                unknown.append(suite.task_id)
        if unknown:
            raise ValueError(f"campaign contains unknown tasks: {unknown}")
        return cls(values, release_id=release_id)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(suite.task_id for suite in self.suites)

    def run(self, runner_factory: Any, *, resume: bool = True, retry_errors: bool = False) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        for suite in self.suites:
            runner: RolloutRunner = runner_factory(suite)
            adapter = runner_factory.adapter(suite) if hasattr(runner_factory, "adapter") else None
            if adapter is None:
                raise TypeError("runner_factory must provide adapter(suite)")
            results.extend(runner.run_suite(adapter, resume=resume, retry_errors=retry_errors))
        return results


__all__ = ["EvaluationCampaign"]
