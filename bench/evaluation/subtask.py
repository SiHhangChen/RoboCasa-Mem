"""Contracts for optional prompt-driven subtask scheduling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SubtaskTransition:
    """Result of inspecting one simulator transition."""

    changed: bool = False
    complete: bool = False
    failed: bool = False
    previous_subtask: str | None = None
    current_subtask: str | None = None
    prompt: str | None = None
    environment_step: int | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": bool(self.changed),
            "complete": bool(self.complete),
            "failed": bool(self.failed),
            "previous_subtask": self.previous_subtask,
            "current_subtask": self.current_subtask,
            "prompt": self.prompt,
            "environment_step": self.environment_step,
            "failure_reason": self.failure_reason,
        }


class SubtaskScheduler(Protocol):
    """Duck-typed scheduler consumed by :class:`RolloutRunner`."""

    @property
    def current_subtask(self) -> str: ...

    @property
    def current_prompt(self) -> str: ...

    def after_step(
        self,
        runtime: Any,
        info: Mapping[str, Any],
    ) -> SubtaskTransition: ...

    def to_metadata(self) -> Mapping[str, Any]: ...


__all__ = ["SubtaskScheduler", "SubtaskTransition"]
