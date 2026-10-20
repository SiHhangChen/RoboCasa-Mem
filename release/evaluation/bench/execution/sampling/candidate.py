"""Candidate records shared by station and pose sampling code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class Candidate(Generic[T]):
    """One sampled option.

    `value` carries the domain object, such as a base pose or TCP pose. The
    selector only looks at `score` and metadata; task-specific validators own
    feasibility checks.
    """

    value: T
    id: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        if self.id is not None:
            return str(self.id)
        for key in ("name", "path", "filename", "station_id"):
            if key in self.metadata:
                return str(self.metadata[key])
        return "<candidate>"


@dataclass(frozen=True)
class CandidateValidation(Generic[T]):
    """Validation result for a sampled candidate."""

    candidate: Candidate[T]
    valid: bool
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSelection(Generic[T]):
    """Auditable result of choosing one candidate."""

    selected: Candidate[T] | None
    selected_index: int | None
    mode: str
    candidate_count: int
    valid_count: int
    seed: int | None = None
    reason: str | None = None
    candidates: list[Candidate[T]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.selected is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass": self.passed,
            "mode": self.mode,
            "candidate_count": self.candidate_count,
            "valid_count": self.valid_count,
            "selected_index": self.selected_index,
            "selected_id": None if self.selected is None else self.selected.id,
            "seed": self.seed,
            "reason": self.reason,
        }
