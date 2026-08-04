"""Deterministic recording work plans with resume-friendly item identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from bench.core.identity import canonical_hash


@dataclass(frozen=True, slots=True)
class RecordingWorkItem:
    task_id: str
    seed: int
    variant: str = "default"
    attempt: int = 0

    @property
    def work_id(self) -> str:
        return canonical_hash({"task_id": self.task_id, "seed": self.seed, "variant": self.variant, "attempt": self.attempt})


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    """Task × seed × variant × attempt matrix for the recorder."""

    items: tuple[RecordingWorkItem, ...]

    @classmethod
    def expand(
        cls,
        task_ids: Iterable[str],
        seeds: Iterable[int],
        *,
        variants: Iterable[str] = ("default",),
        attempts: int = 1,
    ) -> "DatasetPlan":
        if int(attempts) <= 0:
            raise ValueError("attempts must be positive")
        items = tuple(
            RecordingWorkItem(task_id=str(task), seed=int(seed), variant=str(variant), attempt=attempt)
            for task in task_ids
            for seed in seeds
            for variant in variants
            for attempt in range(int(attempts))
        )
        return cls(items)

    def __len__(self) -> int:
        return len(self.items)


__all__ = ["DatasetPlan", "RecordingWorkItem"]
