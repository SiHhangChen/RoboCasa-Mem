"""Pure exact-count episode and stage-shape helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SugarCountSpec:
    target_count: int
    total_count: int

    def __post_init__(self) -> None:
        if self.target_count <= 0 or self.total_count < self.target_count:
            raise ValueError("invalid sugar-count relation")


def target_names(total_count: int, target_count: int) -> tuple[str, ...]:
    spec = SugarCountSpec(target_count, total_count)
    return tuple(f"item_{index}" for index in range(spec.target_count))
