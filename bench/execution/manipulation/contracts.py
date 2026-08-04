"""Manipulation contracts with no task-specific operation names."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PoseTarget:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    frame: str = "world"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MotionSegment:
    target: PoseTarget
    gripper: float | None = None
    dwell_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MotionResult:
    success: bool
    steps: int = 0
    reason: str | None = None


__all__ = ["MotionResult", "MotionSegment", "PoseTarget"]
