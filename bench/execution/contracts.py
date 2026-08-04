"""Expert-only execution contracts.

The contracts describe *what* an expert wants to execute.  Navigation and
manipulation implementations decide *how* to execute already resolved targets;
they do not contain task-specific verbs or object names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bench.execution.manipulation.contracts import MotionSegment
    from bench.execution.navigation.contracts import StationTarget


class StageKind(str, Enum):
    NAVIGATE = "navigate"
    OPERATE = "operate"


@dataclass(frozen=True, slots=True)
class NavigateStage:
    target: "StationTarget"
    name: str = ""
    skip_if_reachable: bool = False
    fallback_if_unreachable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> StageKind:
        return StageKind.NAVIGATE


@dataclass(frozen=True, slots=True)
class OperateStage:
    segments: tuple["MotionSegment", ...] = ()
    # Task-local semantic operation metadata is resolved by the leaf expert
    # module. The generic runner only forwards it to the manipulation port.
    operation: str | None = None
    target: str | None = None
    name: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> StageKind:
        return StageKind.OPERATE


@dataclass(frozen=True, slots=True)
class ExpertPlan:
    task_id: str
    stages: tuple[NavigateStage | OperateStage, ...]
    version: str = "membench.expert_plan/v2"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("ExpertPlan.task_id must be non-empty")
        if not self.stages:
            raise ValueError("ExpertPlan must contain at least one stage")


@dataclass(slots=True)
class StageResult:
    index: int
    kind: StageKind
    success: bool
    failure_code: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Controllers enrich the details mapping after a stage has run.  Keep
        # the public contract mutable at this boundary while normalising the
        # mapping to a plain dict owned by the result.
        self.index = int(self.index)
        self.kind = StageKind(self.kind)
        self.success = bool(self.success)
        self.failure_code = None if self.failure_code is None else str(self.failure_code)
        self.details = dict(self.details)


__all__ = [
    "ExpertPlan",
    "NavigateStage",
    "OperateStage",
    "StageKind",
    "StageResult",
]
