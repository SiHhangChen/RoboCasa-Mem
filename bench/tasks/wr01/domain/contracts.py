"""Immutable cross-layer contracts for the red/blue-can WR-01 task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "WR-01"
TASK_REVISION = "WR01-v3"
ENV_NAME = "WR01RetrieveRememberedCan"
ITEM_IDS = ("red_can", "blue_can")
CABINET_LEVELS = ("lower", "upper")


class RobotBaseMode(str, Enum):
    """Supported reset-time mobile-base placement modes."""

    EXPLICIT_POSE = "explicit_pose"
    PRESERVE_RESET_POSE = "preserve_reset_pose"


@dataclass(frozen=True)
class RobotBaseConfig:
    mode: RobotBaseMode
    xy: tuple[float, float] | None = None
    yaw: float | None = None
    sampling_radius_m: float = 0.0


@dataclass(frozen=True)
class InitialStateConfig:
    robot_base: RobotBaseConfig
    robot_torso_height_m: float
    cabinets: Mapping[str, Any]
    objects: Mapping[str, Any]


@dataclass(frozen=True)
class TaskConfig:
    """Strictly validated task-owned configuration at the bundle boundary."""

    source_path: Path
    task: Mapping[str, Any]
    scene: Mapping[str, Any]
    episode: Mapping[str, Any]
    initial_state: InitialStateConfig
    sensors: Mapping[str, Any]
    resources: Mapping[str, Any]
    navigation: Mapping[str, Any]
    manipulation: Mapping[str, Any]
    operations: Mapping[str, Any]
    semantics: Mapping[str, Any]
    recording: Mapping[str, Any]

    def path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    display_name: str
    description: str
    success_conditions: tuple[str, ...]


@dataclass(frozen=True)
class SceneConfig:
    env_name: str
    env_split: str
    layout_id: int | None
    style_id: int | None
    scene_seed: int
    fixed_scene: bool


@dataclass(frozen=True)
class EpisodePlan:
    """All private, deterministic decisions for one WR-01 episode."""

    seed: int
    item_by_level: Mapping[str, str]
    level_by_item: Mapping[str, str]
    target_item_id: str
    target_level: str
    close_order: tuple[str, str]
    max_horizon: int

    @property
    def storage_order(self) -> tuple[str, str]:
        """Legacy alias retained for serialized descriptors."""
        return tuple(self.item_by_level[level] for level in CABINET_LEVELS)

    @property
    def distractor_item_id(self) -> str:
        return "blue_can" if self.target_item_id == "red_can" else "red_can"


@dataclass(frozen=True)
class ActionConfig:
    """Resolved production operation metadata, never a source of randomness."""

    storage_pick_metadata: Mapping[str, Any]
    storage_place_metadata: Mapping[str, Any]
    retrieval_pick_metadata: Mapping[str, Any]
    door_metadata: Mapping[str, Any]
    return_arm_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RunContext:
    split: str
    variant: str
    execute: bool
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ActionConfig",
    "CABINET_LEVELS",
    "ENV_NAME",
    "EpisodePlan",
    "ITEM_IDS",
    "InitialStateConfig",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SceneConfig",
    "TASK_ID",
    "TASK_REVISION",
    "TaskConfig",
    "TaskDefinition",
]
