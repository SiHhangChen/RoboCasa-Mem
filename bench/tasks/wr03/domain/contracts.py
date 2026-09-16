"""WR-03 identity and immutable episode contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "WR-03"
TASK_REVISION = "WR03-v4"
PARENT_TASK_ID = "WR-01"
ENV_NAME = "WR03StoreAndRetrieveRememberedCan"
DISPLAY_NAME = "Store and retrieve remembered can"
CAN_ID = "red_can"
ITEM_IDS = (CAN_ID,)
CABINET_LEVELS = ("lower", "upper")


class RobotBaseMode(str, Enum):
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
class ActionConfig:
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


@dataclass(frozen=True)
class EpisodePlan:
    """Private reset relation and recording-only close order for one episode."""

    seed: int
    stored_level: str
    close_order: tuple[str, str]
    max_horizon: int
    cabinet_material_id: str = "original"

    @property
    def demo_storage_level(self) -> str:
        """Legacy alias for the initial hidden cabinet level."""
        return self.stored_level

    @property
    def placement_pair_seed(self) -> int:
        return self.seed // 2


__all__ = [
    "CABINET_LEVELS",
    "CAN_ID",
    "ITEM_IDS",
    "DISPLAY_NAME",
    "ENV_NAME",
    "EpisodePlan",
    "ActionConfig",
    "InitialStateConfig",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SceneConfig",
    "TaskConfig",
    "TaskDefinition",
    "PARENT_TASK_ID",
    "TASK_ID",
    "TASK_REVISION",
]
