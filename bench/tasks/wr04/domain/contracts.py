"""WR-04 identity and immutable cross-layer contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Any, Mapping


class RobotBaseMode(str, Enum):
    EXPLICIT_POSE = "explicit_pose"
    PRESERVE_RESET_POSE = "preserve_reset_pose"


@dataclass(frozen=True)
class RobotBaseConfig:
    mode: RobotBaseMode
    xy: tuple[float, float] | None = None
    yaw: float | None = None


@dataclass(frozen=True)
class InitialStateConfig:
    robot_base: RobotBaseConfig
    home_regions: Mapping[str, Mapping[str, Any]]
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
class RunContext:
    split: str
    variant: str
    execute: bool
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


TASK_ID = "WR-04"
TASK_REVISION = "WR04-v1"
ENV_NAME = "WR04ReturnSingleObjectToOriginalLevel"
DISPLAY_NAME = "Return one item to its original shelf level"
ITEM_ID = "ketchup"
ITEM_LABEL = "sponge"
HOME_LEVELS = ("shelf_lower", "shelf_upper")
HOME_REGIONS = HOME_LEVELS
SINK_STAGE_REGION = "sink_left"
USE_REGIONS = (SINK_STAGE_REGION,)
SINK_SIDE_WORLD_OFFSETS_M = {SINK_STAGE_REGION: (0.20, 0.00, -0.02)}


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    demo_original_level: str
    placement_pair_seed: int
    max_horizon: int


@dataclass(frozen=True)
class EpisodeTargets:
    sink_stage_pose: tuple[float, float, float]
    home_return_pose: tuple[float, float, float]


@dataclass(frozen=True)
class ActionConfig:
    shelf_pick_metadata: Mapping[str, Any]
    sink_pick_metadata: Mapping[str, Any]
    sink_place_metadata: Mapping[str, Any]
    home_place_metadata: Mapping[str, Any]
    normalize_torso_metadata: Mapping[str, Any]
    return_arm_empty: Mapping[str, Any]
    return_arm_holding: Mapping[str, Any]
    return_navigation: Mapping[str, Any]
    handler_settings: Mapping[str, Any]


__all__ = [
    "ActionConfig",
    "DISPLAY_NAME",
    "ENV_NAME",
    "EpisodePlan",
    "EpisodeTargets",
    "HOME_LEVELS",
    "HOME_REGIONS",
    "InitialStateConfig",
    "ITEM_ID",
    "ITEM_LABEL",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SINK_STAGE_REGION",
    "SINK_SIDE_WORLD_OFFSETS_M",
    "SceneConfig",
    "TASK_ID",
    "TASK_REVISION",
    "TaskConfig",
    "TaskDefinition",
    "USE_REGIONS",
]
