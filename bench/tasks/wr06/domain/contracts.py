"""WR-06 identity and immutable cross-layer contracts.

WR-06 is the single-object simplification of WR-05: one real ketchup bottle
starts on one open-shelf level, is staged beside a fixed serving plate on
either its left or right side (seed-decided), and must be returned to its
remembered original shelf level. The strict physical-event protocol (memory
barrier, wrong-first-return rejection, terminal verification) follows WR-04's
single-object design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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


TASK_ID = "WR-06"
TASK_REVISION = "WR06-v3"
ENV_NAME = "WR06ReturnSingleBottleToOriginalLevel"
DISPLAY_NAME = "Return the ketchup bottle to its original shelf level"
ITEM_ID = "ketchup"
ITEM_LABEL = "ketchup bottle"
HOME_LEVELS = ("shelf_lower", "shelf_upper")
HOME_REGIONS = HOME_LEVELS
PLATE_SIDES = ("plate_left", "plate_right")
USE_REGIONS = PLATE_SIDES
# World offsets of the two valid staging targets relative to the fixed serving
# plate center (legacy HS-WR-02 plate geometry: +/- 0.25 m in X). The expert
# side is seed-decided, while task success accepts either target.
PLATE_SIDE_WORLD_OFFSETS_M = {
    "plate_left": (-0.25, 0.0, 0.0),
    "plate_right": (0.25, 0.0, 0.0),
}


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    demo_original_level: str
    plate_stage_region: str
    placement_pair_seed: int
    max_horizon: int


@dataclass(frozen=True)
class EpisodeTargets:
    plate_stage_pose: tuple[float, float, float]
    home_return_pose: tuple[float, float, float]


@dataclass(frozen=True)
class ActionConfig:
    shelf_pick_metadata: Mapping[str, Any]
    plate_pick_metadata: Mapping[str, Any]
    plate_place_metadata: Mapping[str, Any]
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
    "PLATE_SIDE_WORLD_OFFSETS_M",
    "PLATE_SIDES",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SceneConfig",
    "TASK_ID",
    "TASK_REVISION",
    "TaskConfig",
    "TaskDefinition",
    "USE_REGIONS",
]
