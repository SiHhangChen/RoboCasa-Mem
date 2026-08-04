"""Immutable contracts passed between the WR-02 architecture layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "WR-02"
ENV_NAME = "WR02ReturnCondimentPair"
CONDIMENT_IDS = ("ketchup", "mustard")
HOME_REGIONS = ("shelf_lower", "shelf_upper")
USE_REGIONS = ("sink_left", "sink_right")
SINK_SIDE_WORLD_OFFSETS_M = {
    "sink_left": (0.20, 0.00, -0.02),
    "sink_right": (0.40, -0.03, -0.02),
}


class RobotBaseMode(str, Enum):
    """Supported reset-time mobile-base placement modes."""

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
    """Strictly validated task-owned configuration."""

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
    """All randomized WR-02 decisions frozen before environment creation."""

    seed: int
    case_id: int
    home_assignment: Mapping[str, str]
    use_assignment: Mapping[str, str]
    transport_order: tuple[str, str]
    return_sampled_offsets_xy: Mapping[str, tuple[float, float]]
    return_sampling_metadata: Mapping[str, Any]
    max_horizon: int


@dataclass(frozen=True)
class EpisodeTargets:
    use_pose_by_condiment: Mapping[str, tuple[float, float, float]]
    home_pose_by_condiment: Mapping[str, tuple[float, float, float]]
    return_place_sampling: Mapping[str, Any]
    original_home_pose_by_condiment: Mapping[str, tuple[float, float, float]]


@dataclass(frozen=True)
class ActionConfig:
    shelf_pick_by_condiment: Mapping[str, Mapping[str, Any]]
    sink_pick_by_condiment: Mapping[str, Mapping[str, Any]]
    sink_place_by_region: Mapping[str, Mapping[str, Any]]
    home_place_by_condiment: Mapping[str, Mapping[str, Any]]
    lower_torso_after_upper_pick: Mapping[str, Any]
    return_arm_empty: Mapping[str, Any]
    return_arm_holding: Mapping[str, Any]
    return_navigation: Mapping[str, Any]
    handler_settings: Mapping[str, Any]


@dataclass(frozen=True)
class RunContext:
    split: str
    variant: str
    execute: bool
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ActionConfig",
    "CONDIMENT_IDS",
    "ENV_NAME",
    "EpisodePlan",
    "EpisodeTargets",
    "HOME_REGIONS",
    "InitialStateConfig",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SINK_SIDE_WORLD_OFFSETS_M",
    "SceneConfig",
    "TASK_ID",
    "TaskConfig",
    "TaskDefinition",
    "USE_REGIONS",
]
