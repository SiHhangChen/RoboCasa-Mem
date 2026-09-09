"""Immutable contracts passed between the WX-02 architecture layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "WX-02"
ENV_NAME = "WX02CleanUnderCupAndReturnIt"
CLEANING_TOOL = "cleaning_tool"
PRIMARY_CUP = "primary_cup"
SECONDARY_CUP = "secondary_cup"
CUP_NAMES = (PRIMARY_CUP, SECONDARY_CUP)
OBJECT_NAMES = (CLEANING_TOOL, *CUP_NAMES)


class RobotBaseMode(str, Enum):
    """Mutually exclusive reset-time Base initialization modes."""

    EXPLICIT_POSE = "explicit_pose"
    MANUAL_PATH_SAMPLE = "manual_path_sample"
    STATION_SAMPLE = "station_sample"
    PRESERVE_RESET_POSE = "preserve_reset_pose"


@dataclass(frozen=True)
class RobotBaseConfig:
    """Resolved Base initialization parameters."""

    mode: RobotBaseMode
    xy: tuple[float, float] | None = None
    yaw: float | None = None
    sample_index: int | None = None
    station_id: str | None = None


@dataclass(frozen=True)
class StartStateConfig:
    """The explicit pre-recording state for the full fourteen-stage task."""

    mode: str
    setup_operation: str
    cleaning_tool_lift_height: float
    transport_probe_height: float
    max_end_effector_distance: float
    max_relative_offset_drift: float
    minimum_motion_ratio: float


@dataclass(frozen=True)
class InitialStateConfig:
    """Resolved reset-time configuration."""

    robot_base: RobotBaseConfig
    objects: Mapping[str, Any]
    short_start: StartStateConfig


@dataclass(frozen=True)
class TaskConfig:
    """Strictly validated task configuration loaded from one YAML file."""

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
        """Resolve a task resource relative to the YAML that declares it."""

        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path


@dataclass(frozen=True)
class TaskDefinition:
    """Stable benchmark identity and authoritative success-condition names."""

    task_id: str
    display_name: str
    description: str
    success_conditions: tuple[str, ...]


@dataclass(frozen=True)
class SceneConfig:
    """Resolved RoboCasa scene and object-construction settings."""

    env_name: str
    env_split: str
    layout_id: int | None
    style_id: int | None
    scene_seed: int
    fixed_scene: bool
    cup_scale: float
    cup_placement_size: tuple[float, float]
    primary_cup_placement_pos: tuple[float, float]
    secondary_cup_placement_pos: tuple[float, float]
    cup_rotation: tuple[float, float]
    cleaning_tool_return_region_center_xy: tuple[float, float]
    cleaning_tool_return_region_size_xy: tuple[float, float]
    cleaning_tool_return_region_rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class InitialObjectPose:
    """Episode-seeded world-frame XY and yaw for one task object."""

    object_name: str
    xy: tuple[float, float]
    yaw_deg: float
    region_center_xy: tuple[float, float]
    region_size_xy: tuple[float, float]
    seed: int


@dataclass(frozen=True)
class EpisodePlan:
    """All random decisions frozen before WX-02 environment execution."""

    seed: int
    initial_object_poses: Mapping[str, InitialObjectPose]
    temporary_object_xy: Mapping[str, tuple[float, float]]
    staged_tool_xy: tuple[float, float]
    temporary_target_metadata: Mapping[str, Any]
    wipe_offsets_xy_by_object: Mapping[
        str, tuple[tuple[float, float], ...]
    ]
    wipe_metadata_by_object: Mapping[str, Mapping[str, Any]]
    max_horizon: int


@dataclass(frozen=True)
class EpisodeTargets:
    """World targets resolved after the reset-time original poses are known."""

    original_tool_pose: tuple[float, float, float]
    original_object_poses: Mapping[str, tuple[float, float, float]]
    temporary_object_poses: Mapping[str, tuple[float, float, float]]
    staged_tool_pose: tuple[float, float, float]
    post_wipe_tool_pose: tuple[float, float, float]
    cleaning_tool_return_region_pose: tuple[float, float, float]
    stain_center_poses: Mapping[str, tuple[float, float, float]]


@dataclass(frozen=True)
class ActionConfig:
    """Effective metadata consumed by the fourteen production stages."""

    observe_metadata: Mapping[str, Any]
    cleaning_tool_pick_metadata: Mapping[str, Any]
    staged_tool_place_metadata: Mapping[str, Any]
    object_temporary_place_metadata: Mapping[str, Any]
    wipe_metadata_by_object: Mapping[str, Mapping[str, Any]]
    post_wipe_tool_place_metadata: Mapping[str, Any]
    object_pick_metadata: Mapping[str, Any]
    object_origin_place_metadata: Mapping[str, Any]
    cleaning_tool_return_place_metadata: Mapping[str, Any]
    handler_settings: Mapping[str, Any]
    pick_proxy_settings: Mapping[str, Any]


@dataclass(frozen=True)
class RunContext:
    """Invocation data that must not affect episode random decisions."""

    split: str
    variant: str
    execute: bool
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ActionConfig",
    "CLEANING_TOOL",
    "CUP_NAMES",
    "ENV_NAME",
    "EpisodePlan",
    "EpisodeTargets",
    "InitialObjectPose",
    "InitialStateConfig",
    "OBJECT_NAMES",
    "PRIMARY_CUP",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SECONDARY_CUP",
    "SceneConfig",
    "StartStateConfig",
    "TASK_ID",
    "TaskConfig",
    "TaskDefinition",
]
