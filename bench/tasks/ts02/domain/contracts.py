"""Immutable contracts passed between the TS02 architecture layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "TS02"
ENV_NAME = "TS02InspectThreeCabinets"
AXIS = "TQ-S"
TARGET_OBJECT_NAME = "red_cube"
FIXTURE_NAMES = (
    "cabinet_slot_0",
    "cabinet_slot_1",
    "cabinet_slot_2",
)
FIXTURE_DISPLAY_NAMES = {
    "cabinet_slot_0": "left",
    "cabinet_slot_1": "center",
    "cabinet_slot_2": "right",
}


class RobotBaseMode(str, Enum):
    """Mutually exclusive reset-time mobile-base initialization modes."""

    EXPLICIT_POSE = "explicit_pose"
    MANUAL_PATH_SAMPLE = "manual_path_sample"
    STATION_SAMPLE = "station_sample"
    PRESERVE_RESET_POSE = "preserve_reset_pose"


@dataclass(frozen=True)
class RobotBaseConfig:
    mode: RobotBaseMode
    xy: tuple[float, float] | None = None
    yaw: float | None = None
    station_id: str | None = None
    sample_index: int | None = None


@dataclass(frozen=True)
class InitialStateConfig:
    robot_base: RobotBaseConfig
    drawers: Mapping[str, Any]


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
    recording: Mapping[str, Any]

    def path(self, value: str | Path | None) -> Path | None:
        """Resolve a resource relative to the task YAML."""

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
    fixture_names: tuple[str, ...]
    drawer_world_xyzs: tuple[tuple[float, float, float], ...]
    drawer_world_yaw_deg: float
    completion_button_world_xyz: tuple[float, float, float]
    open_threshold: float
    closed_threshold: float


@dataclass(frozen=True)
class DrawerOpenVariation:
    prehook_variant: str
    prehook_rotation_offset_deg: tuple[float, float, float]
    handle_side_offset_m: float


@dataclass(frozen=True)
class EpisodePlan:
    """All per-episode decisions, frozen before the environment is created."""

    seed: int
    demo_fixture_order: tuple[str, ...]
    target_fixture: str
    max_horizon: int
    drawer_material_by_fixture: Mapping[str, str] = field(default_factory=dict)
    drawer_open_variation_by_fixture: Mapping[str, DrawerOpenVariation] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ActionConfig:
    drawer_open_metadata: Mapping[str, Any]
    inspect_metadata: Mapping[str, Any]
    drawer_close_metadata: Mapping[str, Any]
    completion_button_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RunContext:
    split: str
    env_split: str
    variant: str
    execute: bool
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "AXIS",
    "ActionConfig",
    "DrawerOpenVariation",
    "ENV_NAME",
    "EpisodePlan",
    "FIXTURE_DISPLAY_NAMES",
    "FIXTURE_NAMES",
    "InitialStateConfig",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SceneConfig",
    "TASK_ID",
    "TARGET_OBJECT_NAME",
    "TaskConfig",
    "TaskDefinition",
]
