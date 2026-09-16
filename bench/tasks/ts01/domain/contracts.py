"""Immutable contracts shared by the TS01 architecture layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "TS01"
ENV_NAME = "TS01AddKSugarToHotChocolate"
DISPLAY_NAME = "Add K Sugar To Hot Chocolate"
SUGAR_PLATE_NAME = "sugar_plate"
RECEPTACLE_NAME = "saucepan"
COMPLETION_BUTTON_NAME = "completion_button"
VALID_TARGET_SUGAR_COUNTS = (1, 2, 3)
MAX_SUPPORTED_TOTAL_SUGAR_COUNT = 5
DEFAULT_MAX_TOTAL_SUGAR_COUNT = 5
AXIS = "TQ-S"


class RobotBaseMode(str, Enum):
    """Mutually exclusive reset-time base initialization modes."""

    EXPLICIT_POSE = "explicit_pose"
    MANUAL_PATH_SAMPLE = "manual_path_sample"
    STATION_SAMPLE = "station_sample"
    PRESERVE_RESET_POSE = "preserve_reset_pose"


@dataclass(frozen=True)
class RobotBaseConfig:
    """Resolved reset-time base initialization parameters."""

    mode: RobotBaseMode
    xy: tuple[float, float] | None = None
    yaw: float | None = None
    sample_index: int | None = None
    station_id: str | None = None


@dataclass(frozen=True)
class InitialStateConfig:
    """Reset-time base and object initialization settings."""

    robot_base: RobotBaseConfig
    objects: Mapping[str, Any]


@dataclass(frozen=True)
class TaskConfig:
    """Strictly validated top-level task configuration."""

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
        """Resolve a resource relative to this task's YAML file."""

        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path


@dataclass(frozen=True)
class TaskDefinition:
    """Stable task identity and authoritative success condition names."""

    task_id: str
    display_name: str
    description: str
    success_conditions: tuple[str, ...]


@dataclass(frozen=True)
class SceneConfig:
    """Resolved RoboCasa scene identity."""

    env_name: str
    env_split: str
    layout_id: int | None
    style_id: int | None
    scene_seed: int
    fixed_scene: bool
    sugar_plate_name: str
    receptacle_name: str


@dataclass(frozen=True)
class EpisodePlan:
    """All TS01 episode decisions frozen before environment construction."""

    seed: int
    target_sugar_count: int
    total_sugar_count: int
    sugar_names: tuple[str, ...]
    target_sugar_names: tuple[str, ...]
    distractor_sugar_names: tuple[str, ...]
    sugar_release_region_orders: tuple[tuple[str, ...], ...]
    sugar_plate_layout_seed: int
    max_horizon: int
    saucepan_type: str = "Saucepan002"
    saucepan_asset: str = "lightwheel/saucepan/Saucepan002/model.xml"
    saucepan_material: str = "original"
    sugar_plate_material: str = "original"
    sugar_cube_material: str = "original"


@dataclass(frozen=True)
class ActionConfig:
    """Effective metadata consumed by the production workflow."""

    sugar_pick_metadata: Mapping[str, Any]
    sugar_delivery_metadata_by_index: Mapping[int, Mapping[str, Any]]
    completion_button_metadata: Mapping[str, Any]
    return_arm_metadata: Mapping[str, Any]
    handler_settings: Mapping[str, Any]
    grasp_yaw_offsets_deg: tuple[float, ...]
    grasp_finger_offset_x: float = 0.0
    grasp_finger_offset_y: float = 0.0


@dataclass(frozen=True)
class RunContext:
    """Invocation metadata that must not affect episode random decisions."""

    split: str
    env_split: str
    variant: str
    execute: bool
    force_manual_base_path_navigation: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "AXIS",
    "ActionConfig",
    "COMPLETION_BUTTON_NAME",
    "DEFAULT_MAX_TOTAL_SUGAR_COUNT",
    "DISPLAY_NAME",
    "ENV_NAME",
    "EpisodePlan",
    "InitialStateConfig",
    "MAX_SUPPORTED_TOTAL_SUGAR_COUNT",
    "RECEPTACLE_NAME",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SceneConfig",
    "SUGAR_PLATE_NAME",
    "TASK_ID",
    "TaskConfig",
    "TaskDefinition",
    "VALID_TARGET_SUGAR_COUNTS",
]
