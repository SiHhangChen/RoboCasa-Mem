"""Task-local physical configuration contracts for TS03 sugar execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RobotBaseMode(str, Enum):
    EXPLICIT_POSE = "explicit_pose"
    MANUAL_PATH_SAMPLE = "manual_path_sample"
    STATION_SAMPLE = "station_sample"
    PRESERVE_RESET_POSE = "preserve_reset_pose"


@dataclass(frozen=True)
class RobotBaseConfig:
    mode: RobotBaseMode
    xy: tuple[float, float] | None = None
    yaw: float | None = None
    sample_index: int | None = None
    station_id: str | None = None


@dataclass(frozen=True)
class InitialStateConfig:
    robot_base: RobotBaseConfig
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
    recording: Mapping[str, Any]

    def path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path


@dataclass(frozen=True)
class ActionConfig:
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
    split: str
    env_split: str
    variant: str
    execute: bool
    force_manual_base_path_navigation: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ActionConfig", "InitialStateConfig", "RobotBaseConfig", "RobotBaseMode",
    "RunContext", "TaskConfig",
]
