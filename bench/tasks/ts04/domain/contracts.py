"""TS04 identity and fixed two-drawer contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections.abc import Mapping
from typing import Any

TASK_ID = "TS04"
TASK_REVISION = "TS04-v1"
PARENT_TASK_ID = ""
ENV_NAME = "TS04InspectTwoDrawers"
DISPLAY_NAME = "Inspect Two Drawers"
AXIS = "TQ-S"
FIXTURE_NAMES = ("cabinet_slot_0", "cabinet_slot_1")
FIXTURE_DISPLAY_NAMES = {
    "cabinet_slot_0": "left",
    "cabinet_slot_1": "right",
}
DRAWER_WORLD_XYZS = (
    (4.10, -2.37, 1.1363333333333332),
    (4.60, -2.37, 1.1363333333333332),
)
COMPLETION_BUTTON_WORLD_XYZ = (4.35, -2.06, 0.92)

@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    demo_fixture_order: tuple[str, str]
    max_horizon: int

@dataclass(frozen=True)
class SceneConfig:
    env_name: str
    env_split: str
    layout_id: int
    style_id: int
    scene_seed: int
    fixed_scene: bool
    fixture_names: tuple[str, ...]
    drawer_world_xyzs: tuple[tuple[float, float, float], ...]
    drawer_world_yaw_deg: float
    completion_button_world_xyz: tuple[float, float, float]
    open_threshold: float = 0.70
    closed_threshold: float = 0.08

@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    display_name: str
    description: str
    success_conditions: tuple[str, ...]

@dataclass(frozen=True)
class ActionConfig:
    drawer_open_metadata: Mapping[str, Any] = field(default_factory=dict)
    inspect_metadata: Mapping[str, Any] = field(default_factory=dict)
    drawer_close_metadata: Mapping[str, Any] = field(default_factory=dict)
    completion_button_metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RunContext:
    split: str
    env_split: str
    variant: str
    execute: bool
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


class RobotBaseMode(str, Enum):
    EXPLICIT_POSE = "explicit_pose"
    MANUAL_PATH_SAMPLE = "manual_path_sample"
    STATION_SAMPLE = "station_sample"
    PRESERVE_RESET_POSE = "preserve_reset_pose"


@dataclass(frozen=True)
class RobotBaseConfig:
    mode: RobotBaseMode = RobotBaseMode.PRESERVE_RESET_POSE
    xy: tuple[float, float] | None = None
    yaw: float | None = None
    station_id: str | None = None
    sample_index: int | None = None


@dataclass(frozen=True)
class InitialStateConfig:
    robot_base: RobotBaseConfig = field(default_factory=RobotBaseConfig)
    drawers: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class TaskConfig:
    source_path: Path
    task: Mapping[str, Any]
    scene: Mapping[str, Any]
    episode: Mapping[str, Any]
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    sensors: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    navigation: Mapping[str, Any] = field(default_factory=dict)
    manipulation: Mapping[str, Any] = field(default_factory=dict)
    operations: Mapping[str, Any] = field(default_factory=dict)
    recording: Mapping[str, Any] = field(default_factory=dict)

    def path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path


__all__ = [
    "AXIS",
    "ActionConfig", "EpisodePlan", "InitialStateConfig", "RobotBaseConfig", "RobotBaseMode", "RunContext", "SceneConfig", "TaskConfig", "TaskDefinition",
    "COMPLETION_BUTTON_WORLD_XYZ",
    "DISPLAY_NAME",
    "DRAWER_WORLD_XYZS",
    "ENV_NAME",
    "FIXTURE_DISPLAY_NAMES",
    "FIXTURE_NAMES",
    "PARENT_TASK_ID",
    "TASK_ID",
    "TASK_REVISION",
]
