"""Immutable contracts shared by the WA-02 architecture layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "WA-02"
ENV_NAME = "WA02ServeHotColdWaterCups"
DISPLAY_NAME = "Serve remembered hot or cold water cup"
AXIS = "W-A / WA2"
CUP_IDS = ("cup_A", "cup_B")
TEMPERATURES = ("hot", "cold")
SERVICE_SLOTS = ("slot_1", "slot_2")


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
    """Strictly validated top-level WA-02 task configuration."""

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
        """Resolve a task resource relative to the YAML file."""

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


@dataclass(frozen=True)
class EpisodePlan:
    """All hidden task decisions frozen before the environment is created."""

    seed: int
    hot_cup_id: str
    cold_cup_id: str
    requested_temperature: str
    requested_cup_id: str
    fill_temperature_order: tuple[str, str]
    service_slot_by_cup: Mapping[str, str]
    max_horizon: int

    @property
    def fill_cup_order(self) -> tuple[str, str]:
        """Return cup identities in the physical fill order."""

        by_temperature = {
            "hot": self.hot_cup_id,
            "cold": self.cold_cup_id,
        }
        return tuple(by_temperature[value] for value in self.fill_temperature_order)  # type: ignore[return-value]


@dataclass(frozen=True)
class ActionConfig:
    """Effective workflow metadata derived from YAML and local resources."""

    cup_pick_metadata: Mapping[str, Any]
    cup_pick_metadata_by_cup: Mapping[str, Mapping[str, Any]]
    cup_pick_navigation_xy_tolerance: float
    cup_pick_navigation_yaw_tolerance: float
    sensor_faucet_metadata: Mapping[str, Any]
    held_fill_navigation_xy_tolerance: float
    held_fill_metadata_by_temperature: Mapping[str, Mapping[str, Any]]
    service_place_metadata_by_slot: Mapping[str, Mapping[str, Any]]
    final_pick_metadata: Mapping[str, Any]
    return_arm_metadata: Mapping[str, Any]
    return_arm_holding_cup_metadata: Mapping[str, Any]
    sink_exit_lift_metadata: Mapping[str, Any]
    handler_settings: Mapping[str, Any]
    semantic_thresholds: Mapping[str, Any]


@dataclass(frozen=True)
class RunContext:
    """Invocation metadata that must not affect hidden episode decisions."""

    split: str
    env_split: str
    variant: str
    execute: bool
    force_manual_base_path_navigation: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "AXIS",
    "ActionConfig",
    "CUP_IDS",
    "DISPLAY_NAME",
    "ENV_NAME",
    "EpisodePlan",
    "InitialStateConfig",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SERVICE_SLOTS",
    "SceneConfig",
    "TASK_ID",
    "TEMPERATURES",
    "TaskConfig",
    "TaskDefinition",
]
