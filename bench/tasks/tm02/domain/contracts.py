"""Immutable contracts shared by the TM02 architecture layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "TM02"
ENV_NAME = "TM02FillPotAndRetrieveCan"
AXIS = "TQ-M"
CANONICAL_PROMPT = (
    "You have two tasks to complete. Task A: press the blue water button "
    "beside the sink to turn on the faucet and fill the pot already in the "
    "sink, then press the same blue button to turn off the faucet after the "
    "pot has filled for a while. Task B: go to the cabinet, open the sliding "
    "door, take out the can, place it in the marked area, and close the "
    "cabinet door. After both tasks are complete, press the red completion "
    "button beside the sink to indicate completion."
)
CANONICAL_PROMPT_SHA256 = (
    "10fdaf85133b54744b92c82883e7b34a6815e7341b095cdbf31c25f2c075d67b"
)

POT_NAME = "pot"
CAN_NAME = "can"
CABINET_NAME = "tm02_can_cabinet"
WATER_BUTTON = "water_button"
COMPLETION_BUTTON = "completion_button"
BUTTON_NAMES = (WATER_BUTTON, COMPLETION_BUTTON)


class Route(str, Enum):
    L1 = "L1"
    L2 = "L2"


class StageEvent(str, Enum):
    A1 = "A1"
    B1 = "B1"
    A2 = "A2"
    B2 = "B2"
    COMPLETION_BUTTON = "completion_button"


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    route: Route
    max_horizon: int


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
    layout_id: int
    style_id: int
    scene_seed: int
    water_button_world_xyz: tuple[float, float, float]
    completion_button_world_xyz: tuple[float, float, float]
    cabinet_world_xyz: tuple[float, float, float]
    cabinet_yaw_deg: float
    can_place_region_world_xyz: tuple[float, float, float]
    can_place_region_size_xy: tuple[float, float]
    pot_name: str
    pot_xml: str
    can_name: str
    can_xml: str
    can_scale: float


@dataclass(frozen=True)
class ObserverConfig:
    cabinet_open_threshold: float
    cabinet_closed_threshold: float
    button_pressed_threshold_m: float
    button_released_threshold_m: float
    button_release_dwell_steps: int
    can_place_xy_tolerance_m: float
    pot_water_xy_threshold_m: float
    pot_translation_guard_m: float
    pot_tilt_guard_deg: float
    pot_fill_required_steps: int
    pot_filled_threshold: float


@dataclass(frozen=True)
class ActionConfig:
    water_button_metadata: Mapping[str, Any]
    completion_button_metadata: Mapping[str, Any]
    cabinet_open_metadata: Mapping[str, Any]
    can_pick_metadata: Mapping[str, Any]
    can_place_metadata: Mapping[str, Any]
    cabinet_close_metadata: Mapping[str, Any]
    return_arm_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RunContext:
    split: str
    env_split: str
    variant: str
    execute: bool
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskConfig:
    source_path: Path
    task: Mapping[str, Any]
    scene: Mapping[str, Any]
    episode: Mapping[str, Any]
    initial_state: Mapping[str, Any]
    semantics: Mapping[str, Any]
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


__all__ = [
    "AXIS",
    "ActionConfig",
    "BUTTON_NAMES",
    "CABINET_NAME",
    "CAN_NAME",
    "CANONICAL_PROMPT",
    "CANONICAL_PROMPT_SHA256",
    "ENV_NAME",
    "COMPLETION_BUTTON",
    "EpisodePlan",
    "ObserverConfig",
    "POT_NAME",
    "Route",
    "RunContext",
    "SceneConfig",
    "StageEvent",
    "TASK_ID",
    "TaskConfig",
    "TaskDefinition",
    "WATER_BUTTON",
]
