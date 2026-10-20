"""Immutable contracts shared by the TM01 architecture layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "TM01"
ENV_NAME = "TM01BoilWaterAndInspectTwoDrawers"
AXIS = "TQ-M"
CANONICAL_PROMPT = (
    "You have two tasks to complete. Task A: use the stove's control knob to "
    "turn on the heating function and heat the water, then turn off the heating "
    "after the water has heated for a while. Task B: go to the other side of "
    "the workspace and inspect both drawers, opening each drawer, looking "
    "inside, and closing it. After both tasks are complete, press either green "
    "button at the two ends to indicate completion."
)
CANONICAL_PROMPT_SHA256 = (
    "3d6c2d5997d6b1f0f00f39e2294915df6eb8ad3e497b7197d302531b9631b6f3"
)
DRAWER_NAMES = ("drawer_left", "drawer_right")
KNOB_ID = "front_right"
KNOB_JOINT = "knob_front_right_joint"
KNOB_COLLISION_GEOM = "knob_front_right_main"
BURNER_ID = "rear_left"
BURNER_SITE = "burner_on_rear_left"
DRAWER_SIDE_COMPLETION_BUTTON = "completion_button_drawer_side"
STOVE_SIDE_COMPLETION_BUTTON = "completion_button_stove_side"
COMPLETION_BUTTON_NAMES = (
    DRAWER_SIDE_COMPLETION_BUTTON,
    STOVE_SIDE_COMPLETION_BUTTON,
)


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
    drawer_order: tuple[str, str]
    max_horizon: int
    drawer_material_by_fixture: Mapping[str, str] = field(default_factory=dict)


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
    drawer_names: tuple[str, str]
    drawer_world_xyzs: tuple[tuple[float, float, float], ...]
    drawer_world_yaw_deg: float
    completion_button_world_xyzs: tuple[tuple[str, tuple[float, float, float]], ...]
    stove_fixture: str
    expected_stove_asset: str
    knob_id: str
    burner_id: str
    knob_joint: str
    knob_collision_geom: str
    burner_site: str
    pot_name: str
    pot_xml: str


@dataclass(frozen=True)
class ObserverConfig:
    drawer_open_threshold: float
    drawer_closed_threshold: float
    inspect_dwell_steps: int
    button_pressed_threshold_m: float
    button_released_threshold_m: float
    button_release_dwell_steps: int
    off_commit_max_rad: float
    burner_on_min_rad: float
    on_commit_min_rad: float
    commit_dwell_steps: int
    min_gripper_contact_steps: int
    contact_evidence_window_steps: int
    min_interaction_motion_rad: float
    max_unexplained_delta_rad_per_step: float
    pot_burner_xy_threshold_m: float
    pot_translation_guard_m: float
    pot_tilt_guard_deg: float


@dataclass(frozen=True)
class ActionConfig:
    knob_on_metadata: Mapping[str, Any]
    knob_off_metadata: Mapping[str, Any]
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


@dataclass(frozen=True)
class TaskConfig:
    source_path: Path
    task: Mapping[str, Any]
    scene: Mapping[str, Any]
    episode: Mapping[str, Any]
    initial_state: Mapping[str, Any]
    semantics: Mapping[str, Any]
    appliance: Mapping[str, Any]
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
    "BURNER_ID",
    "BURNER_SITE",
    "CANONICAL_PROMPT",
    "CANONICAL_PROMPT_SHA256",
    "COMPLETION_BUTTON_NAMES",
    "DRAWER_SIDE_COMPLETION_BUTTON",
    "DRAWER_NAMES",
    "ENV_NAME",
    "EpisodePlan",
    "KNOB_COLLISION_GEOM",
    "KNOB_ID",
    "KNOB_JOINT",
    "ObserverConfig",
    "Route",
    "RunContext",
    "SceneConfig",
    "StageEvent",
    "STOVE_SIDE_COMPLETION_BUTTON",
    "TASK_ID",
    "TaskConfig",
    "TaskDefinition",
]
