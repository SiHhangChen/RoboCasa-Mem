"""Immutable TM03 identities, route definitions, and typed configuration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# TM03 owns its appliance and drawer identities.  These values intentionally
# remain local instead of importing another task package.
DRAWER_NAMES = ("drawer_left", "drawer_right")
KNOB_ID = "front_right"
KNOB_JOINT = "knob_front_right_joint"
KNOB_COLLISION_GEOM = "knob_front_right_main"
BURNER_ID = "rear_left"
BURNER_SITE = "burner_on_rear_left"
DRAWER_SIDE_COMPLETION_BUTTON = "completion_button_drawer_side"
STOVE_SIDE_COMPLETION_BUTTON = "completion_button_stove_side"
COMPLETION_BUTTON_NAMES = (DRAWER_SIDE_COMPLETION_BUTTON, STOVE_SIDE_COMPLETION_BUTTON)


TASK_ID = "TM03"
ENV_NAME = "TM03BoilWaterInspectDrawersAndRunToasterOven"
AXIS = "TQ-M"
EXPECTED_TOASTER_OVEN_ASSET = "ToasterOven059"
TOASTER_OVEN_TARGET = "toaster_oven"
TOASTER_OVEN_FOOD_NAME = "toaster_oven_food"
TOASTER_OVEN_TIME_KNOB = "time"
TOASTER_OVEN_TIME_JOINT = "knob_time_joint"
TOASTER_OVEN_TIME_COLLISION_GEOM = "knob_time_main"

CANONICAL_PROMPT = (
    "First, use the stove's own control knob to turn on the burner under the "
    "pot. The toaster oven already contains food and its door must remain closed. "
    "Next, start the toaster oven by turning its own timer knob and inspect one "
    "of the two drawers, in either "
    "order; inspect a drawer by opening it, looking inside, and closing it. "
    "After the toaster oven has been started and one drawer has been inspected, "
    "complete the three remaining steps in any order: turn off the burner with "
    "the same stove knob, stop the toaster oven by returning the same timer knob "
    "to zero, and inspect the other drawer. When the burner and toaster oven are "
    "off and both drawers have been inspected "
    "and closed, press either green completion button."
)
CANONICAL_PROMPT_SHA256 = hashlib.sha256(
    CANONICAL_PROMPT.encode("utf-8")
).hexdigest()


class StageEvent(str, Enum):
    A1 = "A1"
    B1 = "B1"
    C1 = "C1"
    A2 = "A2"
    B2 = "B2"
    C2 = "C2"
    COMPLETION_BUTTON = "completion_button"


class PrefixOrder(str, Enum):
    CB = "CB"
    BC = "BC"

    @property
    def events(self) -> tuple[StageEvent, StageEvent, StageEvent]:
        if self is PrefixOrder.CB:
            return (StageEvent.A1, StageEvent.C1, StageEvent.B1)
        return (StageEvent.A1, StageEvent.B1, StageEvent.C1)


class TailOrder(str, Enum):
    ABC = "ABC"
    ACB = "ACB"
    BAC = "BAC"
    BCA = "BCA"
    CAB = "CAB"
    CBA = "CBA"

    @property
    def events(self) -> tuple[StageEvent, StageEvent, StageEvent]:
        mapping = {
            TailOrder.ABC: (StageEvent.A2, StageEvent.B2, StageEvent.C2),
            TailOrder.ACB: (StageEvent.A2, StageEvent.C2, StageEvent.B2),
            TailOrder.BAC: (StageEvent.B2, StageEvent.A2, StageEvent.C2),
            TailOrder.BCA: (StageEvent.B2, StageEvent.C2, StageEvent.A2),
            TailOrder.CAB: (StageEvent.C2, StageEvent.A2, StageEvent.B2),
            TailOrder.CBA: (StageEvent.C2, StageEvent.B2, StageEvent.A2),
        }
        return mapping[self]


PREFIX_ORDERS = (PrefixOrder.CB, PrefixOrder.BC)
TAIL_ORDERS = (
    TailOrder.ABC,
    TailOrder.ACB,
    TailOrder.BAC,
    TailOrder.BCA,
    TailOrder.CAB,
    TailOrder.CBA,
)


@dataclass(frozen=True)
class RouteSpec:
    prefix: PrefixOrder
    tail: TailOrder

    @property
    def route_id(self) -> str:
        return f"{self.prefix.value}_{self.tail.value}"

    @property
    def events(self) -> tuple[StageEvent, ...]:
        return (*self.prefix.events, *self.tail.events)


ROUTE_SPECS = tuple(
    RouteSpec(prefix=prefix, tail=tail)
    for tail in TAIL_ORDERS
    for prefix in PREFIX_ORDERS
)


@dataclass(frozen=True)
class EpisodePlan:
    seed: int
    prefix: PrefixOrder
    tail: TailOrder
    drawer_order: tuple[str, str]
    max_horizon: int

    @property
    def route_id(self) -> str:
        return f"{self.prefix.value}_{self.tail.value}"

    @property
    def route_events(self) -> tuple[StageEvent, ...]:
        return (*self.prefix.events, *self.tail.events)


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
    expected_stove_asset: str
    knob_id: str
    burner_id: str
    knob_joint: str
    knob_collision_geom: str
    burner_site: str
    pot_name: str
    pot_xml: str
    expected_toaster_oven_asset: str
    toaster_oven_food_name: str
    toaster_oven_food_xml: str
    toaster_oven_time_joint: str
    toaster_oven_time_collision_geom: str


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
    toaster_oven_off_commit_max_rad: float
    toaster_oven_on_min_rad: float
    toaster_oven_on_commit_min_rad: float
    toaster_oven_commit_dwell_steps: int
    toaster_oven_min_gripper_contact_steps: int
    toaster_oven_contact_evidence_window_steps: int
    toaster_oven_min_interaction_motion_rad: float
    toaster_oven_max_unexplained_delta_rad_per_step: float


@dataclass(frozen=True)
class ActionConfig:
    knob_on_metadata: Mapping[str, Any]
    knob_off_metadata: Mapping[str, Any]
    drawer_open_metadata: Mapping[str, Any]
    inspect_metadata: Mapping[str, Any]
    drawer_close_metadata: Mapping[str, Any]
    toaster_oven_start_metadata: Mapping[str, Any]
    toaster_oven_stop_metadata: Mapping[str, Any]
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
    toaster_oven: Mapping[str, Any]
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
    "DRAWER_NAMES",
    "DRAWER_SIDE_COMPLETION_BUTTON",
    "ENV_NAME",
    "EXPECTED_TOASTER_OVEN_ASSET",
    "EpisodePlan",
    "KNOB_COLLISION_GEOM",
    "KNOB_ID",
    "KNOB_JOINT",
    "TOASTER_OVEN_FOOD_NAME",
    "TOASTER_OVEN_TARGET",
    "TOASTER_OVEN_TIME_COLLISION_GEOM",
    "TOASTER_OVEN_TIME_JOINT",
    "TOASTER_OVEN_TIME_KNOB",
    "ObserverConfig",
    "PREFIX_ORDERS",
    "PrefixOrder",
    "ROUTE_SPECS",
    "RouteSpec",
    "RunContext",
    "STOVE_SIDE_COMPLETION_BUTTON",
    "SceneConfig",
    "StageEvent",
    "TAIL_ORDERS",
    "TASK_ID",
    "TailOrder",
    "TaskConfig",
    "TaskDefinition",
]
