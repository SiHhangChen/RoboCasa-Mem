"""Immutable TM04 identities, route definitions, and typed configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "TM04"
ENV_NAME = "TM04FillPotRetrieveCanAndAddTwoSugars"
AXIS = "TQ-M"
TASK_REVISION = "TM04-v1"
TARGET_SUGAR_COUNT = 2
TOTAL_SUGAR_COUNT = 3
SUGAR_NAMES = ("sugar0", "sugar1", "sugar2")
WATER_POT_NAME = "pot"
CAN_NAME = "can"
SUGAR_PLATE_NAME = "sugar_plate"
HOT_CHOCOLATE_NAME = "hot_chocolate_saucepan"
WATER_BUTTON = "water_button"
COMPLETION_BUTTON = "completion_button"

CANONICAL_PROMPT = (
    "First, press the blue water button beside the sink to turn on the faucet and fill "
    "the pot already in the sink. While the pot fills, complete the first part of each "
    "other task in either order: retrieve the can from the cabinet and place it in the "
    "marked area, and add one of the three sugar cubes to the hot chocolate saucepan. "
    "After the can has been placed and one sugar cube has been added, complete the three "
    "remaining steps in any order: turn off the faucet with the same blue button, close "
    "the cabinet door, and add one more sugar cube to the hot chocolate so that exactly "
    "two cubes are in the saucepan and one remains outside. When the faucet is off, the "
    "filled pot is stable in the sink, the can is in the marked area, the cabinet is "
    "closed, and exactly two sugar cubes are in the hot chocolate, press the red "
    "completion button beside the sink."
)
CANONICAL_PROMPT_SHA256 = (
    "c0e549c8d3582c38874ed520fecc813d89e87331d8e394ad60e87ae024eb85ef"
)


class StageEvent(str, Enum):
    A1 = "A1"
    B1 = "B1"
    C1 = "C1"
    A2 = "A2"
    B2 = "B2"
    C2 = "C2"
    COMPLETION_BUTTON = "completion_button"


class PrefixOrder(str, Enum):
    BC = "BC"
    CB = "CB"

    @property
    def events(self) -> tuple[StageEvent, StageEvent, StageEvent]:
        if self is PrefixOrder.BC:
            return (StageEvent.A1, StageEvent.B1, StageEvent.C1)
        return (StageEvent.A1, StageEvent.C1, StageEvent.B1)


class TailOrder(str, Enum):
    ABC = "ABC"
    ACB = "ACB"
    BAC = "BAC"
    BCA = "BCA"
    CAB = "CAB"
    CBA = "CBA"

    @property
    def events(self) -> tuple[StageEvent, StageEvent, StageEvent]:
        return tuple(StageEvent(f"{letter}2") for letter in self.value)  # type: ignore[return-value]


PREFIX_ORDERS = (PrefixOrder.BC, PrefixOrder.CB)
TAIL_ORDERS = tuple(TailOrder)


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
    target_sugar_count: int
    total_sugar_count: int
    sugar_names: tuple[str, ...]
    selected_sugar_names: tuple[str, ...]
    distractor_sugar_names: tuple[str, ...]
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
    water_button_world_xyz: tuple[float, float, float]
    completion_button_world_xyz: tuple[float, float, float]
    cabinet_world_xyz: tuple[float, float, float]
    cabinet_yaw_deg: float
    can_place_region_world_xyz: tuple[float, float, float]
    can_place_region_size_xy: tuple[float, float]
    water_pot_name: str
    can_name: str
    sugar_plate_name: str
    hot_chocolate_name: str
    sugar_names: tuple[str, ...]
    sugar_plate_world_xyz: tuple[float, float, float] | None
    hot_chocolate_world_xyz: tuple[float, float, float] | None


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
    saucepan_translation_guard_m: float
    saucepan_tilt_guard_deg: float
    saucepan_unstable_dwell_steps: int = 20


@dataclass(frozen=True)
class ActionConfig:
    water_button_metadata: Mapping[str, Any]
    cabinet_open_metadata: Mapping[str, Any]
    can_pick_metadata: Mapping[str, Any]
    can_place_metadata: Mapping[str, Any]
    cabinet_close_metadata: Mapping[str, Any]
    sugar_pick_metadata: Mapping[str, Any]
    sugar_delivery_metadata: Mapping[str, Any]
    completion_button_metadata: Mapping[str, Any]
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
    "AXIS", "ActionConfig", "CANONICAL_PROMPT", "CANONICAL_PROMPT_SHA256",
    "COMPLETION_BUTTON", "ENV_NAME", "EpisodePlan", "HOT_CHOCOLATE_NAME",
    "ObserverConfig", "PREFIX_ORDERS", "PrefixOrder", "ROUTE_SPECS", "RouteSpec",
    "RunContext", "SUGAR_NAMES", "SUGAR_PLATE_NAME", "SceneConfig", "StageEvent",
    "TARGET_SUGAR_COUNT", "TASK_ID", "TASK_REVISION", "TAIL_ORDERS", "TOTAL_SUGAR_COUNT",
    "TailOrder", "TaskConfig", "TaskDefinition", "WATER_BUTTON", "WATER_POT_NAME",
]
