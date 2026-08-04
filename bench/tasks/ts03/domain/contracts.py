"""TS03 identity and the four-cell hierarchy contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any


TASK_ID = "TS03"
TASK_REVISION = "TS03-v1"
ENV_NAME = "TS03HierarchicalAddKSugarToHotChocolate"
DISPLAY_NAME = "Hierarchical Add K Sugar To Hot Chocolate"
HIERARCHY_SCHEMA_VERSION = 1
AXIS = "TQ-S"
SUGAR_PLATE_NAME = "sugar_plate"
RECEPTACLE_NAME = "saucepan"
COMPLETION_BUTTON_NAME = "completion_button"


class RobotBaseMode(str, Enum):
    """Mutually exclusive reset-time base initialization modes."""

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
class EpisodePlan:
    seed: int
    target_sugar_count: int
    total_sugar_count: int
    sugar_names: tuple[str, ...]
    target_sugar_names: tuple[str, ...]
    distractor_sugar_names: tuple[str, ...]
    sugar_release_region_orders: tuple[tuple[str, ...], ...]
    sugar_plate_layout_seed: int
    max_horizon: int

@dataclass(frozen=True)
class SceneConfig:
    env_name: str
    env_split: str
    layout_id: int
    style_id: int
    scene_seed: int
    fixed_scene: bool
    sugar_plate_name: str
    receptacle_name: str

@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    display_name: str
    description: str
    success_conditions: tuple[str, ...]

@dataclass(frozen=True)
class ActionConfig:
    sugar_pick_metadata: Mapping[str, Any] = field(default_factory=dict)
    sugar_delivery_metadata_by_index: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    completion_button_metadata: Mapping[str, Any] = field(default_factory=dict)
    return_arm_metadata: Mapping[str, Any] = field(default_factory=dict)
    handler_settings: Mapping[str, Any] = field(default_factory=dict)
    grasp_yaw_offsets_deg: tuple[float, ...] = ()

@dataclass(frozen=True)
class RunContext:
    split: str
    env_split: str
    variant: str
    execute: bool
    force_manual_base_path_navigation: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

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
class SugarCountProfile:
    """One separately reported TS03 K/N cell."""

    profile_id: str
    hierarchy_tier: int
    target_sugar_count: int
    total_sugar_count: int
    role: str
    hierarchy_branch: str

    @property
    def distractor_count(self) -> int:
        return self.total_sugar_count - self.target_sugar_count

    @property
    def is_add_all_control(self) -> bool:
        return self.profile_id == "k2_n2"

    @property
    def is_core_memory_cell(self) -> bool:
        return self.profile_id == "k2_n3"


_PROFILE_SEQUENCE = (
    SugarCountProfile("k1_n2", 1, 1, 2, "weak_one_step_stop", "foundation"),
    SugarCountProfile("k1_n3", 2, 1, 3, "one_step_with_extra_distractors", "selection"),
    SugarCountProfile("k2_n2", 2, 2, 2, "add_all_skill_control", "control"),
    SugarCountProfile("k2_n3", 3, 2, 3, "core_memory_cell", "core"),
)
PROFILES = MappingProxyType({item.profile_id: item for item in _PROFILE_SEQUENCE})
PROFILE_IDS = tuple(PROFILES)
VALID_CELLS = tuple(
    (item.target_sugar_count, item.total_sugar_count) for item in _PROFILE_SEQUENCE
)


def profile_for_id(profile_id: Any) -> SugarCountProfile:
    normalized = str(profile_id).strip().lower()
    try:
        return PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown TS03 profile_id {profile_id!r}; expected one of {PROFILE_IDS}"
        ) from exc


def profile_for_counts(target_sugar_count: Any, total_sugar_count: Any) -> SugarCountProfile:
    try:
        cell = (int(target_sugar_count), int(total_sugar_count))
    except (TypeError, ValueError) as exc:
        raise ValueError("TS03 K and N must be integers") from exc
    for profile in _PROFILE_SEQUENCE:
        if cell == (profile.target_sugar_count, profile.total_sugar_count):
            return profile
    raise ValueError(f"TS03 only supports K/N cells {VALID_CELLS}; got {cell}")


def profile_for_seed(seed: Any) -> SugarCountProfile:
    episode_seed = int(seed)
    if episode_seed < 0:
        raise ValueError("TS03 seed must be non-negative")
    return _PROFILE_SEQUENCE[episode_seed % len(_PROFILE_SEQUENCE)]


def profile_metadata(profile: SugarCountProfile) -> dict[str, Any]:
    return {
        "task_revision": TASK_REVISION,
        "hierarchy_schema_version": HIERARCHY_SCHEMA_VERSION,
        "profile_id": profile.profile_id,
        "hierarchy_tier": profile.hierarchy_tier,
        "hierarchy_branch": profile.hierarchy_branch,
        "profile_role": profile.role,
        "target_sugar_count": profile.target_sugar_count,
        "total_sugar_count": profile.total_sugar_count,
        "distractor_count": profile.distractor_count,
        "is_add_all_control": profile.is_add_all_control,
        "is_core_memory_cell": profile.is_core_memory_cell,
    }


def episode_profile_metadata(episode: Any) -> dict[str, Any]:
    return profile_metadata(
        profile_for_counts(episode.target_sugar_count, episode.total_sugar_count)
    )


__all__ = [
    "AXIS", "ActionConfig", "COMPLETION_BUTTON_NAME", "EpisodePlan", "InitialStateConfig", "RobotBaseConfig", "RobotBaseMode", "SceneConfig", "TaskConfig", "TaskDefinition", "RunContext",
    "DISPLAY_NAME",
    "ENV_NAME",
    "HIERARCHY_SCHEMA_VERSION",
    "PROFILES",
    "PROFILE_IDS",
    "SugarCountProfile",
    "TASK_ID",
    "TASK_REVISION",
    "SUGAR_PLATE_NAME", "RECEPTACLE_NAME",
    "VALID_CELLS",
    "episode_profile_metadata",
    "profile_for_counts",
    "profile_for_id",
    "profile_for_seed",
    "profile_metadata",
]
