"""WA-01 各层之间传递的不可变数据契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


TASK_ID = "WA-01"
ENV_NAME = "WA01SortSugaredAndUnsugaredCoffee"
COFFEE_IDS = ("coffee_A", "coffee_B")


class RobotBaseMode(str, Enum):
    """定义互斥的 reset-time Base 初始化模式。"""

    EXPLICIT_POSE = "explicit_pose"
    MANUAL_PATH_SAMPLE = "manual_path_sample"
    STATION_SAMPLE = "station_sample"
    PRESERVE_RESET_POSE = "preserve_reset_pose"


@dataclass(frozen=True)
class RobotBaseConfig:
    """描述 reset 时唯一生效的 Base 初始化方式及其参数。"""

    mode: RobotBaseMode
    xy: tuple[float, float] | None = None
    yaw: float | None = None
    sample_index: int | None = None
    station_id: str | None = None


@dataclass(frozen=True)
class InitialStateConfig:
    """汇总 Base 初始化配置和物品初始化配置。"""

    robot_base: RobotBaseConfig
    objects: Mapping[str, Any]


@dataclass(frozen=True)
class TaskConfig:
    """表示在 bundle 边界完成校验的顶层任务配置。"""

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
        """将资源路径解析为相对任务 YAML 的绝对或规范路径。"""

        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.source_path.parent / path


@dataclass(frozen=True)
class TaskDefinition:
    """定义稳定的任务身份、说明文字和最终成功条件名称。"""

    task_id: str
    display_name: str
    description: str
    success_conditions: tuple[str, ...]


@dataclass(frozen=True)
class SceneConfig:
    """描述一个 bundle 使用的环境、布局、风格和固定场景 seed。"""

    env_name: str
    env_split: str
    layout_id: int | None
    style_id: int | None
    scene_seed: int
    fixed_scene: bool
    sugar_name: str


@dataclass(frozen=True)
class EpisodePlan:
    """保存一个 seed 在整条轨迹中不可再改变的任务决策。"""

    seed: int
    delivery_order: tuple[str, str]
    final_slot_assignment: Mapping[str, str]
    max_horizon: int


@dataclass(frozen=True)
class PoseDefaults:
    """保存从校准 JSON 中解析出的操作位姿默认值。"""

    cup_place_file: Path
    sugar_cube_file: Path
    dining_place: Mapping[str, Any]
    sugar_delivery: Mapping[str, Any]
    sugar_pick: Mapping[str, Any]


@dataclass(frozen=True)
class ActionConfig:
    """保存合并配置后每类操作实际使用的 metadata。"""

    sugar_pick_metadata: Mapping[str, Any]
    sugar_delivery_metadata: Mapping[str, Any]
    dining_place_metadata_by_mug: Mapping[str, Mapping[str, Any]]
    dining_region_config: Any
    dining_regions_by_slot: Mapping[str, Mapping[str, Any]]
    sampled_dining_targets_by_slot: Mapping[str, Mapping[str, Any]]
    transport_pick_metadata_by_mug: Mapping[str, Mapping[str, Any]]
    transport_lift_metadata_by_mug: Mapping[str, Mapping[str, Any]]
    final_selection_pick_metadata: Mapping[str, Any]
    return_arm_metadata: Mapping[str, Any]
    handler_settings: Mapping[str, Any]


@dataclass(frozen=True)
class RunContext:
    """保存录制调用信息，不参与 episode 的随机决策。"""

    split: str
    variant: str
    execute: bool
    force_manual_base_path_navigation: bool = False
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "ActionConfig",
    "COFFEE_IDS",
    "ENV_NAME",
    "EpisodePlan",
    "InitialStateConfig",
    "PoseDefaults",
    "RobotBaseConfig",
    "RobotBaseMode",
    "RunContext",
    "SceneConfig",
    "TASK_ID",
    "TaskConfig",
    "TaskDefinition",
]
