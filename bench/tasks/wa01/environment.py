"""WA-01 的 RoboCasa 场景与物理观察边界。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.wa01.domain.contracts import COFFEE_IDS
from bench.tasks.wa01.domain.geometry import (
    DiningPlacementRegionConfig,
    compute_dining_regions,
    dining_region_config_summary,
    is_xy_inside_region,
)
from bench.tasks.wa01.domain.semantics import (
    SemanticState,
    evaluate_success,
    latch_cups_together,
    record_selection,
    record_sugar_delivery,
)


COFFEE_MUG_XML = f"{assets_root}/objects/objaverse/cup/cup_2/model.xml"
SUGAR_PLATE_XML = f"{assets_root}/objects/objaverse/plate/plate_3/model.xml"
SUGAR_CUBE_XML = f"{assets_root}/objects/lightwheel/sugar_cube/SugarCube003/model.xml"
NORMAL_COFFEE_RGBA = (0.65, 0.45, 0.25, 0.8)
OVERHEAD_CABINET_FIXTURES = ("cab_1_left_group", "cab_2_left_group")
MIN_FINAL_CUP_XY_SEPARATION = 0.20


class WA01SortSugaredAndUnsugaredCoffee(Kitchen):
    """承载 WA-01 场景与物理观察，流程和执行逻辑位于类外。"""

    def __init__(
        self,
        minimum_cup_xy_separation: float = MIN_FINAL_CUP_XY_SEPARATION,
        cup_upright_threshold_deg: float = 15.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """保存任务阈值和初始语义状态，再交由 Kitchen 构建环境。"""

        self.minimum_cup_xy_separation = float(minimum_cup_xy_separation)
        self.cup_upright_threshold_deg = float(cup_upright_threshold_deg)
        self._semantic_state = SemanticState()
        super().__init__(*args, **kwargs)

    # Scene construction -------------------------------------------------

    def edit_model_xml(self, xml_str: str) -> str:
        """在布局 29 中移除遮挡咖啡机操作空间的上方柜体。"""

        xml_str = super().edit_model_xml(xml_str)
        if int(self.layout_id) != 29:
            return xml_str
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_str
        root_names = {f"{name}_main" for name in OVERHEAD_CABINET_FIXTURES}
        parent_by_child = {
            child: parent for parent in worldbody.iter() for child in parent
        }
        for body in list(worldbody.iter("body")):
            if body.get("name") not in root_names:
                continue
            parent = parent_by_child.get(body)
            if parent is not None:
                parent.remove(body)
        return ET.tostring(root, encoding="unicode")

    def _setup_references(self) -> None:
        """同步清理已从 XML 移除的 fixture 引用和几何映射。"""

        if int(self.layout_id) != 29:
            super()._setup_references()
            return
        removed_names = set(OVERHEAD_CABINET_FIXTURES)
        removed_fixtures = {
            fixture for name, fixture in self.fixtures.items() if name in removed_names
        }
        self.model.mujoco_objects = [
            model for model in self.model.mujoco_objects if model not in removed_fixtures
        ]
        for name in removed_names:
            self.fixtures.pop(name, None)
        self.fixture_cfgs = [
            config for config in self.fixture_cfgs if config.get("name") not in removed_names
        ]
        super()._setup_references()

    def _setup_kitchen_references(self) -> None:
        """注册咖啡机、操作台、炉灶、凳子和餐桌等场景引用。"""

        super()._setup_kitchen_references()
        preferred_machine = self.get_fixture(
            "coffee_machine_front_group_1",
            full_name_check=True,
        )
        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine",
            dict(id=preferred_machine or FixtureType.COFFEE_MACHINE),
        )
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.coffee_machine),
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool),
        )
        self.init_robot_base_ref = self.counter

    def _setup_scene(self) -> None:
        """重置语义状态、餐桌区域、物理缓存和咖啡视觉。"""

        super()._setup_scene()
        self.coffee_machine._turned_on = False
        self._semantic_state = SemanticState()
        default_regions = DiningPlacementRegionConfig()
        self.dining_region_config = dining_region_config_summary(default_regions)
        self.dining_regions_by_slot = compute_dining_regions(default_regions)
        self.sampled_dining_targets_by_slot: dict[str, dict[str, Any]] = {}
        self.planned_dining_slot_by_coffee = {
            "coffee_A": "left",
            "coffee_B": "right",
        }
        self.coffee_A_on_dining_counter = False
        self.coffee_B_on_dining_counter = False
        self.coffee_A_upright = False
        self.coffee_B_upright = False
        self.dining_regions_occupied = False
        self.cups_together = False
        self._set_coffee_liquid_visual("coffee_A", visible=True)
        self._set_coffee_liquid_visual("coffee_B", visible=True)

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        """定义两只同款杯子、糖碟和糖块的 RoboCasa 物品配置。"""

        return [
            dict(
                name="coffee_A",
                obj_groups=COFFEE_MUG_XML,
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.coffee_machine, loc="right"),
                    size=(0.26, 0.18),
                    pos=("ref", -0.55),
                    rotation=[np.pi / 4, np.pi / 2],
                ),
            ),
            dict(
                name="coffee_B",
                obj_groups=COFFEE_MUG_XML,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.coffee_machine, loc="right"),
                    size=(0.26, 0.18),
                    pos=("ref", 0.28),
                    rotation=[np.pi / 4, np.pi / 2],
                ),
            ),
            dict(
                name="sugar_plate",
                obj_groups=SUGAR_PLATE_XML,
                object_scale=[0.6, 0.6, 1.0],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove, loc="left"),
                    size=(0.32, 0.24),
                    pos=("ref", -1.0),
                ),
            ),
            dict(
                name="sugar_cube",
                obj_groups=SUGAR_CUBE_XML,
                object_scale=1.2,
                placement=dict(object="sugar_plate", size=(0.5, 0.5)),
            ),
        ]

    # Semantic events ----------------------------------------------------

    @property
    def sugar_consumed(self) -> bool:
        """返回糖块是否已经被语义事件消费。"""

        return self._semantic_state.sugar_consumed

    @property
    def sugared_coffee_id(self) -> str | None:
        """返回实际收到糖块的杯子标识。"""

        return self._semantic_state.sugared_coffee_id

    @property
    def unsugared_coffee_id(self) -> str | None:
        """返回没有收到糖块的另一只杯子标识。"""

        return self._semantic_state.unsugared_coffee_id

    @property
    def cups_together_before_selection(self) -> bool:
        """返回最终抓杯前是否曾锁存双杯正确就位。"""

        return self._semantic_state.cups_together_before_selection

    @property
    def selected_sugared_coffee_id(self) -> str | None:
        """返回最终被选择并抓起的杯子标识。"""

        return self._semantic_state.selected_sugared_coffee_id

    @property
    def correct_sugared_selection(self) -> bool:
        """返回最终选择是否指向实际加糖杯。"""

        return evaluate_success(self._semantic_state).correct_sugared_selection

    def record_sugar_delivery(self, coffee_id: str) -> None:
        """记录糖投递到任意一杯咖啡的事件，并隐藏糖块。"""

        if record_sugar_delivery(self._semantic_state, coffee_id):
            self._hide_consumed_sugar_cube()
        self.update_state()

    def record_cups_together_before_selection(self) -> bool:
        """刷新物理状态并锁存双杯已正确放好的历史条件。"""

        self.update_state()
        return latch_cups_together(
            self._semantic_state,
            currently_together=self.cups_together,
        )

    def record_sugared_coffee_selection(self, coffee_id: str) -> None:
        """记录最终选杯事件并刷新派生语义结果。"""

        record_selection(self._semantic_state, coffee_id)
        self.update_state()

    # Physical observations ---------------------------------------------

    def update_state(self) -> None:
        """从 MuJoCo 当前状态刷新接触、倾角、区域占用和双杯条件。"""

        super().update_state()
        for coffee_id in COFFEE_IDS:
            setattr(
                self,
                f"{coffee_id}_on_dining_counter",
                self._is_cup_on_dining_counter(coffee_id),
            )
            setattr(self, f"{coffee_id}_upright", self._is_cup_upright(coffee_id))
        self.dining_regions_occupied = self._are_both_dining_regions_occupied()
        self.cups_together = self._are_cups_correctly_arranged()

    def is_coffee_cup_correctly_placed_on_dining_counter(self, coffee_id: str) -> bool:
        """检查指定杯子是否接触餐桌且保持竖直。"""

        self._validate_coffee_id(coffee_id)
        return bool(
            getattr(self, f"{coffee_id}_on_dining_counter", False)
            and getattr(self, f"{coffee_id}_upright", False)
        )

    def is_coffee_cup_center_inside_dining_region(self, coffee_id: str, slot: str) -> bool:
        """检查指定杯子中心是否位于给定左右区域中。"""

        self._validate_coffee_id(coffee_id)
        region = self.dining_regions_by_slot.get(str(slot))
        if not isinstance(region, Mapping):
            return False
        return is_xy_inside_region(self.get_object_world_position(coffee_id)[:2], region)

    def configure_dining_placement_regions(
        self,
        *,
        region_config: Mapping[str, Any],
        regions_by_slot: Mapping[str, Mapping[str, Any]],
        sampled_targets_by_slot: Mapping[str, Mapping[str, Any]],
        planned_slot_assignment: Mapping[str, str],
    ) -> None:
        """安装本 episode 固定的区域、目标采样和杯槽分配。"""

        required_slots = {"left", "right"}
        if set(regions_by_slot) != required_slots or set(sampled_targets_by_slot) != required_slots:
            raise ValueError("WA-01 dining placement requires left and right slots")
        assignment = {str(key): str(value) for key, value in planned_slot_assignment.items()}
        if set(assignment) != set(COFFEE_IDS) or set(assignment.values()) != required_slots:
            raise ValueError("WA-01 must assign exactly one coffee to each dining slot")
        regions = {slot: dict(regions_by_slot[slot]) for slot in required_slots}
        targets = {slot: dict(sampled_targets_by_slot[slot]) for slot in required_slots}
        for slot in required_slots:
            if not is_xy_inside_region(targets[slot].get("xy"), regions[slot]):
                raise ValueError(f"sampled target for {slot} is outside its region")
        self.dining_region_config = dict(region_config)
        self.dining_regions_by_slot = regions
        self.sampled_dining_targets_by_slot = targets
        self.planned_dining_slot_by_coffee = assignment
        refs = self._ep_meta.setdefault("refs", {})
        refs.update(
            {
                "dining_region_config": dict(region_config),
                "dining_regions_by_slot": regions,
                "sampled_dining_targets_by_slot": targets,
                "planned_dining_slot_by_coffee": assignment,
            }
        )

    def get_object_world_position(self, object_name: str) -> np.ndarray:
        """读取指定任务物品当前的世界坐标位置。"""

        return np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[object_name]],
            dtype=float,
        )

    def get_coffee_machine_pour_site_world_position(self) -> np.ndarray:
        """读取咖啡机接液位点当前的世界坐标。"""

        site_name = f"{self.coffee_machine.naming_prefix}receptacle_place_site"
        return np.asarray(
            self.sim.data.site_xpos[self.sim.model.site_name2id(site_name)],
            dtype=float,
        )

    def get_ep_meta(self) -> dict[str, Any]:
        """补充自然语言任务描述。"""

        from bench.tasks.wa01.prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        return metadata

    # Authoritative task result -----------------------------------------

    def _check_success(self) -> bool:
        """刷新状态并委托语义层给出唯一最终成功结果。"""

        self.update_state()
        return evaluate_success(self._semantic_state).success

    def task_diagnostics(self) -> dict[str, object]:
        """输出成功条件、杯子物理状态、区域占用和最终选择诊断。"""

        self.update_state()
        report = evaluate_success(self._semantic_state)
        occupancy = self._dining_region_occupancy()
        return {
            "success": report.success,
            "success_checks": report.checks(),
            "sugar": {
                "consumed": self.sugar_consumed,
                "sugared_coffee_id": self.sugared_coffee_id,
                "unsugared_coffee_id": self.unsugared_coffee_id,
            },
            **{
                coffee_id: {
                    "on_dining_counter": bool(
                        getattr(self, f"{coffee_id}_on_dining_counter", False)
                    ),
                    "tilt_degrees": self._cup_tilt_degrees(coffee_id),
                    "upright": bool(getattr(self, f"{coffee_id}_upright", False)),
                    "dining_regions": [
                        slot for slot, values in occupancy.items() if coffee_id in values
                    ],
                }
                for coffee_id in COFFEE_IDS
            },
            "coffee_pair": {
                "xy_distance": self._cup_xy_distance(),
                "minimum_xy_separation": self.minimum_cup_xy_separation,
                "separate_regions_occupied": bool(self.dining_regions_occupied),
                "together": bool(self.cups_together),
                "currently_together": bool(self.cups_together),
                "together_before_selection": self.cups_together_before_selection,
            },
            "dining_placement": {
                "region_config": dict(self.dining_region_config),
                "regions_by_slot": dict(self.dining_regions_by_slot),
                "sampled_targets_by_slot": dict(self.sampled_dining_targets_by_slot),
                "planned_slot_by_coffee": dict(self.planned_dining_slot_by_coffee),
                "occupancy": occupancy,
            },
            "selection": {
                "selected_coffee_id": self.selected_sugared_coffee_id,
                "correct": report.correct_sugared_selection,
            },
        }

    # Private physical and visual helpers -------------------------------

    def _is_cup_on_dining_counter(self, coffee_id: str) -> bool:
        """检查杯子几何体当前是否与餐桌 fixture 接触。"""

        return bool(OU.check_obj_fixture_contact(self, coffee_id, self.dining_counter))

    def _cup_tilt_degrees(self, coffee_id: str) -> float:
        """根据杯子局部 Z 轴计算相对世界竖直方向的当前倾角。"""

        rotation = np.asarray(
            self.sim.data.body_xmat[self.obj_body_id[coffee_id]],
            dtype=float,
        ).reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(rotation[:, 2][2], -1.0, 1.0))))

    def _is_cup_upright(self, coffee_id: str) -> bool:
        """判断杯子当前倾角是否不超过任务阈值。"""

        return self._cup_tilt_degrees(coffee_id) <= self.cup_upright_threshold_deg

    def _dining_region_occupancy(self) -> dict[str, list[str]]:
        """列出左右区域中当前各自包含的杯子。"""

        return {
            slot: [
                coffee_id
                for coffee_id in COFFEE_IDS
                if self.is_coffee_cup_center_inside_dining_region(coffee_id, slot)
            ]
            for slot in ("left", "right")
        }

    def _are_both_dining_regions_occupied(self) -> bool:
        """要求左右区域各有且仅有一只不同的咖啡杯。"""

        occupancy = self._dining_region_occupancy()
        coffees = occupancy["left"] + occupancy["right"]
        return bool(
            len(occupancy["left"]) == 1
            and len(occupancy["right"]) == 1
            and set(coffees) == set(COFFEE_IDS)
        )

    def _are_cups_correctly_arranged(self) -> bool:
        """综合接触、竖直、分区和最小间距判断当前双杯就位。"""

        return bool(
            all(
                self.is_coffee_cup_correctly_placed_on_dining_counter(coffee_id)
                for coffee_id in COFFEE_IDS
            )
            and self._are_both_dining_regions_occupied()
            and self._cup_xy_distance() >= self.minimum_cup_xy_separation
        )

    def _cup_xy_distance(self) -> float:
        """计算两只杯子中心在世界 XY 平面的距离。"""

        return float(
            np.linalg.norm(
                self.get_object_world_position("coffee_A")[:2]
                - self.get_object_world_position("coffee_B")[:2]
            )
        )

    def _hide_consumed_sugar_cube(self) -> None:
        """将已消费糖块移出可见和可碰撞的任务区域。"""

        obj = self.objects["sugar_cube"]
        qpos = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float)
        qpos[:3] = np.asarray([10.0, 10.0, -1.0], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()

    def _set_coffee_liquid_visual(
        self,
        coffee_id: str,
        *,
        visible: bool,
        rgba: tuple[float, float, float, float] = NORMAL_COFFEE_RGBA,
    ) -> None:
        """设置杯内液体 site 的可见性和 RGBA 颜色。"""

        color = list(rgba)
        color[3] = float(color[3]) if visible else 0.0
        OU.add_obj_liquid_site(self, coffee_id, color)

    @staticmethod
    def _validate_coffee_id(coffee_id: str) -> None:
        """校验环境接口接收到的杯子标识是否合法。"""

        if str(coffee_id) not in COFFEE_IDS:
            raise ValueError(f"invalid WA-01 coffee id: {coffee_id!r}")


__all__ = ["WA01SortSugaredAndUnsugaredCoffee"]
