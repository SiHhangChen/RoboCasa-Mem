"""WA-01 的 RoboCasa 场景与物理观察边界。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.wa01.domain.contracts import COFFEE_IDS
from bench.tasks.wa01.materials import (
    COFFEE_MATERIAL_PROFILES,
    SUGAR_CUBE_MATERIAL_PROFILES,
    apply_coffee_material,
    apply_object_material,
)
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
from bench.tasks.wa01.evaluation import WA01EvaluationSnapshot, WA01Evaluator


COFFEE_MUG_XML = f"{assets_root}/objects/objaverse/cup/cup_2/model.xml"
SUGAR_PLATE_XML = f"{assets_root}/objects/objaverse/plate/plate_3/model.xml"
SUGAR_CUBE_XML = f"{assets_root}/objects/lightwheel/sugar_cube/SugarCube003/model.xml"
COFFEE_MUG_OBJECT_SCALE = (0.75, 0.75, 0.85)
NORMAL_COFFEE_RGBA = (0.65, 0.45, 0.25, 0.8)
OVERHEAD_CABINET_FIXTURES = ("cab_1_left_group", "cab_2_left_group")
MIN_FINAL_CUP_XY_SEPARATION = 0.20
SUGAR_RELEASE_DISTANCE = 0.15
FINAL_SELECTION_MAX_EEF_DISTANCE = 0.18
FINAL_SELECTION_MIN_LIFT = 0.02
FINAL_SELECTION_RELEASE_DISTANCE = 0.18


class WA01SortSugaredAndUnsugaredCoffee(Kitchen):
    """承载 WA-01 场景与物理观察，流程和执行逻辑位于类外。"""

    def __init__(
        self,
        minimum_cup_xy_separation: float = MIN_FINAL_CUP_XY_SEPARATION,
        cup_upright_threshold_deg: float = 15.0,
        coffee_material_by_id: Mapping[str, str] | None = None,
        sugar_cube_material: str = "original",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """保存任务阈值和初始语义状态，再交由 Kitchen 构建环境。"""

        self.minimum_cup_xy_separation = float(minimum_cup_xy_separation)
        self.cup_upright_threshold_deg = float(cup_upright_threshold_deg)
        self._semantic_state = SemanticState()
        materials = dict(coffee_material_by_id or {})
        if set(materials) - set(COFFEE_IDS):
            raise ValueError("WA-01 coffee materials name unknown cups")
        self.coffee_material_by_id = {
            coffee_id: str(materials.get(coffee_id, "dark_blue")) for coffee_id in COFFEE_IDS
        }
        unknown_materials = set(self.coffee_material_by_id.values()) - set(
            COFFEE_MATERIAL_PROFILES
        )
        if unknown_materials:
            raise ValueError(
                "unknown WA-01 coffee materials: " + ", ".join(sorted(unknown_materials))
            )
        cube_material = str(sugar_cube_material)
        if cube_material not in SUGAR_CUBE_MATERIAL_PROFILES:
            raise ValueError(f"unknown WA-01 sugar cube material: {cube_material!r}")
        self.sugar_cube_material = cube_material
        super().__init__(*args, **kwargs)

    # Scene construction -------------------------------------------------

    def edit_model_xml(self, xml_str: str) -> str:
        """移除遮挡柜体，并写入本 episode 抽到的杯子外观材质。"""

        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        if worldbody is not None and int(self.layout_id) == 29:
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
        for coffee_id, material_id in self.coffee_material_by_id.items():
            apply_coffee_material(
                root,
                coffee_id=coffee_id,
                material_id=material_id,
                assets_root=Path(assets_root),
            )
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

        self._wa01_evaluator: WA01Evaluator | None = None
        self._last_task_event_timestep = -1
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
        self._selection_ready_positions_by_coffee: dict[str, list[float]] = {}
        self._selection_armed = False
        self._physical_event_diagnostics: dict[str, Any] = {
            "sugar_delivery": None,
            "cups_ready_latched": False,
            "cups_ready_released": False,
            "selection_armed": False,
            "selection": None,
        }
        for coffee_id in COFFEE_IDS:
            self._set_coffee_liquid_visual(coffee_id, visible=True)
        self.material_randomization = {
            "sugar_cube": apply_object_material(
                self,
                "sugar_cube",
                self.sugar_cube_material,
                SUGAR_CUBE_MATERIAL_PROFILES,
            )
        }

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        """定义两只同款杯子、糖碟和糖块的 RoboCasa 物品配置。"""

        return [
            dict(
                name="coffee_A",
                obj_groups=COFFEE_MUG_XML,
                object_scale=COFFEE_MUG_OBJECT_SCALE,
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
                object_scale=COFFEE_MUG_OBJECT_SCALE,
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

        self._commit_sugar_delivery(coffee_id, source="expert_handler")
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
        """刷新物理状态，并把已验证的物理变化提交为共享任务事件。"""

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
        step_id = int(getattr(self, "timestep", 0))
        if step_id > self._last_task_event_timestep:
            self._update_task_events_from_physics()
            self._last_task_event_timestep = step_id

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
        """Use the attached evaluator after runtime initialization."""

        evaluator = self._wa01_evaluator
        if evaluator is not None:
            return evaluator.observe(self.evaluation_snapshot()).success
        self.update_state()
        return evaluate_success(self._semantic_state).success

    def task_diagnostics(self) -> dict[str, object]:
        """Return evaluator diagnostics, with a construction-time fallback."""

        snapshot = self.evaluation_snapshot()
        evaluator = self._wa01_evaluator
        if evaluator is not None:
            return evaluator.observe(snapshot).to_diagnostics()
        report = evaluate_success(self._semantic_state)
        return {
            **dict(snapshot.details),
            "success": report.success,
            "terminal": report.success,
            "failure_code": None,
            "success_checks": report.checks(),
        }

    def evaluation_snapshot(self) -> WA01EvaluationSnapshot:
        """Expose raw physical and detected-event facts to the evaluator."""

        self.update_state()
        report = evaluate_success(self._semantic_state)
        occupancy = self._dining_region_occupancy()
        on_counter = {
            coffee_id: bool(
                getattr(self, f"{coffee_id}_on_dining_counter", False)
            )
            for coffee_id in COFFEE_IDS
        }
        upright = {
            coffee_id: bool(getattr(self, f"{coffee_id}_upright", False))
            for coffee_id in COFFEE_IDS
        }
        details: dict[str, object] = {
            "sugar": {
                "consumed": self.sugar_consumed,
                "sugared_coffee_id": self.sugared_coffee_id,
                "unsugared_coffee_id": self.unsugared_coffee_id,
            },
            **{
                coffee_id: {
                    "on_dining_counter": on_counter[coffee_id],
                    "tilt_degrees": self._cup_tilt_degrees(coffee_id),
                    "upright": upright[coffee_id],
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
            "physical_events": {
                **dict(self._physical_event_diagnostics),
                "selection_ready_positions_by_coffee": dict(
                    self._selection_ready_positions_by_coffee
                ),
            },
        }
        return WA01EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            sugar_delivery_coffee_id=self.sugared_coffee_id,
            coffee_on_dining_counter=on_counter,
            coffee_upright=upright,
            selected_coffee_id=self.selected_sugared_coffee_id,
            details=details,
        )

    # Private physical and visual helpers -------------------------------

    def _update_task_events_from_physics(self) -> None:
        """为 expert 和 learned-policy rollout 共用地锁存物理任务事件。"""

        delivered_to = self._physically_delivered_sugar_coffee_id()
        if delivered_to is not None:
            self._commit_sugar_delivery(delivered_to, source="physical_detection")

        just_latched = False
        if self.sugar_consumed and self.cups_together:
            was_latched = self.cups_together_before_selection
            latch_cups_together(self._semantic_state, currently_together=True)
            just_latched = not was_latched and self.cups_together_before_selection
            if just_latched:
                self._selection_ready_positions_by_coffee = {
                    coffee_id: self.get_object_world_position(coffee_id)
                    .astype(float)
                    .tolist()
                    for coffee_id in COFFEE_IDS
                }
                self._physical_event_diagnostics["cups_ready_latched"] = True

        if not self.cups_together_before_selection:
            return
        if self.selected_sugared_coffee_id is not None:
            return

        # “两杯已放好”和“最终选杯”必须是两个先后发生的动作。只有在两杯
        # 仍正确就位、且夹爪已经完全离开两杯后，才开放后续选杯检测。
        if not self._selection_armed:
            if (
                just_latched
                or not self.cups_together
                or not self._are_both_cups_released_for_selection()
            ):
                return
            self._selection_armed = True
            self._physical_event_diagnostics["cups_ready_released"] = True
            self._physical_event_diagnostics["selection_armed"] = True
            return

        selected = self._physically_selected_coffee_id()
        if selected is not None:
            record_selection(self._semantic_state, selected)
            self._physical_event_diagnostics["selection"] = {
                "coffee_id": selected,
                "source": "physical_detection",
            }

    def _commit_sugar_delivery(self, coffee_id: str, *, source: str) -> bool:
        """提交一次投糖事件并隐藏已消费的糖块。"""

        self._validate_coffee_id(coffee_id)
        if not record_sugar_delivery(self._semantic_state, coffee_id):
            return False
        self._physical_event_diagnostics["sugar_delivery"] = {
            "coffee_id": coffee_id,
            "source": str(source),
        }
        self._hide_consumed_sugar_cube()
        return True

    def _physically_delivered_sugar_coffee_id(self) -> str | None:
        """返回唯一包含已释放糖块的杯子，否则不产生投糖事件。"""

        if self.sugar_consumed:
            return None
        if not OU.gripper_obj_far(
            self,
            "sugar_cube",
            th=SUGAR_RELEASE_DISTANCE,
        ):
            return None
        containing = [
            coffee_id
            for coffee_id in COFFEE_IDS
            if OU.check_obj_in_receptacle(self, "sugar_cube", coffee_id)
        ]
        return containing[0] if len(containing) == 1 else None

    def _physically_selected_coffee_id(self) -> str | None:
        """检测双杯就位后第一次被可靠抓持并抬起的唯一杯子。"""

        selected = [
            coffee_id
            for coffee_id in COFFEE_IDS
            if self._is_coffee_held_after_ready(coffee_id)
        ]
        return selected[0] if len(selected) == 1 else None

    def _are_both_cups_released_for_selection(self) -> bool:
        """确认双杯就位后夹爪已离开两杯，后续抓取才可视为选择。"""

        return all(
            OU.gripper_obj_far(
                self,
                coffee_id,
                th=FINAL_SELECTION_RELEASE_DISTANCE,
            )
            for coffee_id in COFFEE_IDS
        )

    def _is_coffee_held_after_ready(self, coffee_id: str) -> bool:
        """要求杯子被夹爪持有，且已离开就位表面或产生明确抬升。"""

        ready_position = self._selection_ready_positions_by_coffee.get(coffee_id)
        if ready_position is None:
            return False
        object_position = self.get_object_world_position(coffee_id)
        end_effector_position = np.asarray(
            self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]],
            dtype=float,
        )
        near_end_effector = bool(
            np.linalg.norm(object_position - end_effector_position)
            <= FINAL_SELECTION_MAX_EEF_DISTANCE
        )
        try:
            gripper_contact = bool(
                self.check_contact(
                    self.robots[0].gripper["right"],
                    self.objects[coffee_id],
                )
            )
        except Exception:
            gripper_contact = False
        try:
            grasped = bool(OU.check_obj_grasped(self, coffee_id, threshold=0.6))
        except Exception:
            grasped = False
        lifted = bool(
            object_position[2]
            >= float(ready_position[2]) + FINAL_SELECTION_MIN_LIFT
        )
        left_dining_surface = not bool(
            getattr(self, f"{coffee_id}_on_dining_counter", False)
        )
        return bool(
            near_end_effector
            and (gripper_contact or grasped)
            and (lifted or left_dining_surface)
        )

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
        """判断双杯是否都放上餐桌且保持直立（就位）。"""

        return bool(
            all(
                self.is_coffee_cup_correctly_placed_on_dining_counter(coffee_id)
                for coffee_id in COFFEE_IDS
            )
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
