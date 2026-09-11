"""WA-03 的 RoboCasa 场景与物理观察边界。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.wa03.domain.contracts import COFFEE_IDS
from bench.tasks.wa03.domain.geometry import (
    DINING_REGION_SLOTS,
    DiningPlacementRegionConfig,
    compute_dining_regions,
    dining_region_config_summary,
    is_xy_inside_region,
)
from bench.tasks.wa03.domain.semantics import (
    SemanticState,
    evaluate_success,
    latch_all_cups_ready,
    record_selection,
    record_sugar_delivery,
)
from bench.tasks.wa03.evaluation import WA03EvaluationSnapshot, WA03Evaluator


COFFEE_MUG_XML = f"{assets_root}/objects/objaverse/cup/cup_2/model.xml"
SUGAR_PLATE_XML = f"{assets_root}/objects/objaverse/plate/plate_3/model.xml"
SUGAR_CUBE_XML = f"{assets_root}/objects/lightwheel/sugar_cube/SugarCube003/model.xml"
COFFEE_MUG_OBJECT_SCALE = (0.75, 0.75, 0.85)
NORMAL_COFFEE_RGBA = (0.65, 0.45, 0.25, 0.8)
TARGET_MARKER_RGBA = (0.20, 0.85, 0.25, 0.95)
OVERHEAD_CABINET_FIXTURES = ("cab_1_left_group", "cab_2_left_group")
MIN_FINAL_CUP_XY_SEPARATION = 0.10
SUGAR_RELEASE_DISTANCE = 0.15
FINAL_SELECTION_MAX_EEF_DISTANCE = 0.18
FINAL_SELECTION_MIN_LIFT = 0.02
FINAL_SELECTION_RELEASE_DISTANCE = 0.18


class WA03SortSugaredAndUnsugaredCoffee(Kitchen):
    """承载 WA-03 场景与物理观察，流程和执行逻辑位于类外。"""

    def __init__(
        self,
        minimum_cup_xy_separation: float = MIN_FINAL_CUP_XY_SEPARATION,
        cup_upright_threshold_deg: float = 15.0,
        sugar_target_mode: str = "unmarked",
        target_sugared_coffee_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """保存任务阈值和初始语义状态，再交由 Kitchen 构建环境。"""

        self.minimum_cup_xy_separation = float(minimum_cup_xy_separation)
        self.cup_upright_threshold_deg = float(cup_upright_threshold_deg)
        self._semantic_state = SemanticState(
            sugar_target_mode=sugar_target_mode,
            target_sugared_coffee_id=target_sugared_coffee_id,
        )
        self._configured_sugar_target_mode = self._semantic_state.sugar_target_mode
        self._configured_target_sugared_coffee_id = (
            self._semantic_state.target_sugared_coffee_id
        )
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
        """重置语义状态、餐桌区域、物理缓存和目标视觉标记。"""

        self._last_task_event_timestep = -1
        super()._setup_scene()
        self.coffee_machine._turned_on = False
        self._semantic_state = SemanticState(
            sugar_target_mode=self._configured_sugar_target_mode,
            target_sugared_coffee_id=self._configured_target_sugared_coffee_id,
        )
        default_regions = DiningPlacementRegionConfig()
        self.dining_region_config = dining_region_config_summary(default_regions)
        self.dining_regions_by_slot = compute_dining_regions(default_regions)
        self.sampled_dining_targets_by_slot: dict[str, dict[str, Any]] = {}
        self.planned_dining_slot_by_coffee = {
            "coffee_A": "left",
            "coffee_B": "center",
            "coffee_C": "right",
        }
        for coffee_id in COFFEE_IDS:
            setattr(self, f"{coffee_id}_on_dining_counter", False)
            setattr(self, f"{coffee_id}_upright", False)
        self.dining_regions_occupied = False
        self.all_cups_ready = False
        self._selection_ready_positions_by_coffee: dict[str, list[float]] = {}
        self._selection_armed = False
        self._physical_event_diagnostics: dict[str, Any] = {
            "sugar_delivery": None,
            "cups_ready_latched": False,
            "cups_ready_released": False,
            "selection_armed": False,
            "selection": None,
        }
        self._wa03_evaluator: WA03Evaluator | None = None
        self.target_marker_visible = False
        for coffee_id in COFFEE_IDS:
            self._set_coffee_liquid_visual(coffee_id, visible=True)
        self._set_target_coffee_marker_visibility(
            self.sugar_target_mode == "marked"
        )

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        """定义三只同款杯子、糖碟和糖块的 RoboCasa 物品配置。"""

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
                    pos=("ref", -0.135),
                    rotation=[np.pi / 4, np.pi / 2],
                ),
            ),
            dict(
                name="coffee_C",
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
    def sugar_target_mode(self) -> str:
        """返回当前 episode 使用的 unmarked 或 marked 模式。"""

        return self._semantic_state.sugar_target_mode

    @property
    def target_sugared_coffee_id(self) -> str | None:
        """返回 marked 模式的目标杯；unmarked 模式返回 None。"""

        return self._semantic_state.target_sugared_coffee_id

    @property
    def sugar_consumed(self) -> bool:
        """返回糖块是否已经被语义事件消费。"""

        return self._semantic_state.sugar_consumed

    @property
    def sugared_coffee_id(self) -> str | None:
        """返回实际收到糖块的杯子标识。"""

        return self._semantic_state.sugared_coffee_id

    @property
    def unsugared_coffee_ids(self) -> tuple[str, ...]:
        """返回没有收到糖块的另外两只杯子标识。"""

        return self._semantic_state.unsugared_coffee_ids

    @property
    def all_cups_ready_before_selection(self) -> bool:
        """返回最终抓杯前是否曾锁存三杯正确就位。"""

        return self._semantic_state.all_cups_ready_before_selection

    @property
    def selected_sugared_coffee_id(self) -> str | None:
        """返回最终被选择并抓起的杯子标识。"""

        return self._semantic_state.selected_sugared_coffee_id

    @property
    def sugar_assignment_correct(self) -> bool:
        """返回 marked 目标是否满足；unmarked 模式不施加该约束。"""

        value = evaluate_success(self._semantic_state).sugar_assignment_correct
        return True if value is None else bool(value)

    @property
    def correct_sugared_selection(self) -> bool:
        """返回最终选择是否指向实际加糖杯。"""

        return evaluate_success(self._semantic_state).correct_sugared_selection

    def record_sugar_delivery(self, coffee_id: str) -> None:
        """记录糖投递事件，并隐藏糖块和目标提示颜色。"""

        self._commit_sugar_delivery(coffee_id, source="expert_handler")
        self.update_state()
        self._note_evaluator_event("sugar_delivery")

    def record_all_cups_ready_before_selection(self) -> bool:
        """刷新物理状态并锁存三杯已正确放好的历史条件。"""

        self.update_state()
        return latch_all_cups_ready(
            self._semantic_state,
            currently_ready=self.all_cups_ready,
        )

    def record_sugared_coffee_selection(self, coffee_id: str) -> None:
        """记录最终选杯事件并刷新派生语义结果。"""

        record_selection(self._semantic_state, coffee_id)
        self.update_state()
        self._note_evaluator_event("selection")

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
        self.dining_regions_occupied = self._are_all_dining_regions_occupied()
        self.all_cups_ready = self._are_cups_correctly_arranged()
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
        """检查指定杯子中心是否位于给定左、中、右区域中。"""

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

        required_slots = set(DINING_REGION_SLOTS)
        if set(regions_by_slot) != required_slots or set(sampled_targets_by_slot) != required_slots:
            raise ValueError("WA-03 dining placement requires left, center, and right slots")
        assignment = {str(key): str(value) for key, value in planned_slot_assignment.items()}
        if set(assignment) != set(COFFEE_IDS) or set(assignment.values()) != required_slots:
            raise ValueError("WA-03 must assign exactly one of three coffees to each dining slot")
        regions = {slot: dict(regions_by_slot[slot]) for slot in DINING_REGION_SLOTS}
        targets = {slot: dict(sampled_targets_by_slot[slot]) for slot in DINING_REGION_SLOTS}
        for slot in DINING_REGION_SLOTS:
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
        """补充自然语言任务描述和目标杯 metadata。"""

        from bench.tasks.wa03.prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(
            sugar_target_mode=self.sugar_target_mode,
            variant="memory",
        )
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        if self.target_sugared_coffee_id is not None:
            metadata.setdefault("refs", {})["target_sugared_coffee_id"] = (
                self.target_sugared_coffee_id
            )
        return metadata

    # Authoritative task result -----------------------------------------

    def _check_success(self) -> bool:
        """Use the attached evaluator after runtime initialization."""

        evaluator = self._wa03_evaluator
        if evaluator is not None:
            return evaluator.observe(self.evaluation_snapshot()).success
        self.update_state()
        return evaluate_success(self._semantic_state).success

    def task_diagnostics(self) -> dict[str, object]:
        """Return evaluator diagnostics, with a construction-time fallback."""

        snapshot = self.evaluation_snapshot()
        evaluator = self._wa03_evaluator
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

    def evaluation_snapshot(self) -> WA03EvaluationSnapshot:
        """Expose raw physical and detected-event facts to the evaluator."""

        self.update_state()
        report = evaluate_success(self._semantic_state)
        occupancy = self._dining_region_occupancy()
        slot_by_coffee = {
            coffee_id: next(
                (slot for slot, values in occupancy.items() if coffee_id in values),
                None,
            )
            for coffee_id in COFFEE_IDS
        }
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
                "target_mode": self.sugar_target_mode,
                "target_coffee_id": self.target_sugared_coffee_id,
                "consumed": self.sugar_consumed,
                "sugared_coffee_id": self.sugared_coffee_id,
                "unsugared_coffee_ids": list(self.unsugared_coffee_ids),
                "assignment_correct": report.sugar_assignment_correct,
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
            "coffee_set": {
                "pairwise_xy_distances": self._cup_pairwise_xy_distances(),
                "minimum_xy_separation": self.minimum_cup_xy_separation,
                "separate_regions_occupied": bool(self.dining_regions_occupied),
                "all_ready": bool(self.all_cups_ready),
                "currently_all_ready": bool(self.all_cups_ready),
                "together_before_selection": self.all_cups_ready_before_selection,
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
        return WA03EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            sugar_target_mode=self.sugar_target_mode,
            target_sugared_coffee_id=self.target_sugared_coffee_id,
            sugar_delivery_coffee_id=self.sugared_coffee_id,
            coffee_on_dining_counter=on_counter,
            coffee_upright=upright,
            coffee_in_dining_region=slot_by_coffee,
            selected_coffee_id=self.selected_sugared_coffee_id,
            details=details,
        )

    # Private physical and visual helpers -------------------------------

    def _update_task_events_from_physics(self) -> None:
        """为 expert 和纯动作重放共同锁存可由物理状态验证的事件。"""

        delivered_to = self._physically_delivered_sugar_coffee_id()
        if delivered_to is not None:
            self._commit_sugar_delivery(delivered_to, source="physical_detection")

        just_latched = False
        if self.sugar_consumed and self.all_cups_ready:
            was_latched = self.all_cups_ready_before_selection
            latch_all_cups_ready(self._semantic_state, currently_ready=True)
            just_latched = not was_latched and self.all_cups_ready_before_selection
            if just_latched:
                self._selection_ready_positions_by_coffee = {
                    coffee_id: self.get_object_world_position(coffee_id)
                    .astype(float)
                    .tolist()
                    for coffee_id in COFFEE_IDS
                }
                self._physical_event_diagnostics["cups_ready_latched"] = True

        if not self.all_cups_ready_before_selection:
            return
        if self.selected_sugared_coffee_id is not None:
            return

        if not self._selection_armed:
            if (
                just_latched
                or not self.all_cups_ready
                or not self._are_all_cups_released_for_selection()
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
        """提交一次投糖事件，并撤销目标杯的临时视觉提示。"""

        self._validate_coffee_id(coffee_id)
        if not record_sugar_delivery(self._semantic_state, coffee_id):
            return False
        self._physical_event_diagnostics["sugar_delivery"] = {
            "coffee_id": coffee_id,
            "source": str(source),
        }
        self._hide_consumed_sugar_cube()
        if self.sugar_target_mode == "marked":
            self._set_target_coffee_marker_visibility(False)
        return True

    def _note_evaluator_event(self, name: str) -> None:
        """在专家手动提交事件后推进 evaluator,使步骤级诊断立即可用。

        学习策略复用同一 evaluator,其事件经 ``update_state`` 里的物理检测
        触发;此处仅补上专家执行路径在事件提交时缺失的 observe 推进。
        """

        evaluator = self._wa03_evaluator
        if evaluator is not None:
            evaluator.observe(self.evaluation_snapshot())

    def _physically_delivered_sugar_coffee_id(self) -> str | None:
        """返回唯一包含已释放糖块的杯子，否则不产生投糖事件。"""

        if self.sugar_consumed:
            return None
        if not OU.gripper_obj_far(self, "sugar_cube", th=SUGAR_RELEASE_DISTANCE):
            return None
        containing = [
            coffee_id
            for coffee_id in COFFEE_IDS
            if OU.check_obj_in_receptacle(self, "sugar_cube", coffee_id)
        ]
        return containing[0] if len(containing) == 1 else None

    def _physically_selected_coffee_id(self) -> str | None:
        """检测三杯就位后第一次被可靠抓持并抬起的唯一杯子。"""

        selected = [
            coffee_id
            for coffee_id in COFFEE_IDS
            if self._is_coffee_held_after_ready(coffee_id)
        ]
        return selected[0] if len(selected) == 1 else None

    def _are_all_cups_released_for_selection(self) -> bool:
        """确认夹爪已离开全部三杯，后续抓取才可视为最终选择。"""

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
        """列出左、中、右区域中当前各自包含的杯子。"""

        return {
            slot: [
                coffee_id
                for coffee_id in COFFEE_IDS
                if self.is_coffee_cup_center_inside_dining_region(coffee_id, slot)
            ]
            for slot in DINING_REGION_SLOTS
        }

    def _are_all_dining_regions_occupied(self) -> bool:
        """要求左、中、右区域各有且仅有一只不同的咖啡杯。"""

        occupancy = self._dining_region_occupancy()
        coffees = [coffee_id for values in occupancy.values() for coffee_id in values]
        return bool(
            all(len(occupancy[slot]) == 1 for slot in DINING_REGION_SLOTS)
            and set(coffees) == set(COFFEE_IDS)
        )

    def _are_cups_correctly_arranged(self) -> bool:
        """判断三杯是否都接触餐桌并保持直立。

        Placement-region occupancy is retained as a diagnostic and expert
        recording check, but is intentionally excluded from policy evaluation.
        """

        return bool(
            all(
                self.is_coffee_cup_correctly_placed_on_dining_counter(coffee_id)
                for coffee_id in COFFEE_IDS
            )
        )

    def _cup_pairwise_xy_distances(self) -> dict[str, float]:
        """计算三只杯子中心在世界 XY 平面的全部两两距离。"""

        return {
            f"{first}:{second}": float(
                np.linalg.norm(
                    self.get_object_world_position(first)[:2]
                    - self.get_object_world_position(second)[:2]
                )
            )
            for index, first in enumerate(COFFEE_IDS)
            for second in COFFEE_IDS[index + 1 :]
        }

    def _hide_consumed_sugar_cube(self) -> None:
        """将已消费糖块移出可见和可碰撞的任务区域。"""

        obj = self.objects["sugar_cube"]
        qpos = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float)
        qpos[:3] = np.asarray([10.0, 10.0, -1.0], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()

    def _set_target_coffee_marker_visibility(self, visible: bool) -> None:
        """仅在 marked 兼容模式下显示目标杯提示颜色。"""

        show_marker = bool(
            visible
            and self.sugar_target_mode == "marked"
            and self.target_sugared_coffee_id is not None
        )
        self.target_marker_visible = show_marker
        for coffee_id in COFFEE_IDS:
            rgba = (
                TARGET_MARKER_RGBA
                if show_marker and coffee_id == self.target_sugared_coffee_id
                else NORMAL_COFFEE_RGBA
            )
            self._set_coffee_liquid_visual(coffee_id, visible=True, rgba=rgba)

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
            raise ValueError(f"invalid WA-03 coffee id: {coffee_id!r}")


__all__ = ["WA03SortSugaredAndUnsugaredCoffee"]
