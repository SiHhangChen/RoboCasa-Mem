"""TM03 canonical stove/drawer scene support and authoritative observer hook."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import robocasa.utils.object_utils as OU

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root

from bench.tasks.tm03.config import load_task_config, resolve_observer_config
from bench.tasks.tm03.domain.contracts import (
    AXIS,
    BURNER_ID,
    BURNER_SITE,
    CANONICAL_PROMPT,
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    KNOB_COLLISION_GEOM,
    KNOB_ID,
    KNOB_JOINT,
    EXPECTED_TOASTER_OVEN_ASSET,
    TOASTER_OVEN_FOOD_NAME,
    TOASTER_OVEN_TIME_COLLISION_GEOM,
    TOASTER_OVEN_TIME_JOINT,
    ObserverConfig,
)
from bench.tasks.tm03.domain.episode import (
    DEFAULT_COMPLETION_BUTTON_WORLD_XYZS,
    DEFAULT_DRAWER_WORLD_XYZS,
    DEFAULT_TOASTER_OVEN_FOOD_XML,
)
from bench.tasks.tm03.domain.semantics import (
    PhysicalState,
    diagnostics,
    evaluate_success,
)
from bench.tasks.tm03.physical_events import PhysicalSample, TM03PhysicalObserver
from bench.tasks.tm03.single_drawer import SingleTabletopDrawer
from bench.tasks.tm03.button_asset import TM03_BUTTON_MAX_TRAVEL_M, add_tm03_button_asset


DRAWER_FIXTURE_NAMES = {drawer: f"tm03_{drawer}_fixture" for drawer in DRAWER_NAMES}
COMPLETION_BUTTON_GEOM_NAMES = {
    name: f"bench_tm03_{name}" for name in COMPLETION_BUTTON_NAMES
}
COMPLETION_BUTTON_JOINT_NAMES = {
    name: f"{geom_name}_press_joint"
    for name, geom_name in COMPLETION_BUTTON_GEOM_NAMES.items()
}
COMPLETION_BUTTON_MAX_TRAVEL = TM03_BUTTON_MAX_TRAVEL_M
COMPLETION_BUTTON_RGBA = "0.22 0.72 0.32 1"
COMPLETION_BUTTON_GLOW_RGBA = "0.26 0.84 0.38 1"
DIAGNOSTIC_OVERVIEW_CAMERA = "bench_tm03_overview"
DIAGNOSTIC_DRAWER_CAMERA = "bench_tm03_drawer_overview"


class TM03StoveAndDrawersBase(Kitchen):
    """Two drawers, one native stove knob, and two equivalent completion buttons."""

    # Layout 39 deliberately uses the island stovetop. It is part of RoboCasa's
    # dining-counter exclusion set for unrelated placement tasks, but TM03 owns
    # all of its custom fixture poses and can safely use it.
    EXCLUDE_LAYOUTS: list[int] = []

    def __init__(
        self,
        *,
        drawer_names: tuple[str, str] = DRAWER_NAMES,
        drawer_world_xyzs: tuple[
            tuple[float, float, float], ...
        ] = DEFAULT_DRAWER_WORLD_XYZS,
        drawer_world_yaw_deg: float = 180.0,
        completion_button_world_xyzs: (
            Mapping[str, tuple[float, float, float]] | None
        ) = None,
        expected_stove_asset: str = "Stovetop057",
        knob_id: str = KNOB_ID,
        burner_id: str = BURNER_ID,
        pot_name: str = "pot",
        pot_xml: str = "objects/lightwheel/saucepan/Saucepan002/model.xml",
        observer_config: ObserverConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        names = tuple(str(name) for name in drawer_names)
        if names != DRAWER_NAMES:
            raise ValueError(f"TM03 drawer_names must equal {DRAWER_NAMES}")
        xyzs = tuple(tuple(float(value) for value in xyz) for xyz in drawer_world_xyzs)
        if len(xyzs) != 2 or any(len(xyz) != 3 for xyz in xyzs):
            raise ValueError("TM03 requires exactly two drawer xyz values")
        raw_buttons = dict(
            DEFAULT_COMPLETION_BUTTON_WORLD_XYZS
            if completion_button_world_xyzs is None
            else completion_button_world_xyzs
        )
        if tuple(raw_buttons) != COMPLETION_BUTTON_NAMES:
            raise ValueError(
                f"TM03 completion buttons must equal {COMPLETION_BUTTON_NAMES}"
            )
        button_xyzs: dict[str, tuple[float, float, float]] = {}
        for name in COMPLETION_BUTTON_NAMES:
            value = np.asarray(raw_buttons[name], dtype=float).reshape(-1)
            if value.size != 3 or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"TM03 completion button {name!r} requires a finite xyz"
                )
            button_xyzs[name] = tuple(float(item) for item in value)
        if observer_config is None:
            resolved_observer = resolve_observer_config(load_task_config())
        elif isinstance(observer_config, ObserverConfig):
            resolved_observer = observer_config
        else:
            resolved_observer = ObserverConfig(**dict(observer_config))
        self.drawer_names = (names[0], names[1])
        self.drawer_world_xyzs = xyzs
        self.drawer_world_yaw_deg = float(drawer_world_yaw_deg)
        self.completion_button_world_xyzs = button_xyzs
        self.expected_stove_asset = str(expected_stove_asset)
        if str(knob_id) != KNOB_ID or str(burner_id) != BURNER_ID:
            raise ValueError(
                "TM03 requires the front_right-knob / rear_left-burner binding"
            )
        self.pot_name = str(pot_name)
        self.pot_xml = str(pot_xml)
        self.knob = str(knob_id)
        self.burner = str(burner_id)
        self._tm03_observer = TM03PhysicalObserver(resolved_observer)
        self._observer_armed = False
        self._pot_reset_position = np.zeros(3, dtype=float)
        self._pot_reset_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
        super().__init__(**kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.storage_cabinets = {
            drawer: SingleTabletopDrawer(
                name=DRAWER_FIXTURE_NAMES[drawer],
                rng=self.rng,
            )
            for drawer in self.drawer_names
        }
        for drawer, fixture in self.storage_cabinets.items():
            model_name = DRAWER_FIXTURE_NAMES[drawer]
            self.fixtures[model_name] = fixture
            self.fixture_cfgs.append(
                {"name": model_name, "model": fixture, "type": "fixture"}
            )
        self.model.merge_objects(list(self.storage_cabinets.values()))

    def edit_model_xml(self, xml_str: str) -> str:
        """Add the two green physical completion plungers."""

        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_str
        existing = {
            geom.get("name") for geom in worldbody.iter("geom") if geom.get("name")
        }
        for name in COMPLETION_BUTTON_NAMES:
            geom_name = COMPLETION_BUTTON_GEOM_NAMES[name]
            if geom_name not in existing:
                _add_completion_button_asset(
                    worldbody,
                    button_name=name,
                    pos=self.completion_button_world_xyzs[name],
                )
        if root.find(f".//camera[@name='{DIAGNOSTIC_OVERVIEW_CAMERA}']") is None:
            ET.SubElement(
                worldbody,
                "camera",
                {
                    "name": DIAGNOSTIC_OVERVIEW_CAMERA,
                    "mode": "fixed",
                    "pos": "4.35 -4.20 2.70",
                    "xyaxes": "1 0 0 0 0.50 0.8660254",
                    "fovy": "58",
                },
            )
        if root.find(f".//camera[@name='{DIAGNOSTIC_DRAWER_CAMERA}']") is None:
            ET.SubElement(
                worldbody,
                "camera",
                {
                    "name": DIAGNOSTIC_DRAWER_CAMERA,
                    "mode": "fixed",
                    "pos": "4.35 -0.45 2.45",
                    "xyaxes": "-1 0 0 0 -0.50 0.8660254",
                    "fovy": "55",
                },
            )
        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.stove),
        )
        self.init_robot_base_ref = self.counter
        for drawer, fixture in self.storage_cabinets.items():
            self.fixture_refs[drawer] = (fixture, False)
        yaw = float(np.deg2rad(self.drawer_world_yaw_deg))
        for drawer, xyz in zip(
            self.drawer_names,
            self.drawer_world_xyzs,
            strict=True,
        ):
            self.storage_cabinets[drawer].set_world_pose(
                world_xyz=xyz,
                yaw=yaw,
            )

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        pot_path = Path(self.pot_xml)
        if not pot_path.is_absolute():
            pot_path = Path(assets_root) / pot_path
        return [
            dict(
                name=self.pot_name,
                obj_groups=str(pot_path),
                graspable=False,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self.burner]),
                    size=(0.05, 0.05),
                ),
            )
        ]

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._observer_armed = False
        for drawer in self.drawer_names:
            self.set_drawer_open_fraction(drawer, 0.0)
        # Direct fixture state writes are allowed only during reset.
        self.stove.set_knob_state(
            env=self,
            rng=self.rng,
            knob=self.knob,
            mode="off",
        )
        self.sim.forward()
        self._sync_bound_burner_visual()
        self._validate_native_stove_contract()

    def _reset_internal(self) -> None:
        """Arm TM03 only after RoboCasa places and settles the pot."""

        self._observer_armed = False
        super()._reset_internal()
        self.sim.forward()
        self._center_pot_on_bound_burner()
        self._sync_bound_burner_visual()
        self._validate_native_stove_contract()
        self._pot_reset_position = self.pot_world_position()
        self._pot_reset_quaternion = self.pot_world_quaternion()
        initial = self._physical_sample()
        self._tm03_observer.reset(initial)
        self._observer_armed = not bool(self._tm03_observer.invalid_environment_labels)

    def update_state(self) -> None:
        super().update_state()
        self._sync_bound_burner_visual()
        if self._observer_armed:
            self._update_tm03_observer()

    def reward(self, action=None):
        del action
        if self._observer_armed:
            self._update_tm03_observer()
        return 1.0 if self._check_success() else 0.0

    @property
    def semantic_state(self):
        return self._tm03_observer.state

    @property
    def physical_observer(self) -> TM03PhysicalObserver:
        return self._tm03_observer

    def cabinet_for_drawer(self, drawer: str) -> SingleTabletopDrawer:
        name = str(drawer)
        if name not in self.drawer_names:
            raise KeyError(f"unknown TM03 drawer {drawer!r}")
        return self.storage_cabinets[name]

    def cabinet_for_slot(self, fixture_name: str) -> SingleTabletopDrawer:
        """Expose TM03's task-owned contact primitive fixture boundary."""

        return self.cabinet_for_drawer(fixture_name)

    def drawer_open_fraction(self, drawer: str) -> float:
        return float(self.cabinet_for_drawer(drawer).open_fraction(self))

    def set_drawer_open_fraction(self, drawer: str, fraction: float) -> None:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError("TM03 drawer fraction must be in [0, 1]")
        fixture = self.cabinet_for_drawer(drawer)
        if value <= self._tm03_observer.config.drawer_closed_threshold:
            fixture.close_drawer(env=self, min=value, max=value)
        else:
            fixture.open_drawer(env=self, min=value, max=value)
        self.sim.forward()

    def completion_button_position(self, button_name: str) -> np.ndarray:
        name = _validate_completion_button_name(button_name)
        try:
            geom_id = int(
                self.sim.model.geom_name2id(COMPLETION_BUTTON_GEOM_NAMES[name])
            )
            return np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
        except Exception:
            return np.asarray(
                self.completion_button_world_xyzs[name], dtype=float
            ) + np.asarray(
                [0.0, 0.0, 0.075],
                dtype=float,
            )

    def completion_button_press_depth(self, button_name: str) -> float:
        name = _validate_completion_button_name(button_name)
        try:
            value = -float(
                self.sim.data.get_joint_qpos(COMPLETION_BUTTON_JOINT_NAMES[name])
            )
            return min(COMPLETION_BUTTON_MAX_TRAVEL, max(0.0, value))
        except Exception:
            return 0.0

    def native_knob_joint_name(self) -> str:
        joint = self.stove.knob_joints.get(self.knob)
        if joint is None or not joint.get("name"):
            raise RuntimeError(f"TM03 {self.knob} stove knob joint is missing")
        return str(joint.get("name"))

    def native_knob_geom_name(self) -> str:
        return f"{self.stove.naming_prefix}{KNOB_COLLISION_GEOM}"

    def native_knob_qpos(self) -> float:
        return float(self.sim.data.get_joint_qpos(self.native_knob_joint_name()))

    def native_knob_qvel(self) -> float:
        return float(self.sim.data.get_joint_qvel(self.native_knob_joint_name()))

    def burner_world_position(self) -> np.ndarray:
        site = self.stove.burner_sites.get(self.burner)
        if site is None or not site.get("name"):
            raise RuntimeError(f"TM03 {self.burner} burner site is missing")
        return np.asarray(
            self.sim.data.get_site_xpos(str(site.get("name"))),
            dtype=float,
        )

    def pot_world_position(self) -> np.ndarray:
        return np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[self.pot_name]],
            dtype=float,
        ).copy()

    def pot_world_quaternion(self) -> np.ndarray:
        return np.asarray(
            self.sim.data.body_xquat[self.obj_body_id[self.pot_name]],
            dtype=float,
        ).copy()

    def bound_burner_on(self) -> bool:
        """Whether the right-front control knob energizes the bound rear-left burner."""

        return bool(self.stove.is_burner_on(env=self, burner_loc=self.knob))

    def _center_pot_on_bound_burner(self) -> None:
        """Deterministically center the non-graspable pot during reset."""

        current = self.pot_world_position()
        target = current.copy()
        target[:2] = self.burner_world_position()[:2]
        obj = self.objects[self.pot_name]
        qpos = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
        qpos[:3] += target - current
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()

    def _sync_bound_burner_visual(self) -> None:
        """Render the rear-left flame from the physically rotated front-right knob."""

        is_on = self.bound_burner_on()
        for location in {self.knob, self.burner}:
            site = self.stove.burner_sites.get(location)
            if site is None or not site.get("name"):
                continue
            site_id = int(self.sim.model.site_name2id(str(site.get("name"))))
            self.sim.model.site_rgba[site_id][3] = (
                0.5 if location == self.burner and is_on else 0.0
            )

    def pot_stable_on_burner(self) -> bool:
        position = self.pot_world_position()
        translation = float(np.linalg.norm(position - self._pot_reset_position))
        burner_xy = self.burner_world_position()[:2]
        burner_distance = float(np.linalg.norm(position[:2] - burner_xy))
        current_quat = self.pot_world_quaternion()
        dot = float(
            np.clip(
                abs(np.dot(current_quat, self._pot_reset_quaternion)),
                0.0,
                1.0,
            )
        )
        tilt_deg = math.degrees(2.0 * math.acos(dot))
        config = self._tm03_observer.config
        return bool(
            translation <= config.pot_translation_guard_m
            and burner_distance <= config.pot_burner_xy_threshold_m
            and tilt_deg <= config.pot_tilt_guard_deg
        )

    def _physical_sample(self) -> PhysicalSample:
        knob_geom = self.native_knob_geom_name()
        gripper_contact = False
        any_robot_contact = False
        try:
            gripper = self.robots[0].gripper["right"]
            gripper_contact = bool(self.check_contact(gripper, [knob_geom]))
        except Exception:
            pass
        try:
            any_robot_contact = bool(
                self.check_contact(self.robots[0].robot_model, [knob_geom])
            )
        except Exception:
            pass
        any_robot_contact = any_robot_contact or gripper_contact
        other_burners_on = any(
            bool(self.stove.is_burner_on(env=self, burner_loc=location))
            for location in self.stove.get_knobs_state(self)
            if location != self.knob
        )
        return PhysicalSample(
            step_id=int(getattr(self, "timestep", 0)),
            knob_qpos_rad=self.native_knob_qpos(),
            knob_qvel_rad_s=self.native_knob_qvel(),
            knob_gripper_contact=gripper_contact,
            knob_any_robot_contact=any_robot_contact,
            target_burner_on=bool(
                self.bound_burner_on()
            ),
            other_burners_on=other_burners_on,
            drawer_open_fraction={
                drawer: self.drawer_open_fraction(drawer)
                for drawer in self.drawer_names
            },
            completion_button_press_depth_m={
                name: self.completion_button_press_depth(name)
                for name in COMPLETION_BUTTON_NAMES
            },
            pot_stable_on_burner=self.pot_stable_on_burner(),
            knob_actuator_commanded=self._knob_actuator_commanded(),
        )

    def _update_tm03_observer(self) -> None:
        self._tm03_observer.observe(self._physical_sample())

    def _knob_actuator_commanded(self) -> bool:
        native_joint = self.native_knob_joint_name()
        candidates = (
            f"{native_joint}_actuator",
            f"{KNOB_JOINT}_actuator",
        )
        for name in candidates:
            try:
                actuator_id = int(self.sim.model.actuator_name2id(name))
                if abs(float(self.sim.data.ctrl[actuator_id])) > 1e-8:
                    return True
            except Exception:
                continue
        return False

    def physical_state(self) -> PhysicalState:
        sample = self._physical_sample()
        return PhysicalState(
            knob_qpos_rad=sample.knob_qpos_rad,
            target_burner_on=sample.target_burner_on,
            all_other_burners_off=not sample.other_burners_on,
            drawer_open_fraction=dict(sample.drawer_open_fraction),
            pot_stable_on_burner=sample.pot_stable_on_burner,
        )

    def _check_success(self) -> bool:
        if self._tm03_observer.invalid_environment_labels:
            return False
        return evaluate_success(
            self.semantic_state,
            self.physical_state(),
            off_commit_max_rad=self._tm03_observer.config.off_commit_max_rad,
            drawer_closed_threshold=(
                self._tm03_observer.config.drawer_closed_threshold
            ),
        ).success

    def task_diagnostics(self) -> dict[str, Any]:
        state = diagnostics(self.semantic_state)
        state.update(
            {
                "target_knob_qpos_rad": self.native_knob_qpos(),
                "target_knob_qvel_rad_s": self.native_knob_qvel(),
                "target_burner_on": bool(
                    self.bound_burner_on()
                ),
                "green_button_press_depth_m": {
                    name: self.completion_button_press_depth(name)
                    for name in COMPLETION_BUTTON_NAMES
                },
                "invalid_environment_labels": list(
                    self._tm03_observer.invalid_environment_labels
                ),
                "success": self._check_success(),
            }
        )
        return state

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.tm03.prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        metadata.setdefault("refs", {}).update(
            {
                "axis": AXIS,
                "drawer_names": list(self.drawer_names),
                "knob_id": self.knob,
                "knob_joint": self.native_knob_joint_name(),
                "knob_collision_geom": self.native_knob_geom_name(),
                "burner_id": self.burner,
                "burner_site": (self.stove.burner_sites[self.burner].get("name")),
                "completion_button_geoms": dict(COMPLETION_BUTTON_GEOM_NAMES),
                "completion_button_joints": dict(COMPLETION_BUTTON_JOINT_NAMES),
            }
        )
        return metadata

    def _validate_native_stove_contract(self) -> None:
        actual_asset = Path(self.stove.file).parent.name
        if actual_asset != self.expected_stove_asset:
            raise RuntimeError(
                "TM03 expected stove asset "
                f"{self.expected_stove_asset}, found {actual_asset}"
            )
        joint_name = self.native_knob_joint_name()
        self.sim.model.get_joint_qpos_addr(joint_name)
        self.sim.model.geom_name2id(self.native_knob_geom_name())
        site = self.stove.burner_sites.get(self.burner)
        if site is None or not str(site.get("name", "")).endswith(BURNER_SITE):
            raise RuntimeError(
                f"TM03 Stovetop057 {self.burner} burner site is missing"
            )


def _add_completion_button_asset(
    parent: ET.Element,
    *,
    button_name: str,
    pos: tuple[float, float, float],
) -> None:
    name = _validate_completion_button_name(button_name)
    geom_name = COMPLETION_BUTTON_GEOM_NAMES[name]
    joint_name = COMPLETION_BUTTON_JOINT_NAMES[name]
    add_tm03_button_asset(
        parent,
        geom_name=geom_name,
        pos=pos,
        color_rgba=COMPLETION_BUTTON_RGBA,
        glow_rgba=COMPLETION_BUTTON_GLOW_RGBA,
        joint_name=joint_name,
        gravity_compensation=True,
    )


def _validate_completion_button_name(button_name: str) -> str:
    name = str(button_name)
    if name not in COMPLETION_BUTTON_NAMES:
        raise KeyError(f"unknown TM03 completion button {button_name!r}")
    return name

class TM03BoilWaterInspectDrawersAndRunToasterOven(
    TM03StoveAndDrawersBase
):
    """Three interleaved task threads with native physical controls."""

    EXCLUDE_LAYOUTS: list[int] = []

    def __init__(
        self,
        *,
        expected_toaster_oven_asset: str = EXPECTED_TOASTER_OVEN_ASSET,
        toaster_oven_food_name: str = TOASTER_OVEN_FOOD_NAME,
        toaster_oven_food_xml: str = DEFAULT_TOASTER_OVEN_FOOD_XML,
        toaster_oven_time_joint: str = TOASTER_OVEN_TIME_JOINT,
        toaster_oven_time_collision_geom: str = TOASTER_OVEN_TIME_COLLISION_GEOM,
        observer_config: ObserverConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if observer_config is None:
            resolved = resolve_observer_config(load_task_config())
        elif isinstance(observer_config, ObserverConfig):
            resolved = observer_config
        else:
            resolved = ObserverConfig(**dict(observer_config))
        self.expected_toaster_oven_asset = str(expected_toaster_oven_asset)
        self.toaster_oven_food_name = str(toaster_oven_food_name)
        self.toaster_oven_food_xml = str(toaster_oven_food_xml)
        self.toaster_oven_time_joint_suffix = str(toaster_oven_time_joint)
        self.toaster_oven_time_collision_geom_suffix = str(
            toaster_oven_time_collision_geom
        )
        self._tm03_observer = TM03PhysicalObserver(resolved)
        self._tm03_observer_armed = False
        # The parent contributes only the frozen A/B scene and physical helpers.
        # Its observer is not used by TM03.
        super().__init__(observer_config=None, **kwargs)

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        cfgs = list(super()._get_obj_cfgs())
        food_path = Path(self.toaster_oven_food_xml)
        if not food_path.is_absolute():
            food_path = Path(assets_root) / food_path
        cfgs.append(
            dict(
                name=self.toaster_oven_food_name,
                obj_groups=str(food_path),
                graspable=False,
                placement=dict(
                    fixture=self.toaster_oven,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )
        return cfgs

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._tm03_observer_armed = False
        self.toaster_oven.close_door(env=self)
        # Direct joint writes are reset-only. Episode transitions must come
        # from gripper contact with the native timer knob.
        self.toaster_oven.set_time(env=self, val=0.0)
        self.sim.forward()
        self._validate_native_toaster_oven_contract()

    def _reset_internal(self) -> None:
        self._tm03_observer_armed = False
        super()._reset_internal()
        self.toaster_oven.close_door(env=self)
        self.toaster_oven.set_time(env=self, val=0.0)
        self.sim.forward()
        self._validate_native_toaster_oven_contract()
        initial = self._physical_sample()
        self._tm03_observer.reset(initial)
        self._tm03_observer_armed = not bool(
            self._tm03_observer.invalid_environment_labels
        )

    def update_state(self) -> None:
        # Bypass the stove/drawer base hook so only the full TM03 observer is authoritative.
        Kitchen.update_state(self)
        self._sync_bound_burner_visual()
        if self._tm03_observer_armed:
            self._update_tm03_observer()

    def reward(self, action=None):
        del action
        if self._tm03_observer_armed:
            self._update_tm03_observer()
        return 1.0 if self._check_success() else 0.0

    @property
    def semantic_state(self):
        return self._tm03_observer.state

    @property
    def physical_observer(self) -> TM03PhysicalObserver:
        return self._tm03_observer

    def toaster_oven_time_joint_name(self) -> str:
        return f"{self.toaster_oven.naming_prefix}{self.toaster_oven_time_joint_suffix}"

    def toaster_oven_time_geom_name(self) -> str:
        return (
            f"{self.toaster_oven.naming_prefix}"
            f"{self.toaster_oven_time_collision_geom_suffix}"
        )

    def toaster_oven_time_position(self) -> np.ndarray:
        geom_id = int(
            self.sim.model.geom_name2id(self.toaster_oven_time_geom_name())
        )
        return np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float).copy()

    def toaster_oven_time_qpos(self) -> float:
        joint_id = int(
            self.sim.model.joint_name2id(self.toaster_oven_time_joint_name())
        )
        address = int(self.sim.model.jnt_qposadr[joint_id])
        return float(self.sim.data.qpos[address])

    def toaster_oven_time_qvel(self) -> float:
        joint_id = int(
            self.sim.model.joint_name2id(self.toaster_oven_time_joint_name())
        )
        address = int(self.sim.model.jnt_dofadr[joint_id])
        return float(self.sim.data.qvel[address])

    def toaster_oven_on(self) -> bool:
        return bool(
            self.toaster_oven_time_qpos()
            >= self._tm03_observer.config.toaster_oven_on_min_rad
        )

    def toaster_oven_door_closed(self) -> bool:
        joint_name = f"{self.toaster_oven.naming_prefix}door_joint"
        joint_id = int(self.sim.model.joint_name2id(joint_name))
        address = int(self.sim.model.jnt_qposadr[joint_id])
        return bool(float(self.sim.data.qpos[address]) <= 0.005)

    def toaster_oven_food_inside(self) -> bool:
        try:
            return bool(
                OU.obj_inside_of(self, self.toaster_oven_food_name, self.toaster_oven)
            )
        except Exception:
            return False

    def _toaster_oven_time_contact(self) -> tuple[bool, bool]:
        geom = self.toaster_oven_time_geom_name()
        gripper_contact = False
        any_robot_contact = False
        try:
            gripper_contact = bool(
                self.check_contact(self.robots[0].gripper["right"], [geom])
            )
        except Exception:
            pass
        try:
            any_robot_contact = bool(
                self.check_contact(self.robots[0].robot_model, [geom])
            )
        except Exception:
            pass
        return gripper_contact, any_robot_contact or gripper_contact

    def _toaster_oven_time_eef_distance(self) -> float:
        try:
            eef = np.asarray(
                self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]],
                dtype=float,
            )
            return float(np.linalg.norm(eef - self.toaster_oven_time_position()))
        except Exception:
            return float("inf")

    def _physical_sample(self) -> PhysicalSample:
        base = super()._physical_sample()
        time_gripper, time_any = self._toaster_oven_time_contact()
        return PhysicalSample(
            step_id=base.step_id,
            knob_qpos_rad=base.knob_qpos_rad,
            knob_qvel_rad_s=base.knob_qvel_rad_s,
            knob_gripper_contact=base.knob_gripper_contact,
            knob_any_robot_contact=base.knob_any_robot_contact,
            target_burner_on=base.target_burner_on,
            other_burners_on=base.other_burners_on,
            drawer_open_fraction=dict(base.drawer_open_fraction),
            completion_button_press_depth_m=dict(
                base.completion_button_press_depth_m
            ),
            pot_stable_on_burner=base.pot_stable_on_burner,
            toaster_oven_on=self.toaster_oven_on(),
            toaster_oven_door_closed=self.toaster_oven_door_closed(),
            toaster_oven_food_inside=self.toaster_oven_food_inside(),
            toaster_oven_time_qpos_rad=self.toaster_oven_time_qpos(),
            toaster_oven_time_qvel_rad_s=self.toaster_oven_time_qvel(),
            toaster_oven_time_gripper_contact=time_gripper,
            toaster_oven_time_any_robot_contact=time_any,
            toaster_oven_time_eef_distance_m=self._toaster_oven_time_eef_distance(),
            knob_actuator_commanded=base.knob_actuator_commanded,
        )

    def _update_tm03_observer(self) -> None:
        self._tm03_observer.observe(self._physical_sample())

    def physical_state(self) -> PhysicalState:
        sample = self._physical_sample()
        return PhysicalState(
            knob_qpos_rad=sample.knob_qpos_rad,
            target_burner_on=sample.target_burner_on,
            all_other_burners_off=not sample.other_burners_on,
            drawer_open_fraction=dict(sample.drawer_open_fraction),
            pot_stable_on_burner=sample.pot_stable_on_burner,
            toaster_oven_on=sample.toaster_oven_on,
            toaster_oven_door_closed=sample.toaster_oven_door_closed,
            toaster_oven_food_inside=sample.toaster_oven_food_inside,
        )

    def _check_success(self) -> bool:
        if self._tm03_observer.invalid_environment_labels:
            return False
        return evaluate_success(
            self.semantic_state,
            self.physical_state(),
            off_commit_max_rad=self._tm03_observer.config.off_commit_max_rad,
            drawer_closed_threshold=self._tm03_observer.config.drawer_closed_threshold,
        ).success

    def task_diagnostics(self) -> dict[str, Any]:
        result = diagnostics(self.semantic_state)
        result.update(
            {
                "target_knob_qpos_rad": self.native_knob_qpos(),
                "target_burner_on": self.bound_burner_on(),
                "toaster_oven_on": self.toaster_oven_on(),
                "toaster_oven_door_closed": self.toaster_oven_door_closed(),
                "toaster_oven_food_inside": self.toaster_oven_food_inside(),
                "toaster_oven_time_qpos_rad": self.toaster_oven_time_qpos(),
                "toaster_oven_time_qvel_rad_s": self.toaster_oven_time_qvel(),
                "toaster_oven_time_knob_position": (
                    self.toaster_oven_time_position().tolist()
                ),
                "invalid_environment_labels": list(
                    self._tm03_observer.invalid_environment_labels
                ),
                "success": self._check_success(),
            }
        )
        return result

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.tm03.prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        metadata.setdefault("refs", {}).update(
            {
                "axis": AXIS,
                "toaster_oven_fixture": self.toaster_oven.name,
                "toaster_oven_asset": Path(self.toaster_oven.file).parent.name,
                "toaster_oven_time_joint": self.toaster_oven_time_joint_name(),
                "toaster_oven_time_geom": self.toaster_oven_time_geom_name(),
                "toaster_oven_food": self.toaster_oven_food_name,
            }
        )
        return metadata

    def _validate_native_toaster_oven_contract(self) -> None:
        actual_asset = Path(self.toaster_oven.file).parent.name
        if actual_asset != self.expected_toaster_oven_asset:
            raise RuntimeError(
                f"TM03 expected toaster oven {self.expected_toaster_oven_asset}, "
                f"found {actual_asset}"
            )
        joint_id = int(
            self.sim.model.joint_name2id(self.toaster_oven_time_joint_name())
        )
        self.sim.model.geom_name2id(self.toaster_oven_time_geom_name())
        lower, upper = self.sim.model.jnt_range[joint_id]
        if not (
            float(lower) <= self._tm03_observer.config.toaster_oven_off_commit_max_rad
            < self._tm03_observer.config.toaster_oven_on_commit_min_rad
            <= float(upper)
        ):
            raise RuntimeError("TM03 toaster-oven time-knob thresholds exceed joint range")


__all__ = ["TM03BoilWaterInspectDrawersAndRunToasterOven"]
