"""TM01 RoboCasa scene and authoritative per-control-step observer hook."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root

from bench.tasks.families.drawer_inspection.materials import (
    DRAWER_MATERIAL_TEXTURES,
    apply_drawer_material,
)
from bench.tasks.tm01.config import load_task_config, resolve_observer_config
from bench.tasks.tm01.domain.contracts import (
    AXIS,
    BURNER_ID,
    BURNER_SITE,
    CANONICAL_PROMPT,
    COMPLETION_BUTTON_NAMES,
    DRAWER_NAMES,
    KNOB_COLLISION_GEOM,
    KNOB_ID,
    KNOB_JOINT,
    ObserverConfig,
    StageEvent,
)
from bench.tasks.tm01.domain.episode import (
    DEFAULT_COMPLETION_BUTTON_WORLD_XYZS,
    DEFAULT_DRAWER_WORLD_XYZS,
)
from bench.tasks.tm01.domain.semantics import (
    PhysicalState,
    diagnostics,
    evaluate_success,
    failure_attribution,
    record_close_attempt,
    record_completion_button_attempt,
    record_inspection_attempt,
    record_open_attempt,
)
from bench.tasks.tm01.evaluation import TM01EvaluationSnapshot
from bench.tasks.tm01.physical_events import PhysicalSample, TM01PhysicalObserver
from bench.tasks.tm01.single_drawer import SingleTabletopDrawer
from bench.tasks.tm01.button_asset import TM01_BUTTON_MAX_TRAVEL_M, add_tm01_button_asset


DRAWER_FIXTURE_NAMES = {drawer: f"tm01_{drawer}_fixture" for drawer in DRAWER_NAMES}
COMPLETION_BUTTON_GEOM_NAMES = {
    name: f"bench_tm01_{name}" for name in COMPLETION_BUTTON_NAMES
}
COMPLETION_BUTTON_JOINT_NAMES = {
    name: f"{geom_name}_press_joint"
    for name, geom_name in COMPLETION_BUTTON_GEOM_NAMES.items()
}
COMPLETION_BUTTON_MAX_TRAVEL = TM01_BUTTON_MAX_TRAVEL_M
COMPLETION_BUTTON_RGBA = "0.22 0.72 0.32 1"
COMPLETION_BUTTON_GLOW_RGBA = "0.26 0.84 0.38 1"
DIAGNOSTIC_OVERVIEW_CAMERA = "bench_tm01_overview"
DIAGNOSTIC_DRAWER_CAMERA = "bench_tm01_drawer_overview"


class TM01BoilWaterAndInspectTwoDrawers(Kitchen):
    """Two drawers, one native stove knob, and two equivalent completion buttons."""

    # Layout 39 deliberately uses the island stovetop. It is part of RoboCasa's
    # dining-counter exclusion set for unrelated placement tasks, but TM01 owns
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
        drawer_material_by_fixture: Mapping[str, str] | None = None,
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
            raise ValueError(f"TM01 drawer_names must equal {DRAWER_NAMES}")
        xyzs = tuple(tuple(float(value) for value in xyz) for xyz in drawer_world_xyzs)
        if len(xyzs) != 2 or any(len(xyz) != 3 for xyz in xyzs):
            raise ValueError("TM01 requires exactly two drawer xyz values")
        raw_buttons = dict(
            DEFAULT_COMPLETION_BUTTON_WORLD_XYZS
            if completion_button_world_xyzs is None
            else completion_button_world_xyzs
        )
        if tuple(raw_buttons) != COMPLETION_BUTTON_NAMES:
            raise ValueError(
                f"TM01 completion buttons must equal {COMPLETION_BUTTON_NAMES}"
            )
        button_xyzs: dict[str, tuple[float, float, float]] = {}
        for name in COMPLETION_BUTTON_NAMES:
            value = np.asarray(raw_buttons[name], dtype=float).reshape(-1)
            if value.size != 3 or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"TM01 completion button {name!r} requires a finite xyz"
                )
            button_xyzs[name] = tuple(float(item) for item in value)
        if observer_config is None:
            resolved_observer = resolve_observer_config(load_task_config())
        elif isinstance(observer_config, ObserverConfig):
            resolved_observer = observer_config
        else:
            resolved_observer = ObserverConfig(**dict(observer_config))
        self.drawer_names = (names[0], names[1])
        self.fixture_names = self.drawer_names
        self.drawer_world_xyzs = xyzs
        self.drawer_world_yaw_deg = float(drawer_world_yaw_deg)
        finishes = dict(drawer_material_by_fixture or {})
        if set(finishes) - set(names):
            raise ValueError("TM01 drawer materials name unknown drawers")
        self.drawer_material_by_fixture = {
            drawer: str(finishes.get(drawer, "original")) for drawer in names
        }
        if set(self.drawer_material_by_fixture.values()) - set(
            DRAWER_MATERIAL_TEXTURES
        ):
            raise ValueError("TM01 drawer materials contain unknown finishes")
        self.completion_button_world_xyzs = button_xyzs
        self.expected_stove_asset = str(expected_stove_asset)
        if str(knob_id) != KNOB_ID or str(burner_id) != BURNER_ID:
            raise ValueError(
                "TM01 requires the front_right-knob / rear_left-burner binding"
            )
        self.pot_name = str(pot_name)
        self.pot_xml = str(pot_xml)
        self.knob = str(knob_id)
        self.burner = str(burner_id)
        self._tm01_observer = TM01PhysicalObserver(resolved_observer)
        self._tm01_evaluator: Any | None = None
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
        for drawer, material_id in self.drawer_material_by_fixture.items():
            apply_drawer_material(
                root,
                fixture_name=DRAWER_FIXTURE_NAMES[drawer],
                material_id=material_id,
                texture_root=Path(assets_root) / "textures",
            )
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
        """Arm TM01 only after RoboCasa places and settles the pot."""

        self._observer_armed = False
        super()._reset_internal()
        self.sim.forward()
        self._center_pot_on_bound_burner()
        self._sync_bound_burner_visual()
        self._validate_native_stove_contract()
        self._pot_reset_position = self.pot_world_position()
        self._pot_reset_quaternion = self.pot_world_quaternion()
        initial = self._physical_sample()
        self._tm01_observer.reset(initial)
        self._observer_armed = not bool(self._tm01_observer.invalid_environment_labels)

    def update_state(self) -> None:
        super().update_state()
        self._sync_bound_burner_visual()
        if self._observer_armed:
            self._update_tm01_observer()

    def reward(self, action=None):
        del action
        if self._observer_armed:
            self._update_tm01_observer()
        return 1.0 if self._check_success() else 0.0

    @property
    def semantic_state(self):
        return self._tm01_observer.state

    @property
    def physical_observer(self) -> TM01PhysicalObserver:
        return self._tm01_observer

    @property
    def open_threshold(self) -> float:
        return float(self._tm01_observer.config.drawer_open_threshold)

    @property
    def closed_threshold(self) -> float:
        return float(self._tm01_observer.config.drawer_closed_threshold)

    def cabinet_for_drawer(self, drawer: str) -> SingleTabletopDrawer:
        name = str(drawer)
        if name not in self.drawer_names:
            raise KeyError(f"unknown TM01 drawer {drawer!r}")
        return self.storage_cabinets[name]

    def cabinet_for_slot(self, fixture_name: str) -> SingleTabletopDrawer:
        """Expose TM01's task-owned contact primitive fixture boundary."""

        return self.cabinet_for_drawer(fixture_name)

    def drawer_open_fraction(self, drawer: str) -> float:
        return float(self.cabinet_for_drawer(drawer).open_fraction(self))

    def slot_open_fraction(self, fixture_name: str) -> float:
        return self.drawer_open_fraction(fixture_name)

    def set_slot_open_fraction(self, fixture_name: str, fraction: float) -> None:
        self.set_drawer_open_fraction(fixture_name, fraction)

    def all_drawers_closed(self) -> bool:
        return all(
            self.slot_open_fraction(name) <= self.closed_threshold
            for name in self.fixture_names
        )

    def record_open_event(
        self,
        fixture_name: str,
        *,
        drawer_open: bool,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        result = record_open_attempt(
            self.semantic_state,
            fixture_name,
            drawer_open=drawer_open,
            failure_reason=failure_reason,
        ).as_dict()
        result["diagnostics"] = diagnostics(self.semantic_state)
        return result

    def record_inspection_event(
        self,
        fixture_name: str,
        *,
        drawer_open: bool,
    ) -> dict[str, Any]:
        result = record_inspection_attempt(
            self.semantic_state,
            fixture_name,
            drawer_open=drawer_open,
        ).as_dict()
        result["diagnostics"] = diagnostics(self.semantic_state)
        return result

    def record_close_event(
        self,
        fixture_name: str,
        *,
        drawer_closed: bool,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        report = record_close_attempt(
            self.semantic_state,
            fixture_name,
            drawer_closed=drawer_closed,
            failure_reason=failure_reason,
        )
        if report.passed:
            self._tm01_observer.event_steps.setdefault(
                report.event,
                int(getattr(self, "timestep", 0)),
            )
        result = report.as_dict()
        result["diagnostics"] = diagnostics(self.semantic_state)
        return result

    def record_completion_button_event(
        self,
        *,
        button_physically_pressed: bool,
        all_drawers_closed: bool | None = None,
        button_id: str | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        report = record_completion_button_attempt(
            self.semantic_state,
            button_id=(button_id or COMPLETION_BUTTON_NAMES[0]),
            all_drawers_closed=(
                self.all_drawers_closed()
                if all_drawers_closed is None
                else bool(all_drawers_closed)
            ),
            button_physically_pressed=button_physically_pressed,
            failure_reason=failure_reason,
        )
        if report.passed:
            self._tm01_observer.event_steps.setdefault(
                StageEvent.COMPLETION_BUTTON.value,
                int(getattr(self, "timestep", 0)),
            )
        result = report.as_dict()
        result["diagnostics"] = diagnostics(self.semantic_state)
        return result

    def set_drawer_open_fraction(self, drawer: str, fraction: float) -> None:
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError("TM01 drawer fraction must be in [0, 1]")
        fixture = self.cabinet_for_drawer(drawer)
        if value <= self._tm01_observer.config.drawer_closed_threshold:
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
            raise RuntimeError(f"TM01 {self.knob} stove knob joint is missing")
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
            raise RuntimeError(f"TM01 {self.burner} burner site is missing")
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
        config = self._tm01_observer.config
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

    def _update_tm01_observer(self) -> None:
        self._tm01_observer.observe(self._physical_sample())

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
        if self._tm01_observer.invalid_environment_labels:
            return False
        return evaluate_success(
            self.semantic_state,
            self.physical_state(),
            off_commit_max_rad=self._tm01_observer.config.off_commit_max_rad,
            drawer_closed_threshold=(
                self._tm01_observer.config.drawer_closed_threshold
            ),
        ).success

    def evaluation_snapshot(self) -> TM01EvaluationSnapshot:
        """Expose observer history and the complete domain-success verdict."""

        state = self.semantic_state
        invalid_environment_labels = list(
            self._tm01_observer.invalid_environment_labels
        )
        absorbing_failure = bool(
            state.absorbing_failure or invalid_environment_labels
        )
        failure_code = (
            invalid_environment_labels[0]
            if invalid_environment_labels
            else state.failure_reason
        )
        if absorbing_failure and failure_code is None:
            failure_code = "absorbing_failure"
        return TM01EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            burner_on_completed=state.a1_complete,
            one_drawer_inspected=state.b1_complete,
            boiling_thread_completed=state.a2_complete,
            drawer_thread_completed=state.b2_complete,
            completion_button_pressed=state.completion_button_pressed,
            absorbing_failure=absorbing_failure,
            failure_code=failure_code,
            domain_success=self._check_success(),
            event_steps=dict(self._tm01_observer.event_steps),
            details=self.task_diagnostics(),
        )

    def task_diagnostics(self) -> dict[str, Any]:
        state = diagnostics(self.semantic_state)
        physical = self.physical_state()
        success_report = evaluate_success(
            self.semantic_state,
            physical,
            off_commit_max_rad=self._tm01_observer.config.off_commit_max_rad,
            drawer_closed_threshold=self._tm01_observer.config.drawer_closed_threshold,
        )
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
                    self._tm01_observer.invalid_environment_labels
                ),
                "semantic_event_steps": dict(
                    self._tm01_observer.event_steps
                ),
                "success_checks": success_report.checks(),
                "failure_attribution": (
                    self._tm01_observer.invalid_environment_labels[0]
                    if self._tm01_observer.invalid_environment_labels
                    else failure_attribution(
                        self.semantic_state,
                        physical,
                        off_commit_max_rad=(
                            self._tm01_observer.config.off_commit_max_rad
                        ),
                        drawer_closed_threshold=(
                            self._tm01_observer.config.drawer_closed_threshold
                        ),
                    )
                ),
                "success": self._check_success(),
            }
        )
        return state

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.tm01.prompts import resolve_prompt_spec

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
                "TM01 expected stove asset "
                f"{self.expected_stove_asset}, found {actual_asset}"
            )
        joint_name = self.native_knob_joint_name()
        self.sim.model.get_joint_qpos_addr(joint_name)
        self.sim.model.geom_name2id(self.native_knob_geom_name())
        site = self.stove.burner_sites.get(self.burner)
        if site is None or not str(site.get("name", "")).endswith(BURNER_SITE):
            raise RuntimeError(
                f"TM01 Stovetop057 {self.burner} burner site is missing"
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
    add_tm01_button_asset(
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
        raise KeyError(f"unknown TM01 completion button {button_name!r}")
    return name


__all__ = [
    "COMPLETION_BUTTON_GEOM_NAMES",
    "COMPLETION_BUTTON_GLOW_RGBA",
    "COMPLETION_BUTTON_JOINT_NAMES",
    "COMPLETION_BUTTON_MAX_TRAVEL",
    "COMPLETION_BUTTON_RGBA",
    "DIAGNOSTIC_DRAWER_CAMERA",
    "DIAGNOSTIC_OVERVIEW_CAMERA",
    "TM01BoilWaterAndInspectTwoDrawers",
]
