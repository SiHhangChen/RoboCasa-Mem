"""TS01 RoboCasa scene and authoritative physical observation boundary."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.button_asset import BUTTON_GLOW_RGBA, BUTTON_RGBA, add_button_asset
from bench.tasks.ts01.domain.contracts import (
    AXIS,
    COMPLETION_BUTTON_NAME,
    RECEPTACLE_NAME,
    SUGAR_PLATE_NAME,
)
from bench.tasks.ts01.domain.episode import (
    normalize_target_sugar_count,
    normalize_total_sugar_count,
)
from bench.tasks.ts01.domain.geometry import point_is_inside_cylindrical_region
from bench.tasks.ts01.materials import (
    SAUCEPAN_ASSET_BY_TYPE, SAUCEPAN_MATERIAL_PROFILES,
    SUGAR_PLATE_MATERIAL_PROFILES, SUGAR_CUBE_MATERIAL_PROFILES,
    apply_object_material,
)
from bench.tasks.ts01.domain.semantics import PhysicalState, SuccessReport, evaluate_success
from bench.tasks.ts01.evaluation import TS01EvaluationSnapshot


SUGAR_CUBE_XML = f"{assets_root}/objects/lightwheel/sugar_cube/SugarCube003/model.xml"
SUGAR_PLATE_XML = f"{assets_root}/objects/lightwheel/oven_tray/OvenTray001/model.xml"
SAUCEPAN_XML = f"{assets_root}/objects/lightwheel/saucepan/Saucepan002/model.xml"
SUGAR_CUBE_SCALE = 1.2
SUGAR_PLATE_SCALE = (0.85, 0.85, 0.8)
SUGAR_PLATE_REFERENCE_OFFSET = -1.0
SAUCEPAN_INTERIOR_REGION_SUFFIX = "reg_int"
SAUCEPAN_LIQUID_REGION_SUFFIX = "liquid"
HOT_CHOCOLATE_RGBA = (0.58, 0.37, 0.24, 0.92)
SAUCEPAN_CONTAINMENT_METHOD = "object_center_inside_saucepan_reg_int_cylinder"
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_COMPLETION_BUTTON_POSITION_FILE = (
    PACKAGE_ROOT
    / "resources"
    / "poses"
    / "completion_button_position_layout39_style37_seed0.json"
)
COMPLETION_BUTTON_GEOM_NAME = "bench_ts01_completion_button"
COMPLETION_BUTTON_JOINT_NAME = f"{COMPLETION_BUTTON_GEOM_NAME}_press_joint"
# Fixed-scene fallback: on the island counter, immediately left of the stove.
COMPLETION_BUTTON_DEFAULT_POSITION = (3.30, -3.15, 0.92)


class TS01AddKSugarToHotChocolate(Kitchen):
    """Consume a frozen episode plan and expose final exactly-K semantics."""

    def __init__(
        self,
        *,
        target_sugar_count: int,
        total_sugar_count: int,
        sugar_names: tuple[str, ...],
        target_sugar_names: tuple[str, ...],
        sugar_plate_name: str = SUGAR_PLATE_NAME,
        receptacle_name: str = RECEPTACLE_NAME,
        saucepan_type: str = "Saucepan002",
        saucepan_asset: str | Path | None = None,
        saucepan_material: str = "original",
        sugar_plate_material: str = "original",
        sugar_cube_material: str = "original",
        completion_button_position_file: str | Path | None = None,
        target_threshold: float = 0.16,
        confirmed_sugar_dwell_steps: int = 5,
        minimum_press_displacement_m: float = 0.004,
        **kwargs: Any,
    ) -> None:
        target = normalize_target_sugar_count(target_sugar_count)
        total = normalize_total_sugar_count(total_sugar_count, target_sugar_count=target)
        if target is None or total is None:
            raise ValueError("TS01 environment requires concrete K and N")
        names = tuple(str(name) for name in sugar_names)
        targets = tuple(str(name) for name in target_sugar_names)
        if len(names) != total or len(set(names)) != len(names):
            raise ValueError("TS01 sugar_names must contain exactly N unique names")
        if len(targets) != target or len(set(targets)) != len(targets):
            raise ValueError("TS01 target_sugar_names must contain exactly K unique names")
        if not set(targets).issubset(names):
            raise ValueError("TS01 target_sugar_names must be drawn from sugar_names")
        if confirmed_sugar_dwell_steps < 1:
            raise ValueError("TS01 confirmed-sugar dwell must be positive")
        if minimum_press_displacement_m <= 0.0:
            raise ValueError("TS01 button displacement threshold must be positive")
        self.target_sugar_count = target
        self.total_sugar_count = total
        self.sugar_names = names
        self._target_sugar_names = targets
        self.sugar_plate_name = str(sugar_plate_name)
        self.receptacle_name = str(receptacle_name)
        self.saucepan_type = str(saucepan_type)
        if self.saucepan_type not in SAUCEPAN_ASSET_BY_TYPE:
            raise ValueError(f"unknown TS01 saucepan type: {self.saucepan_type!r}")
        self.saucepan_asset = str(
            saucepan_asset or SAUCEPAN_ASSET_BY_TYPE[self.saucepan_type]
        )
        if not Path(self.saucepan_asset).is_absolute():
            self.saucepan_asset = str(Path(assets_root) / "objects" / self.saucepan_asset)
        expected_asset = str(Path(assets_root) / "objects" / SAUCEPAN_ASSET_BY_TYPE[self.saucepan_type])
        if Path(self.saucepan_asset).resolve() != Path(expected_asset).resolve():
            raise ValueError("TS01 saucepan_asset does not match saucepan_type")
        self.saucepan_material = str(saucepan_material)
        self.sugar_plate_material = str(sugar_plate_material)
        self.sugar_cube_material = str(sugar_cube_material)
        for value, profiles, label in (
            (self.saucepan_material, SAUCEPAN_MATERIAL_PROFILES, "saucepan_material"),
            (self.sugar_plate_material, SUGAR_PLATE_MATERIAL_PROFILES, "sugar_plate_material"),
            (self.sugar_cube_material, SUGAR_CUBE_MATERIAL_PROFILES, "sugar_cube_material"),
        ):
            if value not in profiles:
                raise ValueError(f"unknown TS01 {label}: {value!r}")
        self.completion_button_position_file = (
            None
            if completion_button_position_file is None
            else str(completion_button_position_file)
        )
        self.target_threshold = float(target_threshold)
        self.confirmed_sugar_dwell_steps = int(confirmed_sugar_dwell_steps)
        self.minimum_press_displacement_m = float(minimum_press_displacement_m)
        self._completion_button_press_latched = False
        self._completion_button_peak_displacement_m = 0.0
        self._completion_button_peak_step: int | None = None
        self._completion_button_threshold_active = False
        self._sugar_goal_complete_observed = False
        self._sugar_confirmed = {name: False for name in names}
        self._sugar_dwell = {name: 0 for name in names}
        self._last_confirmed_update_step: int | None = None
        self.confirmed_sugar_count = 0
        self.objects_released = False
        self._ts01_evaluator = None
        self.knob = "front_left"
        super().__init__(**kwargs)

    @property
    def target_sugar_names(self) -> tuple[str, ...]:
        return self._target_sugar_names

    @property
    def distractor_sugar_names(self) -> tuple[str, ...]:
        targets = set(self.target_sugar_names)
        return tuple(name for name in self.sugar_names if name not in targets)

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.stove

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._completion_button_press_latched = False
        self._completion_button_peak_displacement_m = 0.0
        self._completion_button_peak_step = None
        self._completion_button_threshold_active = False
        self._sugar_goal_complete_observed = False
        self._sugar_confirmed = {name: False for name in self.sugar_names}
        self._sugar_dwell = {name: 0 for name in self.sugar_names}
        self._last_confirmed_update_step = None
        self.confirmed_sugar_count = 0
        self.objects_released = False
        self.stove.set_knob_state(env=self, rng=self.rng, knob=self.knob, mode="on")
        self.material_randomization = {
            self.receptacle_name: apply_object_material(
                self, self.receptacle_name, self.saucepan_material, SAUCEPAN_MATERIAL_PROFILES
            ),
            self.sugar_plate_name: apply_object_material(
                self, self.sugar_plate_name, self.sugar_plate_material, SUGAR_PLATE_MATERIAL_PROFILES
            ),
        }
        for name in self.sugar_names:
            self.material_randomization[name] = apply_object_material(
                self, name, self.sugar_cube_material, SUGAR_CUBE_MATERIAL_PROFILES
            )
        self._set_saucepan_liquid_rgba()
        self.original_sugar_poses = {
            name: self._placement_pose(name) for name in self.sugar_names
        }
        self.original_receptacle_pose = self._placement_pose(self.receptacle_name)
        self._update_task_state()

    def edit_model_xml(self, xml_str: str) -> str:
        """Install one task-owned physical completion button left of the stove."""

        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_str
        existing = {
            geom.get("name")
            for geom in worldbody.iter("geom")
            if geom.get("name")
        }
        if COMPLETION_BUTTON_GEOM_NAME not in existing:
            _add_completion_button_asset(
                worldbody,
                geom_name=COMPLETION_BUTTON_GEOM_NAME,
                pos=self._completion_button_position(),
            )
        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        configs: list[dict[str, Any]] = [
            dict(
                name=self.sugar_plate_name,
                obj_groups=SUGAR_PLATE_XML,
                object_scale=list(SUGAR_PLATE_SCALE),
                placement=dict(
                    fixture=self.counter,
                    size=(0.36, 0.32),
                    sample_region_kwargs=dict(ref=self.stove, loc="right"),
                    pos=("ref", SUGAR_PLATE_REFERENCE_OFFSET),
                ),
            )
        ]
        configs.extend(
            dict(
                name=name,
                obj_groups=SUGAR_CUBE_XML,
                object_scale=SUGAR_CUBE_SCALE,
                graspable=True,
                init_robot_here=index == 0,
                placement=dict(object=self.sugar_plate_name, size=(0.8, 0.75)),
            )
            for index, name in enumerate(self.sugar_names)
        )
        configs.append(
            dict(
                name=self.receptacle_name,
                obj_groups=self.saucepan_asset,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self.knob]),
                    size=(0.05, 0.05),
                ),
            )
        )
        return configs

    def update_state(self) -> None:
        super().update_state()
        self._update_task_state()
        self._update_completion_button_state()

    def _update_completion_button_state(self) -> None:
        """Latch a free-policy press only on a post-sugar threshold crossing."""

        displacement = float(self.completion_button_displacement())
        step = int(getattr(self, "timestep", 0))
        if displacement > self._completion_button_peak_displacement_m:
            self._completion_button_peak_displacement_m = displacement
            self._completion_button_peak_step = step

        threshold_active = displacement >= self.minimum_press_displacement_m
        rising_edge = threshold_active and not self._completion_button_threshold_active
        if (
            rising_edge
            and self._sugar_goal_complete_observed
            and not self._completion_button_press_latched
        ):
            self._completion_button_press_latched = True
        self._completion_button_threshold_active = threshold_active
        self._sugar_goal_complete_observed = (
            self.confirmed_sugar_count == self.target_sugar_count
            and self.objects_released
        )

    def completion_button_position(self) -> np.ndarray:
        """Return the live plunger position used by the press controller."""

        try:
            geom_id = int(self.sim.model.geom_name2id(COMPLETION_BUTTON_GEOM_NAME))
            if geom_id >= 0:
                return np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
        except Exception:
            pass
        position = np.asarray(self._completion_button_position(), dtype=float)
        # The configured point is the base origin; the copied WA02 plunger geom
        # is centred 7.5 cm above it.
        return position + np.asarray([0.0, 0.0, 0.075], dtype=float)

    def completion_button_displacement(self) -> float:
        """Measure downward travel of the spring-loaded completion plunger."""

        try:
            qpos = np.asarray(
                self.sim.data.get_joint_qpos(COMPLETION_BUTTON_JOINT_NAME),
                dtype=float,
            ).reshape(-1)
            if qpos.size:
                return max(0.0, -float(qpos[0]))
        except Exception:
            pass
        return 0.0

    def record_verified_completion_button_press(
        self,
        *,
        physical_motion_verified: bool,
    ) -> bool:
        """Latch completion only after the sugar goal and real plunger motion."""

        self._update_task_state()
        if (
            not bool(physical_motion_verified)
            or self.confirmed_sugar_count != self.target_sugar_count
            or not self.objects_released
        ):
            return False
        self._completion_button_press_latched = True
        self._completion_button_threshold_active = True
        self._completion_button_peak_displacement_m = max(
            self._completion_button_peak_displacement_m,
            self.completion_button_displacement(),
        )
        self._completion_button_peak_step = int(getattr(self, "timestep", 0))
        self._update_task_state()
        return True

    def sugar_goal_complete(self) -> bool:
        """Return the pre-button physical goal used as the final-stage gate."""

        self._update_task_state()
        return bool(
            self.confirmed_sugar_count == self.target_sugar_count
            and self.objects_released
        )

    def _set_saucepan_liquid_rgba(self) -> None:
        """Keep hot chocolate independent of saucepan appearance randomization."""

        saucepan = self.objects[self.receptacle_name]
        names = (
            f"{saucepan.naming_prefix}{SAUCEPAN_LIQUID_REGION_SUFFIX}",
            "saucepan_liquid",
        )
        model = self.sim.model
        for name in names:
            try:
                geom_id = int(model.geom_name2id(name))
            except Exception:
                continue
            if geom_id >= 0:
                model.geom_rgba[geom_id] = HOT_CHOCOLATE_RGBA
                return

    def _saucepan_interior_cylinder(self) -> dict[str, Any]:
        """Read the live world-space cylinder encoded by the saucepan asset."""

        saucepan = self.objects[self.receptacle_name]
        geom_name = f"{saucepan.naming_prefix}{SAUCEPAN_INTERIOR_REGION_SUFFIX}"
        try:
            geom_id = int(self.sim.model.geom_name2id(geom_name))
        except Exception as exc:
            raise RuntimeError(
                f"TS01 saucepan is missing required interior region {geom_name!r}"
            ) from exc
        if geom_id < 0:
            raise RuntimeError(
                f"TS01 saucepan is missing required interior region {geom_name!r}"
            )
        size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float).reshape(-1)
        if size.size < 2 or size[0] <= 0.0 or size[1] <= 0.0:
            raise RuntimeError(f"TS01 saucepan interior region {geom_name!r} is invalid")
        return {
            "geom_name": geom_name,
            "center_world": np.asarray(
                self.sim.data.geom_xpos[geom_id], dtype=float
            ).reshape(3),
            "rotation_world_from_local": np.asarray(
                self.sim.data.geom_xmat[geom_id], dtype=float
            ).reshape(3, 3),
            "radius_m": float(size[0]),
            "half_height_m": float(size[1]),
        }

    def _sugar_saucepan_containment(
        self,
        sugar_name: str,
        *,
        cylinder: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Measure one cube against the live saucepan interior without contact."""

        region = self._saucepan_interior_cylinder() if cylinder is None else cylinder
        point_world = np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[sugar_name]], dtype=float
        ).reshape(3)
        center_world = np.asarray(region["center_world"], dtype=float).reshape(3)
        rotation = np.asarray(
            region["rotation_world_from_local"], dtype=float
        ).reshape(3, 3)
        point_local = rotation.T @ (point_world - center_world)
        inside = point_is_inside_cylindrical_region(
            point_world,
            center_world=center_world,
            rotation_world_from_local=rotation,
            radius_m=float(region["radius_m"]),
            half_height_m=float(region["half_height_m"]),
        )
        return {
            "inside": bool(inside),
            "position_world": point_world.astype(float).tolist(),
            "position_local": point_local.astype(float).tolist(),
            "radial_distance_m": float(np.linalg.norm(point_local[:2])),
            "vertical_offset_m": float(point_local[2]),
        }

    def _physical_state(self) -> PhysicalState:
        cylinder = self._saucepan_interior_cylinder()
        sugars = {
            name: bool(
                self._sugar_saucepan_containment(name, cylinder=cylinder)["inside"]
            )
            for name in self.sugar_names
        }
        sugar_released = all(OU.gripper_obj_far(self, name, th=0.15) for name in self.sugar_names)
        receptacle_released = OU.gripper_obj_far(self, self.receptacle_name, th=0.15)
        return PhysicalState(
            sugars_in_saucepan=sugars,
            sugar_objects_released=sugar_released,
            receptacle_released=receptacle_released,
            completion_button_pressed=self._completion_button_press_latched,
        )

    def _update_task_state(self, physical: PhysicalState | None = None) -> None:
        state = self._physical_state() if physical is None else physical
        report = evaluate_success(state, self.target_sugar_count)
        self.sugars_in_saucepan = dict(report.sugars_in_saucepan)
        self.num_sugar_added = report.num_sugar_added
        self.under_added = report.under_added
        self.over_added = report.over_added
        self.next_sugar_step = report.next_sugar_step
        self.objects_released = bool(report.sugars_released)
        step = int(getattr(self, "timestep", 0))
        if self._last_confirmed_update_step != step:
            self._update_confirmed_sugars(state)
            self._last_confirmed_update_step = step

    def _update_confirmed_sugars(self, physical: PhysicalState) -> None:
        """Debounce stable saucepan containment into confirmed sugar count."""

        dwell = int(self.confirmed_sugar_dwell_steps)
        for name in self.sugar_names:
            inside = bool(physical.sugars_in_saucepan.get(str(name), False))
            released = bool(OU.gripper_obj_far(self, name, th=0.15))
            if inside and released:
                count = self._sugar_dwell[name] + 1
                self._sugar_dwell[name] = count
                if count >= dwell:
                    self._sugar_confirmed[name] = True
            elif not inside:
                self._sugar_dwell[name] = 0
                self._sugar_confirmed[name] = False
        self.confirmed_sugar_count = int(sum(self._sugar_confirmed.values()))

    def _check_success(self) -> bool:
        evaluator = self._ts01_evaluator
        if evaluator is not None:
            return evaluator.observe(self.evaluation_snapshot()).success
        self._update_task_state()
        return evaluate_success(self._physical_state(), self.target_sugar_count).success

    def evaluation_snapshot(self) -> TS01EvaluationSnapshot:
        """Expose only physical facts to the task evaluator."""

        physical = self._physical_state()
        self._update_task_state(physical)
        report = evaluate_success(physical, self.target_sugar_count)
        return TS01EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            target_sugar_count=self.target_sugar_count,
            total_sugar_count=self.total_sugar_count,
            confirmed_sugar_count=int(self.confirmed_sugar_count),
            objects_released=bool(self.objects_released),
            completion_button_pressed=bool(self._completion_button_press_latched),
            details=self._snapshot_details(physical, report),
        )

    def _snapshot_details(
        self,
        physical: PhysicalState,
        report: SuccessReport,
    ) -> dict[str, Any]:
        cylinder = self._saucepan_interior_cylinder()
        containment = {
            name: self._sugar_saucepan_containment(name, cylinder=cylinder)
            for name in self.sugar_names
        }
        return {
            "axis": AXIS,
            "target_sugar_count": int(self.target_sugar_count),
            "total_sugar_count": int(self.total_sugar_count),
            "target_sugar_names": list(self.target_sugar_names),
            "distractor_sugar_names": list(self.distractor_sugar_names),
            "sugar_plate_offsets_xy": [
                list(row) for row in getattr(self, "sugar_plate_offsets_xy", ())
            ],
            "sugar_plate_safe_half_extents_xy": list(
                getattr(self, "sugar_plate_safe_half_extents_xy", ())
            ),
            "completion_button": {
                "geom_name": COMPLETION_BUTTON_GEOM_NAME,
                "joint_name": COMPLETION_BUTTON_JOINT_NAME,
                "position_world": self.completion_button_position().astype(float).tolist(),
                "current_displacement_m": self.completion_button_displacement(),
                "peak_displacement_m": self._completion_button_peak_displacement_m,
                "peak_step": self._completion_button_peak_step,
                "minimum_press_displacement_m": self.minimum_press_displacement_m,
                "pressed": bool(report.completion_button_pressed),
            },
            "saucepan_containment": {
                "method": SAUCEPAN_CONTAINMENT_METHOD,
                "region_geom": str(cylinder["geom_name"]),
                "center_world": np.asarray(
                    cylinder["center_world"], dtype=float
                ).tolist(),
                "radius_m": float(cylinder["radius_m"]),
                "half_height_m": float(cylinder["half_height_m"]),
                "sugars": containment,
            },
            **report.as_dict(),
        }

    def sugar_diagnostics(self) -> dict[str, Any]:
        report = evaluate_success(self._physical_state(), self.target_sugar_count)
        cylinder = self._saucepan_interior_cylinder()
        containment = {
            name: self._sugar_saucepan_containment(name, cylinder=cylinder)
            for name in self.sugar_names
        }
        return {
            "axis": AXIS,
            "total_sugar_count": self.total_sugar_count,
            "target_sugar_names": list(self.target_sugar_names),
            "distractor_sugar_names": list(self.distractor_sugar_names),
            "sugar_plate_offsets_xy": [list(row) for row in getattr(self, "sugar_plate_offsets_xy", ())],
            "sugar_plate_safe_half_extents_xy": list(getattr(self, "sugar_plate_safe_half_extents_xy", ())),
            "objects_released": report.sugars_released,
            "confirmed_sugar_count": int(self.confirmed_sugar_count),
            "completion_button": {
                "geom_name": COMPLETION_BUTTON_GEOM_NAME,
                "joint_name": COMPLETION_BUTTON_JOINT_NAME,
                "position_world": self.completion_button_position().astype(float).tolist(),
                "current_displacement_m": self.completion_button_displacement(),
                "peak_displacement_m": self._completion_button_peak_displacement_m,
                "peak_step": self._completion_button_peak_step,
                "minimum_press_displacement_m": self.minimum_press_displacement_m,
                "pressed": bool(report.completion_button_pressed),
            },
            "saucepan_containment": {
                "method": SAUCEPAN_CONTAINMENT_METHOD,
                "region_geom": str(cylinder["geom_name"]),
                "center_world": np.asarray(
                    cylinder["center_world"], dtype=float
                ).tolist(),
                "radius_m": float(cylinder["radius_m"]),
                "half_height_m": float(cylinder["half_height_m"]),
                "sugars": containment,
            },
            **report.as_dict(),
        }

    def task_diagnostics(self) -> dict[str, Any]:
        evaluator = self._ts01_evaluator
        if evaluator is not None:
            return evaluator.observe(self.evaluation_snapshot()).to_diagnostics()
        result = self.sugar_diagnostics()
        if result["success"]:
            attribution = "success"
        elif result["over_added"]:
            attribution = "over_add"
        elif result["under_added"]:
            attribution = "under_add"
        elif not result["objects_released"]:
            attribution = "not_released"
        elif not result["completion_button_pressed"]:
            attribution = "completion_button_not_pressed"
        else:
            attribution = "wrong_receptacle_or_missing_contact"
        result["failure_attribution"] = attribution
        result["success_checks"] = evaluate_success(
            self._physical_state(), self.target_sugar_count
        ).checks()
        return result

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.ts01.prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        k = self.target_sugar_count
        prompt = resolve_prompt_spec(target_sugar_count=k, variant="memory")
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        metadata.setdefault("refs", {}).update(
            {
                "axis": AXIS,
                "target_sugar_count": k,
                "total_sugar_count": self.total_sugar_count,
                "sugar_names": list(self.sugar_names),
                "target_sugar_names": list(self.target_sugar_names),
                "distractor_sugar_names": list(self.distractor_sugar_names),
                "sugar_plate_name": self.sugar_plate_name,
                "receptacle_name": self.receptacle_name,
                "completion_button_name": COMPLETION_BUTTON_NAME,
            }
        )
        return metadata

    def saucepan_place_pose(
        self, *, z_offset: float = 0.10, xy_offset: tuple[float, float] = (0.0, 0.0)
    ) -> np.ndarray:
        pose = np.asarray(self.sim.data.body_xpos[self.obj_body_id[self.receptacle_name]], dtype=float).copy()
        pose[:2] += np.asarray(xy_offset, dtype=float)
        pose[2] += float(z_offset)
        return pose

    def _placement_pose(self, name: str) -> list[float] | None:
        placement = getattr(self, "object_placements", {}).get(name)
        if placement is not None:
            return np.asarray(placement[0], dtype=float).tolist()
        if name in getattr(self, "obj_body_id", {}):
            return np.asarray(self.sim.data.body_xpos[self.obj_body_id[name]], dtype=float).tolist()
        return None

    def _completion_button_position(self) -> tuple[float, float, float]:
        position = COMPLETION_BUTTON_DEFAULT_POSITION
        path = _completion_button_position_path(self.completion_button_position_file)
        if path is None or not path.is_file():
            return position
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("position", payload)
        array = np.asarray(raw, dtype=float).reshape(-1)
        if array.size < 3 or not np.all(np.isfinite(array[:3])):
            raise ValueError(f"invalid TS01 completion button position in {path}")
        return tuple(float(value) for value in array[:3])


def _completion_button_position_path(value: str | None) -> Path | None:
    if value is None:
        return DEFAULT_COMPLETION_BUTTON_POSITION_FILE
    text = str(value).strip()
    if text.lower() in {"", "none", "off", "false"}:
        return None
    return Path(text)


def _add_completion_button_asset(
    parent: ET.Element,
    *,
    geom_name: str,
    pos: tuple[float, float, float],
) -> None:
    """Install the same TM01-style completion button used by TS02."""

    add_button_asset(
        parent,
        geom_name=geom_name,
        pos=pos,
        color_rgba=BUTTON_RGBA,
        glow_rgba=BUTTON_GLOW_RGBA,
        gravity_compensation=True,
    )


__all__ = [
    "COMPLETION_BUTTON_GEOM_NAME",
    "COMPLETION_BUTTON_JOINT_NAME",
    "TS01AddKSugarToHotChocolate",
]
