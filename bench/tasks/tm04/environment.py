"""TM04 layout-15 scene combining water, retrieval, and sugar threads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from robocasa.environments.kitchen.kitchen import FixtureType
from robocasa.utils import object_utils as OU
from robosuite.utils import transform_utils as T

from bench.tasks.tm02.domain.contracts import ObserverConfig as TM02ObserverConfig
from bench.tasks.tm02.environment import (
    BUTTON_NAMES,
    COMPLETION_BUTTON,
    WATER_BUTTON,
    TM02FillPotAndRetrieveCan,
)
from bench.tasks.ts03.environment import (
    SAUCEPAN_INTERIOR_REGION_SUFFIX,
    SAUCEPAN_XML,
    SUGAR_CUBE_SCALE,
    SUGAR_CUBE_XML,
    SUGAR_PLATE_SCALE,
    SUGAR_PLATE_XML,
)

from .config import load_task_config, resolve_observer_config
from .domain.contracts import (
    AXIS,
    CAN_NAME,
    HOT_CHOCOLATE_NAME,
    SUGAR_NAMES,
    SUGAR_PLATE_NAME,
    TARGET_SUGAR_COUNT,
    TOTAL_SUGAR_COUNT,
    WATER_POT_NAME,
    ObserverConfig,
)
from .domain.geometry import point_is_inside_cylindrical_region
from .domain.semantics import diagnostics, evaluate_success
from .physical_events import PhysicalSample, TM04PhysicalObserver


class TM04FillPotRetrieveCanAndAddTwoSugars(TM02FillPotAndRetrieveCan):
    """TM02's physical workspace with a task-owned K=2/N=3 sugar thread."""

    def __init__(
        self,
        *args: Any,
        target_sugar_count: int = TARGET_SUGAR_COUNT,
        total_sugar_count: int = TOTAL_SUGAR_COUNT,
        sugar_names: tuple[str, ...] = SUGAR_NAMES,
        selected_sugar_names: tuple[str, ...] | None = None,
        sugar_plate_name: str = SUGAR_PLATE_NAME,
        hot_chocolate_name: str = HOT_CHOCOLATE_NAME,
        water_pot_name: str = WATER_POT_NAME,
        can_name: str = CAN_NAME,
        sugar_plate_world_xyz: tuple[float, float, float] | None = None,
        hot_chocolate_world_xyz: tuple[float, float, float] | None = None,
        observer_config: ObserverConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if (int(target_sugar_count), int(total_sugar_count)) != (
            TARGET_SUGAR_COUNT,
            TOTAL_SUGAR_COUNT,
        ):
            raise ValueError("TM04 environment is frozen to K=2, N=3")
        names = tuple(str(name) for name in sugar_names)
        if names != SUGAR_NAMES:
            raise ValueError(f"TM04 sugar_names must equal {SUGAR_NAMES}")
        selected = (
            names[:TARGET_SUGAR_COUNT]
            if selected_sugar_names is None
            else tuple(str(name) for name in selected_sugar_names)
        )
        if len(selected) != TARGET_SUGAR_COUNT or len(set(selected)) != TARGET_SUGAR_COUNT:
            raise ValueError("TM04 requires two distinct expert-selected sugars")
        if not set(selected).issubset(names):
            raise ValueError("TM04 selected sugars must be drawn from sugar_names")
        if str(water_pot_name) != WATER_POT_NAME:
            raise ValueError(f"TM04 water pot must retain name {WATER_POT_NAME!r}")
        if str(can_name) != CAN_NAME:
            raise ValueError(f"TM04 can must retain name {CAN_NAME!r}")
        if str(sugar_plate_name) != SUGAR_PLATE_NAME:
            raise ValueError(f"TM04 sugar plate must retain name {SUGAR_PLATE_NAME!r}")
        if str(hot_chocolate_name) != HOT_CHOCOLATE_NAME:
            raise ValueError(
                f"TM04 hot-chocolate saucepan must retain name {HOT_CHOCOLATE_NAME!r}"
            )

        if observer_config is None:
            resolved_observer = resolve_observer_config(load_task_config())
        elif isinstance(observer_config, ObserverConfig):
            resolved_observer = observer_config
        else:
            resolved_observer = ObserverConfig(**dict(observer_config))
        self._tm04_observer = TM04PhysicalObserver(resolved_observer)
        self.target_sugar_count = TARGET_SUGAR_COUNT
        self.total_sugar_count = TOTAL_SUGAR_COUNT
        self.sugar_names = names
        self.selected_sugar_names = selected
        self.sugar_plate_name = str(sugar_plate_name)
        self.hot_chocolate_name = str(hot_chocolate_name)
        self.sugar_plate_world_xyz = _optional_xyz(sugar_plate_world_xyz)
        self.hot_chocolate_world_xyz = _optional_xyz(hot_chocolate_world_xyz)
        self._hot_chocolate_reset_position = np.zeros(3, dtype=float)
        self._hot_chocolate_reset_up = np.asarray([0.0, 0.0, 1.0], dtype=float)

        tm02_observer = TM02ObserverConfig(
            cabinet_open_threshold=resolved_observer.cabinet_open_threshold,
            cabinet_closed_threshold=resolved_observer.cabinet_closed_threshold,
            button_pressed_threshold_m=resolved_observer.button_pressed_threshold_m,
            button_released_threshold_m=resolved_observer.button_released_threshold_m,
            button_release_dwell_steps=resolved_observer.button_release_dwell_steps,
            can_place_xy_tolerance_m=resolved_observer.can_place_xy_tolerance_m,
            pot_water_xy_threshold_m=resolved_observer.pot_water_xy_threshold_m,
            pot_translation_guard_m=resolved_observer.pot_translation_guard_m,
            pot_tilt_guard_deg=resolved_observer.pot_tilt_guard_deg,
            pot_fill_required_steps=resolved_observer.pot_fill_required_steps,
            pot_filled_threshold=resolved_observer.pot_filled_threshold,
        )
        super().__init__(
            *args,
            pot_name=WATER_POT_NAME,
            can_name=CAN_NAME,
            observer_config=tm02_observer,
            **kwargs,
        )

    @property
    def distractor_sugar_names(self) -> tuple[str, ...]:
        selected = set(self.selected_sugar_names)
        return tuple(name for name in self.sugar_names if name not in selected)

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.sugar_counter = self.register_fixture_ref(
            "sugar_counter",
            dict(id=FixtureType.COUNTER, ref=self.stove, full_depth_region=True),
        )

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        configs = list(super()._get_obj_cfgs())
        configs.append(
            dict(
                name=self.sugar_plate_name,
                obj_groups=SUGAR_PLATE_XML,
                object_scale=list(SUGAR_PLATE_SCALE),
                graspable=False,
                placement=dict(
                    fixture=self.sugar_counter,
                    sample_region_kwargs=dict(ref=self.stove, loc="right"),
                    size=(0.36, 0.32),
                    pos=("ref", -0.65),
                ),
            )
        )
        configs.extend(
            dict(
                name=name,
                obj_groups=SUGAR_CUBE_XML,
                object_scale=SUGAR_CUBE_SCALE,
                graspable=True,
                placement=dict(object=self.sugar_plate_name, size=(0.80, 0.72)),
            )
            for name in self.sugar_names
        )
        configs.append(
            dict(
                name=self.hot_chocolate_name,
                obj_groups=SAUCEPAN_XML,
                graspable=False,
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=["front_left"]),
                    size=(0.05, 0.05),
                ),
            )
        )
        return configs

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._set_hot_chocolate_liquid_color()

    def _reset_internal(self) -> None:
        super()._reset_internal()
        if self.sugar_plate_world_xyz is not None:
            self.set_object_pose(self.sugar_plate_name, self.sugar_plate_world_xyz)
        if self.hot_chocolate_world_xyz is not None:
            self.set_object_pose(self.hot_chocolate_name, self.hot_chocolate_world_xyz)
        self._arrange_sugars_on_plate()
        for name in (*self.sugar_names, self.sugar_plate_name, self.hot_chocolate_name):
            self._zero_object_velocity(name)
        self._set_hot_chocolate_liquid_color()
        self.sim.forward()
        self._hot_chocolate_reset_position = self.object_world_position(self.hot_chocolate_name)
        self._hot_chocolate_reset_up = self._hot_chocolate_up_direction()
        initial = self._tm04_physical_sample(
            water_button_press_edge=False,
            completion_button_press_edge=False,
        )
        self._tm04_observer.reset(initial)
        self._observer_armed = not bool(self._tm04_observer.invalid_environment_labels)
        self._last_observed_timestep = int(getattr(self, "timestep", 0))

    def _update_tm02_observer(self) -> None:
        """Override TM02's hook so one TM04 observer owns every event."""

        step_id = int(getattr(self, "timestep", 0))
        if step_id <= self._last_observed_timestep:
            return
        edges = self._button_press_edges()
        if edges[WATER_BUTTON]:
            self.set_verified_faucet_state(
                mode="off" if self.faucet_is_on() else "on",
                temperature="cold",
            )
        self._advance_pot_fill()
        self._tm04_observer.observe(
            self._tm04_physical_sample(
                water_button_press_edge=edges[WATER_BUTTON],
                completion_button_press_edge=edges[COMPLETION_BUTTON],
            )
        )
        self._last_observed_timestep = step_id

    @property
    def semantic_state(self):
        return self._tm04_observer.state

    @property
    def physical_observer(self) -> TM04PhysicalObserver:
        return self._tm04_observer

    def physical_state(self):
        return self._tm04_observer.physical_state()

    def _tm04_physical_sample(
        self,
        *,
        water_button_press_edge: bool,
        completion_button_press_edge: bool,
    ) -> PhysicalSample:
        sugar_states = self.sugars_in_hot_chocolate()
        return PhysicalSample(
            step_id=int(getattr(self, "timestep", 0)),
            faucet_on=self.faucet_is_on(),
            water_button_press_edge=bool(water_button_press_edge),
            water_button_press_depth_m=self.button_press_depth(WATER_BUTTON),
            completion_button_press_edge=bool(completion_button_press_edge),
            completion_button_press_depth_m=self.button_press_depth(COMPLETION_BUTTON),
            pot_in_sink=self.pot_in_sink(),
            pot_under_water=self.pot_under_water(),
            pot_stable=self.pot_stable(),
            pot_fill_progress=self.pot_fill_progress(),
            pot_filled=self.pot_is_filled(),
            cabinet_open_fraction=self.cabinet_open_fraction(),
            can_inside_cabinet=self.can_inside_cabinet(),
            can_in_place_region=self.can_in_place_region(),
            can_released=self.can_released(),
            sugars_in_saucepan=sugar_states,
            sugar_released={name: self.sugar_released(name) for name in self.sugar_names},
            saucepan_stable=self.hot_chocolate_stable(),
        )

    def sugars_in_hot_chocolate(self) -> dict[str, bool]:
        cylinder = self._saucepan_interior_cylinder()
        return {
            name: self._sugar_containment(name, cylinder=cylinder)["inside"]
            for name in self.sugar_names
        }

    def sugar_released(self, sugar_name: str) -> bool:
        try:
            return bool(OU.gripper_obj_far(self, str(sugar_name), th=0.15))
        except Exception:
            return False

    def _saucepan_interior_cylinder(self) -> dict[str, Any]:
        saucepan = self.objects[self.hot_chocolate_name]
        geom_name = f"{saucepan.naming_prefix}{SAUCEPAN_INTERIOR_REGION_SUFFIX}"
        geom_id = int(self.sim.model.geom_name2id(geom_name))
        if geom_id < 0:
            raise RuntimeError(f"TM04 saucepan is missing interior region {geom_name!r}")
        size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float).reshape(-1)
        if size.size < 2 or size[0] <= 0 or size[1] <= 0:
            raise RuntimeError(f"TM04 saucepan interior region {geom_name!r} is invalid")
        return {
            "geom_name": geom_name,
            "center_world": np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float),
            "rotation_world_from_local": np.asarray(self.sim.data.geom_xmat[geom_id], dtype=float).reshape(3, 3),
            "radius_m": float(size[0]),
            "half_height_m": float(size[1]),
        }

    def _sugar_containment(
        self, sugar_name: str, *, cylinder: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        region = self._saucepan_interior_cylinder() if cylinder is None else cylinder
        point = self.object_world_position(str(sugar_name))
        center = np.asarray(region["center_world"], dtype=float)
        rotation = np.asarray(region["rotation_world_from_local"], dtype=float)
        local = rotation.T @ (point - center)
        return {
            "inside": point_is_inside_cylindrical_region(
                point,
                center_world=center,
                rotation_world_from_local=rotation,
                radius_m=float(region["radius_m"]),
                half_height_m=float(region["half_height_m"]),
            ),
            "position_world": point.tolist(),
            "position_local": local.tolist(),
        }

    def hot_chocolate_stable(self) -> bool:
        return bool(self.hot_chocolate_stability()["pass"])

    def hot_chocolate_stability(self) -> dict[str, Any]:
        position = self.object_world_position(self.hot_chocolate_name)
        translation = float(np.linalg.norm(position - self._hot_chocolate_reset_position))
        current_up = self._hot_chocolate_up_direction()
        cosine = float(
            np.clip(np.dot(current_up, self._hot_chocolate_reset_up), -1.0, 1.0)
        )
        tilt_deg = float(np.rad2deg(np.arccos(cosine)))
        config = self._tm04_observer.config
        passed = bool(
            translation <= config.saucepan_translation_guard_m
            and tilt_deg <= config.saucepan_tilt_guard_deg
        )
        return {
            "pass": passed,
            "translation_m": translation,
            "max_translation_m": config.saucepan_translation_guard_m,
            "tilt_deg": tilt_deg,
            "max_tilt_deg": config.saucepan_tilt_guard_deg,
            "current_up_world": current_up.tolist(),
            "reset_up_world": self._hot_chocolate_reset_up.tolist(),
        }

    def _hot_chocolate_up_direction(self) -> np.ndarray:
        body_id = self.obj_body_id[self.hot_chocolate_name]
        rotation = np.asarray(
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        up = rotation[:, 2]
        norm = float(np.linalg.norm(up))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("TM04 saucepan has an invalid world rotation")
        return up / norm

    def saucepan_place_pose(
        self,
        *,
        z_offset: float = 0.10,
        xy_offset: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        pose = self.object_world_position(self.hot_chocolate_name)
        pose[:2] += np.asarray(xy_offset, dtype=float)
        pose[2] += float(z_offset)
        return pose

    def _check_success(self) -> bool:
        if not self._observer_armed:
            return False
        return evaluate_success(
            self.semantic_state,
            self.physical_state(),
            cabinet_closed_threshold=self._tm04_observer.config.cabinet_closed_threshold,
        ).success

    def task_diagnostics(self) -> dict[str, Any]:
        physical = self.physical_state()
        report = evaluate_success(
            self.semantic_state,
            physical,
            cabinet_closed_threshold=self._tm04_observer.config.cabinet_closed_threshold,
        )
        cylinder = self._saucepan_interior_cylinder()
        return {
            "axis": AXIS,
            "success": report.success,
            "success_checks": report.checks(),
            "semantic": diagnostics(self.semantic_state),
            "physical": {
                **physical.__dict__,
                "pot_fill_progress": self.pot_fill_progress(),
                "sugar_world_xyz": {
                    name: self.object_world_position(name).tolist() for name in self.sugar_names
                },
                "hot_chocolate_world_xyz": self.object_world_position(self.hot_chocolate_name).tolist(),
                "saucepan_stability": self.hot_chocolate_stability(),
                "saucepan_stability_observer": (
                    self._tm04_observer.saucepan_stability_diagnostics()
                ),
                "saucepan_region": {
                    "geom_name": cylinder["geom_name"],
                    "center_world": np.asarray(cylinder["center_world"]).tolist(),
                    "radius_m": cylinder["radius_m"],
                    "half_height_m": cylinder["half_height_m"],
                },
            },
            "invalid_environment_labels": list(self._tm04_observer.invalid_environment_labels),
        }

    def get_ep_meta(self) -> dict[str, Any]:
        from .prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        metadata["task_id"] = "TM04"
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        metadata.setdefault("refs", {}).update(
            {
                "axis": AXIS,
                "target_sugar_count": self.target_sugar_count,
                "total_sugar_count": self.total_sugar_count,
                "sugar_names": list(self.sugar_names),
                "selected_sugar_names": list(self.selected_sugar_names),
                "distractor_sugar_names": list(self.distractor_sugar_names),
                "sugar_plate_name": self.sugar_plate_name,
                "hot_chocolate_name": self.hot_chocolate_name,
            }
        )
        return metadata

    def _arrange_sugars_on_plate(self) -> None:
        plate = self.objects[self.sugar_plate_name]
        plate_qpos = np.asarray(
            self.sim.data.get_joint_qpos(plate.joints[0]), dtype=float
        ).reshape(-1)
        if plate_qpos.size < 7:
            raise RuntimeError("TM04 sugar plate must have a free joint")
        rotation = T.quat2mat(T.convert_quat(plate_qpos[3:7], to="xyzw"))
        offsets = ((-0.065, -0.025), (0.0, 0.045), (0.065, -0.025))
        for name, offset in zip(self.sugar_names, offsets, strict=True):
            obj = self.objects[name]
            qpos = np.asarray(
                self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
            ).reshape(-1)
            local = np.asarray([offset[0], offset[1], 0.035], dtype=float)
            qpos[:3] = plate_qpos[:3] + rotation @ local
            self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sugar_plate_offsets_xy = offsets
        self.sim.forward()

    def _set_hot_chocolate_liquid_color(self) -> None:
        saucepan = getattr(self, "objects", {}).get(self.hot_chocolate_name)
        if saucepan is None:
            return
        names = (
            f"{saucepan.naming_prefix}liquid",
            f"{saucepan.naming_prefix}saucepan_liquid",
        )
        for name in names:
            try:
                geom_id = int(self.sim.model.geom_name2id(name))
                self.sim.model.geom_rgba[geom_id] = [0.58, 0.37, 0.24, 0.92]
                return
            except Exception:
                continue


def _optional_xyz(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError("TM04 optional world position must be a finite xyz")
    return (float(array[0]), float(array[1]), float(array[2]))


__all__ = ["TM04FillPotRetrieveCanAndAddTwoSugars"]
