"""TM02 RoboCasa scene on WA02's layout-15 sink workspace."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.models.fixtures.fixture import Fixture
from robocasa.utils import object_utils as OU
from robocasa.utils.object_utils import (
    get_fixture_to_point_rel_offset,
    get_pos_after_rel_offset,
)

from bench.tasks.tm02.config import load_task_config, resolve_observer_config
from bench.tasks.tm02.domain.contracts import (
    AXIS,
    BUTTON_NAMES,
    CABINET_NAME,
    CAN_NAME,
    CANONICAL_PROMPT,
    COMPLETION_BUTTON,
    POT_NAME,
    WATER_BUTTON,
    ObserverConfig,
)
from bench.tasks.tm02.domain.episode import (
    DEFAULT_CABINET_WORLD_XYZ,
    DEFAULT_CABINET_YAW_DEG,
    DEFAULT_CAN_PLACE_REGION_SIZE_XY,
    DEFAULT_CAN_PLACE_REGION_WORLD_XYZ,
    DEFAULT_COMPLETION_BUTTON_WORLD_XYZ,
    DEFAULT_WATER_BUTTON_WORLD_XYZ,
)
from bench.tasks.tm02.domain.fill import PotFillState, advance_pot_fill
from bench.tasks.tm02.domain.semantics import diagnostics, evaluate_success
from bench.tasks.tm02.physical_events import PhysicalSample, TM02PhysicalObserver
from bench.tasks.wa02_button_asset import add_wa02_button_asset
RED_CAN_XML = f"{assets_root}/objects/objaverse/canned_food/canned_food_15/model.xml"
TM02_SLIDE_CABINET_ASSET = (
    Path(__file__).resolve().parent
    / "resources"
    / "assets"
    / "tm02_slide_cabinet.xml"
)
STORAGE_INTERIOR_LOCAL_CENTER = (-0.225, -0.18, 0.0)
STORAGE_INTERIOR_HALF_EXTENTS = (0.18, 0.09, 0.16)
INTERIOR_FLOOR_LOCAL_Z = -0.15
TOP_STAGING_LOCAL_Y = -0.20
TOP_STAGING_LOCAL_Z = 0.22
BUTTON_GEOM_NAMES = {
    WATER_BUTTON: "bench_tm02_water_button",
    COMPLETION_BUTTON: "bench_tm02_completion_button",
}
BUTTON_JOINT_NAMES = {
    name: f"{geom_name}_press_joint"
    for name, geom_name in BUTTON_GEOM_NAMES.items()
}
BUTTON_MAX_TRAVEL_M = 0.012
CAN_PLACE_REGION_GEOM = "bench_tm02_can_place_region"
DIAGNOSTIC_OVERVIEW_CAMERA = "bench_tm02_overview"
DIAGNOSTIC_TOP_CAMERA = "bench_tm02_top"
CABINET_SUPPORT_COUNTER_NAME = "counter_1_left_group_1"
POT_LIQUID_GEOM_NAME = "pot_liquid"
POT_LIQUID_INTERIOR_GEOM_NAME = "pot_reg_int"
POT_LIQUID_LEVEL_RANGE_M = 0.065
POT_LIQUID_RGBA = (0.35, 0.62, 0.95, 0.55)


class TM02SlideStorageCabinet(Fixture):
    """TM02-owned sliding cabinet migrated from the validated prototype."""

    def __init__(self, *, name: str, rng: Any | None = None) -> None:
        super().__init__(
            xml=str(TM02_SLIDE_CABINET_ASSET),
            name=name,
            duplicate_collision_geoms=False,
            joints=None,
            rng=rng,
        )

    def get_state(self) -> dict[str, float]:
        return {}

    def update_state(self, env: Any) -> None:
        del env

    def get_reset_region_names(self) -> tuple[str, ...]:
        return ("interior",)

    def set_world_pose(
        self,
        *,
        world_xyz: tuple[float, float, float],
        yaw: float,
    ) -> None:
        self.set_pos([float(value) for value in world_xyz])
        self.set_euler([0.0, 0.0, float(yaw)])

    @property
    def door_joint_names(self) -> list[str]:
        return [f"{self.name}_slidedoor_joint"]

    @property
    def handle_name(self) -> str:
        return f"{self.name}_slide_door_handle_site"

    def door_open_fraction(self, env: Any) -> float:
        state = self.get_joint_state(env, self.door_joint_names)
        return float(state.get(self.door_joint_names[0], 0.0))

    def set_door_open_fraction(self, env: Any, fraction: float) -> None:
        resolved = float(np.clip(float(fraction), 0.0, 1.0))
        self.set_joint_state(
            env=env,
            min=resolved,
            max=resolved,
            joint_names=self.door_joint_names,
        )

    def interior_floor_pose(self) -> np.ndarray:
        local = np.asarray(
            [
                STORAGE_INTERIOR_LOCAL_CENTER[0],
                STORAGE_INTERIOR_LOCAL_CENTER[1],
                INTERIOR_FLOOR_LOCAL_Z,
            ],
            dtype=float,
        )
        return np.asarray(get_pos_after_rel_offset(self, local), dtype=float)[:3]

    def interior_pose(self) -> np.ndarray:
        pose = self.interior_floor_pose()
        pose[2] += 0.08
        return pose

    def top_staging_pose(self) -> np.ndarray:
        local = np.asarray(
            [0.0, TOP_STAGING_LOCAL_Y, TOP_STAGING_LOCAL_Z], dtype=float
        )
        return np.asarray(get_pos_after_rel_offset(self, local), dtype=float)[:3]

    def contains_point(self, point: Any, *, margin: float = 0.035) -> bool:
        local = np.asarray(
            get_fixture_to_point_rel_offset(self, point), dtype=float
        ).reshape(-1)[:3]
        center = np.asarray(STORAGE_INTERIOR_LOCAL_CENTER, dtype=float)
        half_extents = np.asarray(STORAGE_INTERIOR_HALF_EXTENTS, dtype=float)
        resolved_margin = float(margin)
        return bool(
            np.all(np.isfinite(local))
            and np.isfinite(resolved_margin)
            and resolved_margin >= 0.0
            and np.all(np.abs(local - center) <= half_extents + resolved_margin)
        )


class TM02FillPotAndRetrieveCan(Kitchen):
    """A fixed pot-in-sink scene plus one closed cabinet containing a can."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        *args: Any,
        water_button_world_xyz: tuple[float, float, float] = (
            DEFAULT_WATER_BUTTON_WORLD_XYZ
        ),
        completion_button_world_xyz: tuple[float, float, float] = (
            DEFAULT_COMPLETION_BUTTON_WORLD_XYZ
        ),
        cabinet_world_xyz: tuple[float, float, float] = DEFAULT_CABINET_WORLD_XYZ,
        cabinet_yaw_deg: float = DEFAULT_CABINET_YAW_DEG,
        can_place_region_world_xyz: tuple[float, float, float] = (
            DEFAULT_CAN_PLACE_REGION_WORLD_XYZ
        ),
        can_place_region_size_xy: tuple[float, float] = (
            DEFAULT_CAN_PLACE_REGION_SIZE_XY
        ),
        pot_name: str = POT_NAME,
        pot_xml: str = "objects/lightwheel/saucepan/Saucepan002/model.xml",
        can_name: str = CAN_NAME,
        can_xml: str = RED_CAN_XML,
        can_scale: float = 0.90,
        observer_config: ObserverConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        water_button_xyz = _finite_xyz(
                water_button_world_xyz,
                label="water_button_world_xyz",
            )
        self.completion_button_world_xyz = _finite_xyz(
            completion_button_world_xyz,
            label="completion_button_world_xyz",
        )
        self.button_world_xyzs = {
            WATER_BUTTON: water_button_xyz,
            COMPLETION_BUTTON: self.completion_button_world_xyz,
        }
        self.cabinet_world_xyz = _finite_xyz(
            cabinet_world_xyz,
            label="cabinet_world_xyz",
        )
        self.cabinet_yaw_deg = float(cabinet_yaw_deg)
        self.can_place_region_world_xyz = _finite_xyz(
            can_place_region_world_xyz,
            label="can_place_region_world_xyz",
        )
        size = np.asarray(can_place_region_size_xy, dtype=float).reshape(-1)
        if size.size != 2 or not np.all(np.isfinite(size)) or np.any(size <= 0.0):
            raise ValueError("TM02 can place region requires a positive finite xy size")
        self.can_place_region_size_xy = (float(size[0]), float(size[1]))
        self.pot_name = str(pot_name)
        self.pot_xml = str(pot_xml)
        self.can_name = str(can_name)
        requested_can_path = _asset_path(str(can_xml)).resolve()
        wr01_can_path = Path(RED_CAN_XML).resolve()
        if requested_can_path != wr01_can_path:
            raise ValueError(
                "TM02 can_xml must match the frozen canonical can asset"
            )
        self.can_xml = RED_CAN_XML
        self.can_scale = float(can_scale)
        # WR01 retrieval helpers address the single TM02 cabinet as its lower
        # storage level. Keeping the same public relation avoids duplicating the
        # validated cabinet-front primitive.
        self.level_by_item = {self.can_name: "lower"}
        if self.pot_name != POT_NAME or self.can_name != CAN_NAME:
            raise ValueError(f"TM02 object names must remain {(POT_NAME, CAN_NAME)}")
        if not np.isfinite(self.can_scale) or self.can_scale <= 0.0:
            raise ValueError("TM02 can_scale must be positive and finite")
        if observer_config is None:
            resolved_observer = resolve_observer_config(load_task_config())
        elif isinstance(observer_config, ObserverConfig):
            resolved_observer = observer_config
        else:
            resolved_observer = ObserverConfig(**dict(observer_config))
        self._tm02_observer = TM02PhysicalObserver(resolved_observer)
        self.final_closed_threshold = float(
            resolved_observer.cabinet_closed_threshold
        )
        self.cabinet_contains_margin = 0.025
        self._observer_armed = False
        self._last_observed_timestep = -1
        self._button_armed = {name: True for name in BUTTON_NAMES}
        self._button_release_steps = {name: 0 for name in BUTTON_NAMES}
        self._pot_reset_position = np.zeros(3, dtype=float)
        self._pot_reset_quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._pot_fill_state = PotFillState()
        self._pot_liquid_geom_id: int | None = None
        self._pot_liquid_full_local_pos: np.ndarray | None = None
        self._pot_liquid_original_size: np.ndarray | None = None
        kwargs.setdefault("obj_registries", ("objaverse", "aigen", "lightwheel"))
        super().__init__(*args, **kwargs)

    # Scene construction -------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()
        self.storage_cabinet = TM02SlideStorageCabinet(
            name=CABINET_NAME,
            rng=self.rng,
        )
        self.fixtures[CABINET_NAME] = self.storage_cabinet
        self.fixture_cfgs.append(
            {"name": CABINET_NAME, "model": self.storage_cabinet, "type": "fixture"}
        )
        self.model.merge_objects([self.storage_cabinet])

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink, full_depth_region=True),
        )
        try:
            self.cabinet_counter = self.fixtures[CABINET_SUPPORT_COUNTER_NAME]
        except KeyError as exc:
            raise RuntimeError(
                "TM02 layout 15 is missing its frozen cabinet support counter "
                f"{CABINET_SUPPORT_COUNTER_NAME!r}"
            ) from exc
        self.init_robot_base_ref = self.sink
        self.fixture_refs["tm02_storage_cabinet"] = (
            self.storage_cabinet,
            False,
        )
        self.fixture_refs["tm02_cabinet_counter"] = (
            self.cabinet_counter,
            False,
        )
        self.storage_cabinet.set_world_pose(
            world_xyz=self.cabinet_world_xyz,
            yaw=float(np.deg2rad(self.cabinet_yaw_deg)),
        )

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        pot_path = _asset_path(self.pot_xml)
        can_path = _asset_path(self.can_xml)
        return [
            dict(
                name=self.pot_name,
                obj_groups=str(pot_path),
                graspable=False,
                placement=dict(
                    fixture=self.sink,
                    ensure_object_boundary_in_range=False,
                    size=(0.12, 0.12),
                    pos=(0.0, 0.0),
                    rotation=(np.pi / 2, np.pi / 2),
                ),
            ),
            # The WR01 cabinet is merged after Kitchen builds its placement
            # initializer. Start on a safe counter region, then deterministically
            # move the WR01 red can into that cabinet in _reset_internal.
            dict(
                name=self.can_name,
                obj_groups=str(can_path),
                object_scale=self.can_scale,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="right"),
                    size=(0.14, 0.14),
                    pos=("ref", -0.8),
                ),
            ),
        ]

    def edit_model_xml(self, xml_str: str) -> str:
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
        colors = {
            WATER_BUTTON: "0.24 0.54 0.90 1",
            COMPLETION_BUTTON: "0.88 0.31 0.24 1",
        }
        glow_colors = {
            WATER_BUTTON: "0.32 0.66 1.00 1",
            COMPLETION_BUTTON: "1.00 0.36 0.24 1",
        }
        for name in BUTTON_NAMES:
            if BUTTON_GEOM_NAMES[name] not in existing:
                _add_pressable_button_asset(
                    worldbody,
                    button_name=name,
                    pos=self.button_world_xyzs[name],
                    rgba=colors[name],
                    glow_rgba=glow_colors[name],
                )
        if CAN_PLACE_REGION_GEOM not in existing:
            center = np.asarray(self.can_place_region_world_xyz, dtype=float).copy()
            center[2] += 0.002
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": CAN_PLACE_REGION_GEOM,
                    "type": "box",
                    "pos": " ".join(f"{item:.6f}" for item in center),
                    "size": (
                        f"{self.can_place_region_size_xy[0] * 0.5:.6f} "
                        f"{self.can_place_region_size_xy[1] * 0.5:.6f} 0.002"
                    ),
                    "rgba": "0.96 0.72 0.16 0.72",
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
        if root.find(f".//camera[@name='{DIAGNOSTIC_OVERVIEW_CAMERA}']") is None:
            ET.SubElement(
                worldbody,
                "camera",
                {
                    "name": DIAGNOSTIC_OVERVIEW_CAMERA,
                    "mode": "fixed",
                    "pos": "1.65 -3.75 2.35",
                    "xyaxes": "0 1 0 -0.68 0 0.73",
                    "fovy": "66",
                },
            )
        if root.find(f".//camera[@name='{DIAGNOSTIC_TOP_CAMERA}']") is None:
            ET.SubElement(
                worldbody,
                "camera",
                {
                    "name": DIAGNOSTIC_TOP_CAMERA,
                    "mode": "fixed",
                    "pos": "1.60 -3.15 4.00",
                    "xyaxes": "1 0 0 0 1 0",
                    "fovy": "58",
                },
            )
        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.sink.set_handle_state(env=self, rng=self.rng, mode="off")
        self.sink.set_temp_state(
            env=self,
            rng=self.rng,
            min_temp=0.25,
            max_temp=0.35,
        )
        self.storage_cabinet.set_door_open_fraction(self, 0.0)
        self._pot_fill_state = PotFillState()
        self._set_pot_liquid_progress(0.0)
        self.sim.forward()

    def _reset_internal(self) -> None:
        self._observer_armed = False
        self._last_observed_timestep = -1
        super()._reset_internal()
        self.sink.set_handle_state(env=self, rng=self.rng, mode="off")
        self.sink.set_temp_state(
            env=self,
            rng=self.rng,
            min_temp=0.25,
            max_temp=0.35,
        )
        self.storage_cabinet.set_door_open_fraction(self, 0.0)
        self._move_pot_below_water_site()
        # Match WR01's upright canned-food storage state instead of preserving
        # the temporary counter sampler's arbitrary reset orientation.
        self.set_object_pose(
            self.can_name,
            self.storage_cabinet.interior_pose(),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        self._zero_object_velocity(self.can_name)
        self._zero_object_velocity(self.pot_name)
        self._pot_fill_state = PotFillState()
        self._set_pot_liquid_progress(0.0)
        self._button_armed = {name: True for name in BUTTON_NAMES}
        self._button_release_steps = {name: 0 for name in BUTTON_NAMES}
        self.sim.forward()
        self._pot_reset_position = self.object_world_position(self.pot_name)
        self._pot_reset_quaternion = self.object_world_quaternion(self.pot_name)
        initial = self._physical_sample(water_button_press_edge=False)
        self._tm02_observer.reset(initial)
        self._observer_armed = not bool(
            self._tm02_observer.invalid_environment_labels
        )
        self._last_observed_timestep = int(getattr(self, "timestep", 0))

    # Per-step observation ----------------------------------------------

    def update_state(self) -> None:
        super().update_state()
        if self._observer_armed:
            self._update_tm02_observer()

    def reward(self, action=None):
        del action
        if self._observer_armed:
            self._update_tm02_observer()
        return 1.0 if self._check_success() else 0.0

    def _update_tm02_observer(self) -> None:
        step_id = int(getattr(self, "timestep", 0))
        if step_id <= self._last_observed_timestep:
            return
        edges = self._button_press_edges()
        if edges[WATER_BUTTON]:
            target_mode = "off" if self.faucet_is_on() else "on"
            self.set_verified_faucet_state(
                mode=target_mode,
                temperature="cold",
            )
        self._advance_pot_fill()
        update = self._tm02_observer.observe(
            self._physical_sample(
                water_button_press_edge=edges[WATER_BUTTON],
                completion_button_press_edge=edges[COMPLETION_BUTTON],
            )
        )
        self._last_observed_timestep = step_id

    # WA02-compatible water interface ----------------------------------

    def water_button_position(self, temperature: str = "cold") -> np.ndarray:
        if str(temperature).strip().lower() != "cold":
            raise ValueError("TM02 uses only WA02's blue/cold water button")
        return self.button_position(WATER_BUTTON)

    def water_site_position(self) -> np.ndarray | None:
        try:
            site_name = str(self.sink.water_site.get("name"))
            site_id = int(self.sim.model.site_name2id(site_name))
            return np.asarray(self.sim.data.site_xpos[site_id], dtype=float).copy()
        except Exception:
            return None

    def set_verified_faucet_state(
        self,
        *,
        mode: str,
        temperature: str | None = "cold",
    ) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"on", "off"}:
            raise ValueError("TM02 faucet mode must be on or off")
        if temperature is not None and str(temperature).strip().lower() != "cold":
            raise ValueError("TM02 supports only cold water")
        self.sink.set_temp_state(
            env=self,
            rng=self.rng,
            min_temp=0.25,
            max_temp=0.35,
        )
        self.sink.set_handle_state(env=self, rng=self.rng, mode=normalized)
        try:
            self.sim.data.set_joint_qpos(
                f"{self.sink.naming_prefix}spout_joint",
                0.0,
            )
        except Exception:
            pass
        self.sim.forward()

    def faucet_is_on(self) -> bool:
        state = dict(self.sink.get_handle_state(env=self))
        return bool(state.get("water_on", False))

    def faucet_matches(
        self,
        *,
        mode: str,
        temperature: str | None = None,
    ) -> bool:
        expected_on = str(mode).strip().lower() == "on"
        if self.faucet_is_on() != expected_on:
            return False
        if temperature is None or not expected_on:
            return True
        return str(temperature).strip().lower() == "cold"

    # Physical queries ---------------------------------------------------

    @property
    def semantic_state(self):
        return self._tm02_observer.state

    @property
    def physical_observer(self) -> TM02PhysicalObserver:
        return self._tm02_observer

    def button_position(self, button_name: str) -> np.ndarray:
        name = _button_name(button_name)
        try:
            geom_id = int(self.sim.model.geom_name2id(BUTTON_GEOM_NAMES[name]))
            return np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float).copy()
        except Exception:
            return np.asarray(self.button_world_xyzs[name], dtype=float) + np.asarray(
                [0.0, 0.0, 0.075],
                dtype=float,
            )

    def button_press_depth(self, button_name: str) -> float:
        name = _button_name(button_name)
        try:
            value = -float(
                self.sim.data.get_joint_qpos(BUTTON_JOINT_NAMES[name])
            )
            return min(BUTTON_MAX_TRAVEL_M, max(0.0, value))
        except Exception:
            return 0.0

    def completion_button_position(
        self,
        button_name: str | None = None,
    ) -> np.ndarray:
        if button_name not in (None, COMPLETION_BUTTON):
            raise KeyError(f"unknown TM02 completion button {button_name!r}")
        return self.button_position(COMPLETION_BUTTON)

    def completion_button_press_depth(
        self,
        button_name: str | None = None,
    ) -> float:
        if button_name not in (None, COMPLETION_BUTTON):
            raise KeyError(f"unknown TM02 completion button {button_name!r}")
        return self.button_press_depth(COMPLETION_BUTTON)

    def cabinet_open_fraction(self) -> float:
        return self.storage_cabinet.door_open_fraction(self)

    def set_cabinet_open_fraction(self, fraction: float) -> None:
        self.storage_cabinet.set_door_open_fraction(self, fraction)
        self.sim.forward()

    def cabinet_for_level(self, level: str) -> TM02SlideStorageCabinet:
        """Expose TM02's single cabinet through its lower-level interface."""

        if str(level) != "lower":
            raise KeyError(f"TM02 only provides the WR01 lower cabinet, got {level!r}")
        return self.storage_cabinet

    def door_open_fraction(self, level: str) -> float:
        return self.cabinet_for_level(level).door_open_fraction(self)

    def is_door_open(self, level: str, *, threshold: float = 0.70) -> bool:
        return self.door_open_fraction(level) >= float(threshold)

    def is_door_closed(
        self,
        level: str,
        *,
        threshold: float | None = None,
    ) -> bool:
        resolved = (
            self.final_closed_threshold if threshold is None else float(threshold)
        )
        return self.door_open_fraction(level) <= resolved

    def object_world_position(self, object_name: str) -> np.ndarray:
        return np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[str(object_name)]],
            dtype=float,
        ).copy()

    def object_world_quaternion(self, object_name: str) -> np.ndarray:
        return np.asarray(
            self.sim.data.body_xquat[self.obj_body_id[str(object_name)]],
            dtype=float,
        ).copy()

    def can_inside_cabinet(self) -> bool:
        return self.storage_cabinet.contains_point(
            self.object_world_position(self.can_name),
            margin=self.cabinet_contains_margin,
        )

    def is_item_in_level(self, item_id: str, level: str) -> bool:
        return bool(
            str(item_id) == self.can_name
            and str(level) == "lower"
            and self.can_inside_cabinet()
        )

    def record_retrieval_pick(self, item_id: str) -> bool:
        """Let the physical observer own B1 while satisfying WR01's adapter."""

        return bool(str(item_id) == self.can_name and not self.can_inside_cabinet())

    def can_in_place_region(self) -> bool:
        position = self.object_world_position(self.can_name)
        target = np.asarray(self.can_place_region_world_xyz, dtype=float)
        half = np.asarray(self.can_place_region_size_xy, dtype=float) * 0.5
        tolerance = float(self._tm02_observer.config.can_place_xy_tolerance_m)
        return bool(
            np.all(np.abs(position[:2] - target[:2]) <= half + tolerance)
            and abs(float(position[2] - target[2])) <= 0.18
            and not self.can_inside_cabinet()
        )

    def can_released(self) -> bool:
        try:
            return bool(OU.gripper_obj_far(self, self.can_name))
        except Exception:
            return False

    def pot_in_sink(self) -> bool:
        try:
            return bool(
                OU.obj_inside_of(
                    self,
                    self.pot_name,
                    self.sink,
                    partial_check=False,
                )
            )
        except Exception:
            return False

    def pot_under_water(self) -> bool:
        water = self.water_site_position()
        if water is None:
            return False
        distance = float(
            np.linalg.norm(
                self.object_world_position(self.pot_name)[:2] - water[:2]
            )
        )
        return bool(
            self.pot_in_sink()
            and distance <= self._tm02_observer.config.pot_water_xy_threshold_m
        )

    def pot_stable(self) -> bool:
        position = self.object_world_position(self.pot_name)
        translation = float(np.linalg.norm(position - self._pot_reset_position))
        quaternion = self.object_world_quaternion(self.pot_name)
        dot = float(
            np.clip(
                abs(np.dot(quaternion, self._pot_reset_quaternion)),
                0.0,
                1.0,
            )
        )
        tilt_deg = math.degrees(2.0 * math.acos(dot))
        config = self._tm02_observer.config
        return bool(
            translation <= config.pot_translation_guard_m
            and tilt_deg <= config.pot_tilt_guard_deg
        )

    def pot_fill_progress(self) -> float:
        return float(self._pot_fill_state.progress)

    def pot_is_filled(self) -> bool:
        return bool(self._pot_fill_state.filled)

    def physical_state(self):
        return self._tm02_observer.physical_state()

    def _physical_sample(
        self,
        *,
        water_button_press_edge: bool,
        completion_button_press_edge: bool = False,
    ) -> PhysicalSample:
        return PhysicalSample(
            step_id=int(getattr(self, "timestep", 0)),
            faucet_on=self.faucet_is_on(),
            water_button_press_edge=bool(water_button_press_edge),
            water_button_press_depth_m=self.button_press_depth(WATER_BUTTON),
            completion_button_press_edge=bool(completion_button_press_edge),
            completion_button_press_depth_m=self.button_press_depth(
                COMPLETION_BUTTON
            ),
            cabinet_open_fraction=self.cabinet_open_fraction(),
            can_inside_cabinet=self.can_inside_cabinet(),
            can_in_place_region=self.can_in_place_region(),
            can_released=self.can_released(),
            pot_in_sink=self.pot_in_sink(),
            pot_under_water=self.pot_under_water(),
            pot_stable=self.pot_stable(),
            pot_fill_progress=self.pot_fill_progress(),
            pot_filled=self.pot_is_filled(),
        )

    def _button_press_edges(self) -> dict[str, bool]:
        config = self._tm02_observer.config
        result: dict[str, bool] = {}
        for name in BUTTON_NAMES:
            depth = self.button_press_depth(name)
            edge = bool(
                self._button_armed[name]
                and depth >= config.button_pressed_threshold_m
            )
            result[name] = edge
            if edge:
                self._button_armed[name] = False
                self._button_release_steps[name] = 0
            elif not self._button_armed[name]:
                if depth <= config.button_released_threshold_m:
                    self._button_release_steps[name] += 1
                    if (
                        self._button_release_steps[name]
                        >= config.button_release_dwell_steps
                    ):
                        self._button_armed[name] = True
                        self._button_release_steps[name] = 0
                else:
                    self._button_release_steps[name] = 0
        return result

    # Reset helpers ------------------------------------------------------

    def _move_pot_below_water_site(self) -> None:
        water = self.water_site_position()
        if water is None:
            raise RuntimeError("TM02 sink water site is unavailable")
        current = self.object_world_position(self.pot_name)
        target = current.copy()
        target[:2] = water[:2]
        self._set_object_pose(self.pot_name, target)

    def set_object_pose(
        self,
        object_name: str,
        position: Any,
        quat_xyzw: Any | None = None,
    ) -> None:
        """WR01-compatible free-joint pose helper for reset and held transport."""

        obj = self.objects[str(object_name)]
        qpos = np.asarray(
            self.sim.data.get_joint_qpos(obj.joints[0]),
            dtype=float,
        ).reshape(-1)
        target = np.asarray(position, dtype=float).reshape(-1)
        if qpos.size < 7 or target.size < 3 or not np.all(np.isfinite(target[:3])):
            raise ValueError(f"invalid TM02 reset pose for {object_name!r}")
        updated = qpos.copy()
        updated[:3] = target[:3]
        if quat_xyzw is not None:
            quat = np.asarray(quat_xyzw, dtype=float).reshape(-1)
            if quat.size != 4 or not np.all(np.isfinite(quat)):
                raise ValueError(f"invalid TM02 reset quaternion for {object_name!r}")
            norm = float(np.linalg.norm(quat))
            if norm <= 1e-9:
                raise ValueError(f"zero TM02 reset quaternion for {object_name!r}")
            quat = quat / norm
            updated[3:7] = quat[[3, 0, 1, 2]]
        self.sim.data.set_joint_qpos(obj.joints[0], updated)
        self.sim.forward()

    def _set_object_pose(self, object_name: str, position: Any) -> None:
        self.set_object_pose(object_name, position)

    def _zero_object_velocity(self, object_name: str) -> None:
        obj = self.objects[str(object_name)]
        try:
            velocity = np.asarray(
                self.sim.data.get_joint_qvel(obj.joints[0]),
                dtype=float,
            ).reshape(-1)
            self.sim.data.set_joint_qvel(obj.joints[0], np.zeros_like(velocity))
        except Exception:
            pass

    def _advance_pot_fill(self) -> None:
        config = self._tm02_observer.config
        state = advance_pot_fill(
            self._pot_fill_state,
            water_reaches_pot=bool(
                self.faucet_is_on()
                and self.pot_under_water()
                and self.pot_stable()
            ),
            required_steps=config.pot_fill_required_steps,
            filled_threshold=config.pot_filled_threshold,
        )
        if state.progress != self._pot_fill_state.progress:
            self._pot_fill_state = state
            self._set_pot_liquid_progress(state.progress)
        else:
            self._pot_fill_state = state

    def _set_pot_liquid_progress(self, progress: float) -> None:
        """Animate the existing saucepan liquid disk from bottom to full."""

        value = float(np.clip(progress, 0.0, 1.0))
        try:
            model = self.sim.model
            if self._pot_liquid_geom_id is None:
                self._pot_liquid_geom_id = int(
                    model.geom_name2id(POT_LIQUID_GEOM_NAME)
                )
            geom_id = int(self._pot_liquid_geom_id)
            if self._pot_liquid_full_local_pos is None:
                self._pot_liquid_full_local_pos = np.asarray(
                    model.geom_pos[geom_id], dtype=float
                ).copy()
            if self._pot_liquid_original_size is None:
                self._pot_liquid_original_size = np.asarray(
                    model.geom_size[geom_id], dtype=float
                ).copy()

            full_pos = np.asarray(self._pot_liquid_full_local_pos, dtype=float)
            local_pos = full_pos.copy()
            local_pos[2] = float(
                full_pos[2] - POT_LIQUID_LEVEL_RANGE_M * (1.0 - value)
            )
            model.geom_pos[geom_id] = local_pos

            liquid_size = np.asarray(self._pot_liquid_original_size, dtype=float).copy()
            try:
                interior_id = int(
                    model.geom_name2id(POT_LIQUID_INTERIOR_GEOM_NAME)
                )
                interior_radius = float(model.geom_size[interior_id][0])
                liquid_size[0] = min(liquid_size[0], 0.94 * interior_radius)
            except Exception:
                pass
            model.geom_size[geom_id] = liquid_size
            alpha = 0.0 if value <= 0.0 else 0.18 + 0.37 * value
            model.geom_rgba[geom_id] = [
                POT_LIQUID_RGBA[0],
                POT_LIQUID_RGBA[1],
                POT_LIQUID_RGBA[2],
                alpha,
            ]
            self.sim.forward()
        except Exception:
            # The fixed Saucepan002 asset supplies this geom. Keep scene reset
            # usable if an early construction phase has not compiled it yet.
            return

    def _pot_liquid_surface_world_z(self) -> float | None:
        try:
            geom_id = int(
                self._pot_liquid_geom_id
                if self._pot_liquid_geom_id is not None
                else self.sim.model.geom_name2id(POT_LIQUID_GEOM_NAME)
            )
            return float(self.sim.data.geom_xpos[geom_id][2])
        except Exception:
            return None

    # Result and metadata ------------------------------------------------

    def _check_success(self) -> bool:
        if not self._observer_armed:
            return False
        return evaluate_success(
            self.semantic_state,
            self.physical_state(),
            cabinet_closed_threshold=(
                self._tm02_observer.config.cabinet_closed_threshold
            ),
        ).success

    def task_diagnostics(self) -> dict[str, Any]:
        semantic = diagnostics(self.semantic_state)
        physical = self.physical_state()
        report = evaluate_success(
            self.semantic_state,
            physical,
            cabinet_closed_threshold=(
                self._tm02_observer.config.cabinet_closed_threshold
            ),
        )
        return {
            "success": report.success,
            "success_checks": report.checks(),
            "semantic": semantic,
            "physical": {
                **physical.__dict__,
                "water_button_press_depth_m": self.button_press_depth(
                    WATER_BUTTON
                ),
                "completion_button_press_depth_m": self.button_press_depth(
                    COMPLETION_BUTTON
                ),
                "pot_world_xyz": self.object_world_position(
                    self.pot_name
                ).tolist(),
                "pot_fill_active_steps": int(self._pot_fill_state.active_steps),
                "pot_fill_required_steps": int(
                    self._tm02_observer.config.pot_fill_required_steps
                ),
                "pot_filled_threshold": float(
                    self._tm02_observer.config.pot_filled_threshold
                ),
                "pot_liquid_surface_world_z": self._pot_liquid_surface_world_z(),
                "can_world_xyz": self.object_world_position(
                    self.can_name
                ).tolist(),
            },
            "invalid_environment_labels": list(
                self._tm02_observer.invalid_environment_labels
            ),
        }

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.tm02.prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        metadata.setdefault("refs", {}).update(
            {
                "axis": AXIS,
                "pot_name": self.pot_name,
                "can_name": self.can_name,
                "cabinet_name": self.storage_cabinet.name,
                "water_button_geom": BUTTON_GEOM_NAMES[WATER_BUTTON],
                "completion_button_geom": BUTTON_GEOM_NAMES[
                    COMPLETION_BUTTON
                ],
                "can_place_region_geom": CAN_PLACE_REGION_GEOM,
                "cabinet_support_fixture": self.cabinet_counter.name,
                "cabinet_asset": str(TM02_SLIDE_CABINET_ASSET),
                "cabinet_implementation": "task_local",
                "can_asset": "task_local_frozen_objaverse_path",
                "can_xml": self.can_xml,
            }
        )
        return metadata


def _asset_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(assets_root) / path


def _finite_xyz(value: Any, *, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"TM02 {label} requires a finite xyz")
    return (float(array[0]), float(array[1]), float(array[2]))


def _button_name(value: str) -> str:
    name = str(value)
    if name not in BUTTON_NAMES:
        raise KeyError(f"unknown TM02 button {value!r}")
    return name


def _add_pressable_button_asset(
    parent: ET.Element,
    *,
    button_name: str,
    pos: tuple[float, float, float],
    rgba: str,
    glow_rgba: str,
) -> None:
    """Install the complete original layered housing around the live plunger."""

    name = _button_name(button_name)
    geom_name = BUTTON_GEOM_NAMES[name]
    add_wa02_button_asset(
        parent,
        geom_name=geom_name,
        pos=pos,
        color_rgba=rgba,
        glow_rgba=glow_rgba,
        joint_name=BUTTON_JOINT_NAMES[name],
        gravity_compensation=True,
    )


__all__ = [
    "BUTTON_GEOM_NAMES",
    "BUTTON_JOINT_NAMES",
    "BUTTON_MAX_TRAVEL_M",
    "CAN_PLACE_REGION_GEOM",
    "CABINET_SUPPORT_COUNTER_NAME",
    "DIAGNOSTIC_OVERVIEW_CAMERA",
    "DIAGNOSTIC_TOP_CAMERA",
    "RED_CAN_XML",
    "TM02FillPotAndRetrieveCan",
    "TM02_SLIDE_CABINET_ASSET",
    "TM02SlideStorageCabinet",
]
