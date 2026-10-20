"""RoboCasa scene and physical-state boundary for WA-05."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU
from bench.tasks.button_asset import add_button_asset

from bench.tasks.wa05.domain.contracts import CUP_IDS, EpisodePlan, SERVICE_SLOTS
from bench.tasks.wa05.domain.geometry import (
    CUP_PLACEMENT_SALTS,
    cup_placement_rng,
    resolve_cup_region_constraints,
    resolve_cup_sampling_regions,
    validate_cup_region_layout,
)
from bench.tasks.wa05.domain.semantics import (
    SemanticState,
    evaluate_success,
    record_button_press,
    record_verified_fill,
    record_verified_staging,
)
from bench.tasks.wa05.evaluation import WA05EvaluationSnapshot
from bench.tasks.wa05.sensor_faucet_asset import (
    install_wa05_twin_sensor_faucets,
    wa05_sensor_faucet_stream_name,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SENSOR_FAUCET_LAYOUT_FILE = (
    PACKAGE_ROOT
    / "resources"
    / "poses"
    / "sensor_faucet_layout_layout15_style34_seed0.json"
)
DEFAULT_ANSWER_BUTTON_POSITIONS_FILE = PACKAGE_ROOT / "resources" / "poses" / "answer_button_positions_layout15_style34_seed0.json"
ANSWER_BUTTON_GEOM_NAMES = {"hot": "bench_wa05_hot_answer_button", "cold": "bench_wa05_cold_answer_button"}
ANSWER_BUTTON_JOINT_NAMES = {temperature: f"{name}_press_joint" for temperature, name in ANSWER_BUTTON_GEOM_NAMES.items()}
IDENTICAL_WATER_CUP_XML = f"{assets_root}/objects/objaverse/cup/cup_2/model.xml"
# Match WA-01's scale for the shared cup asset while keeping this task's
# configuration isolated from other tasks.
WATER_CUP_OBJECT_SCALE = (0.75, 0.75, 0.85)
TEMPERATURES = ("hot", "cold")
DEFAULT_CUP_SAMPLING_REGIONS = {
    "cup_A": {
        # Default fallback mirrors the right-hand WA-02 cup_B region.
        "region_center_xy": (0.8310, -0.4700),
        "region_size_xy": (0.1200, 0.1200),
        "yaw_rad": -0.0103,
    },
}
DEFAULT_CUP_REGION_CONSTRAINTS = {
    "sink_exterior_local_x_max": 0.37005,
    "minimum_sink_clearance": 0.070,
    "minimum_inter_region_gap": 0.070,
    "sampling_inset_xy": (0.003, 0.003),
}
WATER_LIQUID_RGBA = (0.45, 0.65, 0.95, 0.45)


class WA05FillOneCupAndMatchTemperatureButton(Kitchen):
    """Physical WA-05 world; episode choices and workflow live outside this class."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        sensor_faucet_layout_file: str | Path | None = None,
        answer_button_positions_file: str | Path | None = None,
        button_pressed_threshold_m: float = 0.004,
        button_released_threshold_m: float = 0.001,
        button_release_dwell_steps: int = 3,
        service_threshold: float = 0.12,
        cup_upright_threshold_deg: float = 15.0,
        cup_placement_seed: int = 0,
        gripper_force_limit: float = 20.0,
        gripper_pad_friction: tuple[float, float, float] = (2.0, 0.05, 0.0001),
        cup_sampling_regions: Mapping[str, Any] | None = None,
        cup_region_constraints: Mapping[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.sensor_faucet_layout_file = (
            None
            if sensor_faucet_layout_file is None
            else str(sensor_faucet_layout_file)
        )
        self._sensor_faucet_layout = _load_sensor_faucet_layout(
            self.sensor_faucet_layout_file
        )
        self.answer_button_positions_file = None if answer_button_positions_file is None else str(answer_button_positions_file)
        self.button_pressed_threshold_m = float(button_pressed_threshold_m)
        self.button_released_threshold_m = float(button_released_threshold_m)
        self.button_release_dwell_steps = int(button_release_dwell_steps)
        if not 0.0 < self.button_released_threshold_m < self.button_pressed_threshold_m:
            raise ValueError("WA-05 button hysteresis is invalid")
        if self.button_release_dwell_steps <= 0:
            raise ValueError("WA-05 button release dwell must be positive")
        self.service_threshold = float(service_threshold)
        self.cup_upright_threshold_deg = float(cup_upright_threshold_deg)
        # Expert recording uses the original slot constraints. Canonical
        # policy evaluation explicitly relaxes this during initialization.
        self.require_service_region_for_staging = True
        self._cup_placement_seed = int(cup_placement_seed)
        self.gripper_force_limit = float(gripper_force_limit)
        self.gripper_pad_friction = tuple(
            float(value) for value in gripper_pad_friction
        )
        if not np.isfinite(self.gripper_force_limit) or self.gripper_force_limit <= 0.0:
            raise ValueError("WA-05 gripper force limit must be positive and finite")
        if (
            len(self.gripper_pad_friction) != 3
            or not np.all(np.isfinite(self.gripper_pad_friction))
            or any(value <= 0.0 for value in self.gripper_pad_friction)
        ):
            raise ValueError("WA-05 gripper pad friction must contain 3 positive values")
        self._cup_sampling_regions = resolve_cup_sampling_regions(
            DEFAULT_CUP_SAMPLING_REGIONS
            if cup_sampling_regions is None
            else cup_sampling_regions
        )
        self._cup_region_constraints = resolve_cup_region_constraints(
            DEFAULT_CUP_REGION_CONSTRAINTS
            if cup_region_constraints is None
            else cup_region_constraints
        )
        validate_cup_region_layout(
            self._cup_sampling_regions,
            self._cup_region_constraints,
        )
        self._cup_counter_reset_region: dict[str, Any] | None = None
        self._episode_plan: EpisodePlan | None = None
        self._semantic_state = _default_semantic_state()
        self.service_slot_poses: dict[str, list[float]] = {}
        self.service_slot_regions: dict[str, dict[str, list[float]]] = {}
        self.cup_temperatures: dict[str, str | None] = {cup: None for cup in CUP_IDS}
        self.cup_filled: dict[str, bool] = {cup: False for cup in CUP_IDS}
        self.cup_fill_progress: dict[str, float] = {cup: 0.0 for cup in CUP_IDS}
        self._cup_liquid_full_local_pos: dict[str, np.ndarray] = {}
        self._cup_liquid_original_size: dict[str, np.ndarray] = {}
        self._water_stream_original_local_pos: dict[str, np.ndarray] = {}
        self._water_stream_original_half_height: dict[str, float] = {}
        self._active_sensor_faucets: set[str] = set()
        self._sensor_cup_by_temperature: dict[str, str | None] = {
            temperature: None for temperature in TEMPERATURES
        }
        self._sensor_zone_entry_steps: dict[str, int] = {
            temperature: 0 for temperature in TEMPERATURES
        }
        self._sensor_faucet_affinity_by_cup: dict[str, str] = {}
        self._water_stream_intercept_trace: dict[str, Any] = {}
        # 记录每个水温当前已被截断到哪个杯子,避免激活期间重复 forward。
        self._sensor_stream_intercepted_for: dict[str, str | None] = {
            temperature: None for temperature in TEMPERATURES
        }
        self._staging_detected_at: dict[str, np.ndarray | None] = {
            cup: None for cup in CUP_IDS
        }
        self._physical_event_diagnostics: dict[str, Any] = {
            "staging_detected_by_cup": {},
        }
        self._task_event_locked: bool = False
        self._last_task_event_update_step: int | None = None
        # Wa05_water_marker_gate_pending tracks that a physical fill happened
        # while the stream was visually active; used only for the marker gate
        # below and never for the semantic label itself.
        self._water_marker_gate_pending: dict[str, str | None] = {
            cup: None for cup in CUP_IDS
        }
        self._answer_button_armed = {temperature: True for temperature in TEMPERATURES}
        self._answer_button_release_steps = {temperature: 0 for temperature in TEMPERATURES}
        super().__init__(*args, **kwargs)

    # Scene construction -------------------------------------------------

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink, full_depth_region=True),
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.DINING_COUNTER, ref=self.stool, size=(0.75, 0.2)),
        )
        self._cup_counter_reset_region = self._resolve_cup_counter_reset_region()
        self.init_robot_base_ref = self.sink

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        if self._cup_counter_reset_region is None:
            raise RuntimeError("WA-05 cup reset surface was not resolved")
        reset_offset = np.asarray(
            self._cup_counter_reset_region["offset"][:2],
            dtype=float,
        )
        configs: list[dict[str, Any]] = []
        for cup_id in CUP_IDS:
            region = self._cup_sampling_regions[cup_id]
            sampling_region = region.inset(
                self._cup_region_constraints.sampling_inset_xy
            )
            center = np.asarray(region.center_xy, dtype=float)
            offset = center - reset_offset
            configs.append(
                dict(
                    name=cup_id,
                    obj_groups=IDENTICAL_WATER_CUP_XML,
                    object_scale=WATER_CUP_OBJECT_SCALE,
                    graspable=True,
                    init_robot_here=cup_id == "cup_A",
                    reset_region=deepcopy(self._cup_counter_reset_region),
                    placement=dict(
                        fixture=self.counter,
                        # Sample inside an inset region so MuJoCo contact settling
                        # cannot immediately violate the task's legal hard boundary.
                        size=tuple(sampling_region.size_xy),
                        pos=(0.0, 0.0),
                        offset=tuple(float(value) for value in offset),
                        margin=0.0,
                        rotation=float(region.yaw_rad - self.counter.rot),
                        ensure_object_boundary_in_range=True,
                        ensure_valid_placement=True,
                    ),
                )
            )
        return configs

    def configure_canonical_physics(self) -> None:
        """Use WR-03-style physical finger contact for the free cup body.

        This changes only MuJoCo contact parameters.  The cup remains a free
        body and is never teleported or attached to the end effector.
        """

        gripper = self.robots[0].gripper["right"]
        for actuator_name in list(gripper.actuators):
            actuator_id = int(self.sim.model.actuator_name2id(actuator_name))
            self.sim.model.actuator_forcelimited[actuator_id] = 1
            self.sim.model.actuator_forcerange[actuator_id] = [
                -self.gripper_force_limit,
                self.gripper_force_limit,
            ]
        contact_names = set(
            [
                *gripper.important_geoms["left_finger"],
                *gripper.important_geoms["right_finger"],
            ]
        )
        for geom_name in contact_names:
            geom_id = int(self.sim.model.geom_name2id(geom_name))
            self.sim.model.geom_friction[geom_id] = self.gripper_pad_friction
            self.sim.model.geom_condim[geom_id] = 4
            self.sim.model.geom_solref[geom_id] = (0.001, 1.0)
            self.sim.model.geom_solimp[geom_id] = (
                0.999,
                0.999,
                0.001,
                0.5,
                2.0,
            )
        for geom_name in self.objects["cup_A"].contact_geoms or ():
            geom_id = int(self.sim.model.geom_name2id(geom_name))
            self.sim.model.geom_friction[geom_id] = (1.0, 0.005, 0.0001)
            self.sim.model.geom_condim[geom_id] = 4
            self.sim.model.geom_solref[geom_id] = (0.001, 1.0)
            self.sim.model.geom_solimp[geom_id] = (
                0.999,
                0.999,
                0.001,
                0.5,
                2.0,
            )

    def _reset_internal(self) -> None:
        """Install independently seeded cup samples before reset settling."""

        self._resample_initial_cups()
        super()._reset_internal()

    def _resample_initial_cups(self) -> None:
        initializer = getattr(self, "placement_initializer", None)
        if initializer is None:
            return
        children = getattr(initializer, "samplers", {})
        previous_rngs: dict[str, Any] = {}
        try:
            for cup_id in CUP_IDS:
                sampler_name = f"{cup_id}_Sampler"
                if sampler_name not in children:
                    raise RuntimeError(
                        f"WA-05 placement initializer is missing {sampler_name}"
                    )
                sampler = children[sampler_name]
                previous_rngs[sampler_name] = getattr(sampler, "rng", None)
                sampler.rng = cup_placement_rng(self._cup_placement_seed, cup_id)
            placements = initializer.sample(placed_objects=self.fxtr_placements)
        finally:
            for sampler_name, previous_rng in previous_rngs.items():
                children[sampler_name].rng = previous_rng
        missing = set(CUP_IDS) - set(placements)
        if missing:
            raise RuntimeError(
                "WA-05 sampled placement is missing cups: "
                + ", ".join(sorted(missing))
            )
        self.object_placements = placements

    def _resolve_cup_counter_reset_region(self) -> dict[str, Any]:
        """Find the one counter surface containing the declared cup region."""

        matches: list[dict[str, Any]] = []
        for reset_region in self.counter.get_reset_regions(env=self).values():
            offset = np.asarray(reset_region["offset"][:2], dtype=float)
            size = np.asarray(reset_region["size"][:2], dtype=float)
            lower = offset - 0.5 * size
            upper = offset + 0.5 * size
            if all(
                region.x_bounds[0] >= lower[0] - 1e-9
                and region.x_bounds[1] <= upper[0] + 1e-9
                and region.y_bounds[0] >= lower[1] - 1e-9
                and region.y_bounds[1] <= upper[1] + 1e-9
                for region in self._cup_sampling_regions.values()
            ):
                matches.append(deepcopy(reset_region))
        if len(matches) != 1:
            raise RuntimeError(
                "WA-05 expected exactly one counter reset surface for the cup "
                f"region, found {len(matches)}"
            )
        return matches[0]

    @property
    def cup_placement_seed(self) -> int:
        return self._cup_placement_seed

    def cup_sampling_regions(self) -> dict[str, dict[str, Any]]:
        """Return the public region and independent RNG declarations."""

        result: dict[str, dict[str, Any]] = {}
        for cup_id, region in self._cup_sampling_regions.items():
            sampling_region = region.inset(
                self._cup_region_constraints.sampling_inset_xy
            )
            result[cup_id] = {
                "coordinate_frame": "counter_fixture_local",
                "center_xy": list(region.center_xy),
                "size_xy": list(region.size_xy),
                "x_bounds": list(region.x_bounds),
                "y_bounds": list(region.y_bounds),
                "sampling_inset_xy": list(
                    self._cup_region_constraints.sampling_inset_xy
                ),
                "sampling_size_xy": list(sampling_region.size_xy),
                "sampling_x_bounds": list(sampling_region.x_bounds),
                "sampling_y_bounds": list(sampling_region.y_bounds),
                "yaw_rad": float(region.yaw_rad),
                "rng_seed_components": [
                    int(self._cup_placement_seed),
                    int(CUP_PLACEMENT_SALTS[cup_id]),
                ],
            }
        return result

    def edit_model_xml(self, xml_str: str) -> str:
        """Replace the legacy single spout with two touchless faucets."""

        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_str
        layout = self._sensor_faucet_layout
        install_wa05_twin_sensor_faucets(
            worldbody,
            source_water_site_name=str(self.sink.water_site.get("name")),
            local_x_offsets=dict(layout["spout_local_x_offsets"]),
            local_yaw_degrees=dict(layout["spout_local_yaw_degrees"]),
            stream_half_height=float(layout["stream_half_height"]),
        )
        existing = {geom.get("name") for geom in worldbody.iter("geom") if geom.get("name")}
        positions = self._answer_button_positions()
        for temperature, geom_name in ANSWER_BUTTON_GEOM_NAMES.items():
            if geom_name in existing:
                continue
            color = (
                "0.88 0.31 0.24 1"
                if temperature == "hot"
                else "0.24 0.54 0.90 1"
            )
            glow = (
                "1.00 0.38 0.28 1"
                if temperature == "hot"
                else "0.30 0.62 1.00 1"
            )
            add_button_asset(
                worldbody,
                geom_name=geom_name,
                pos=positions[temperature],
                color_rgba=color,
                glow_rgba=glow,
                gravity_compensation=True,
            )
        _hide_legacy_sink_controls(
            worldbody,
            water_site_name=str(self.sink.water_site.get("name")),
            handle_joint_name=f"{self.sink.naming_prefix}handle_joint",
        )
        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.sink.set_handle_state(env=self, rng=self.rng, mode="off")
        self.sink.set_temp_state(env=self, rng=self.rng, min_temp=0.30, max_temp=0.40)
        self.cup_temperatures = {cup: None for cup in CUP_IDS}
        self.cup_filled = {cup: False for cup in CUP_IDS}
        self.cup_fill_progress = {cup: 0.0 for cup in CUP_IDS}
        self._semantic_state = _semantic_for_plan(self._episode_plan)
        self._answer_button_armed = {temperature: True for temperature in TEMPERATURES}
        self._answer_button_release_steps = {temperature: 0 for temperature in TEMPERATURES}
        self._sensor_faucet_affinity_by_cup.clear()
        self.deactivate_sensor_faucets()
        self._reset_physical_event_tracking()
        for cup in CUP_IDS:
            self._set_water_liquid(cup, visible=False)

    # Episode and semantic events ---------------------------------------

    def configure_episode(self, episode: EpisodePlan) -> None:
        """Install a pre-resolved plan without adding labels to observations."""

        self._episode_plan = episode
        self._semantic_state = _semantic_for_plan(episode)
        self._answer_button_armed = {temperature: True for temperature in TEMPERATURES}
        self._answer_button_release_steps = {temperature: 0 for temperature in TEMPERATURES}
        self.cup_temperatures = {cup: None for cup in CUP_IDS}
        self.cup_filled = {cup: False for cup in CUP_IDS}
        self.cup_fill_progress = {cup: 0.0 for cup in CUP_IDS}
        self._sensor_faucet_affinity_by_cup.clear()
        self.deactivate_sensor_faucets()
        self._reset_physical_event_tracking()
        for cup in CUP_IDS:
            self._set_water_liquid(cup, visible=False)

    @property
    def episode_plan(self) -> EpisodePlan:
        if self._episode_plan is None:
            raise RuntimeError("WA-05 episode was not configured after reset")
        return self._episode_plan

    @property
    def semantic_state(self) -> SemanticState:
        return self._semantic_state

    def configure_service_slots(
        self,
        *,
        regions_by_slot: Mapping[str, Mapping[str, Any]],
        sampled_relative_xy_by_slot: Mapping[str, tuple[float, float]],
        surface_z_offset: float,
        service_threshold: float | None = None,
        cup_upright_threshold_deg: float | None = None,
    ) -> None:
        """Resolve fixture-relative neutral service slots into world positions."""

        if set(regions_by_slot) != set(SERVICE_SLOTS):
            raise ValueError("WA-05 requires exactly one service placement region")
        if set(sampled_relative_xy_by_slot) != set(SERVICE_SLOTS):
            raise ValueError("WA-05 requires one sampled target per service region")
        if service_threshold is not None:
            self.service_threshold = float(service_threshold)
        if cup_upright_threshold_deg is not None:
            self.cup_upright_threshold_deg = float(cup_upright_threshold_deg)
        resolved_regions: dict[str, dict[str, list[float]]] = {}
        for slot in SERVICE_SLOTS:
            raw = regions_by_slot[slot]
            center = np.asarray(raw.get("center_xy"), dtype=float).reshape(-1)
            size = np.asarray(raw.get("size_xy"), dtype=float).reshape(-1)
            sampled = np.asarray(
                sampled_relative_xy_by_slot[slot], dtype=float
            ).reshape(-1)
            if (
                center.size != 2
                or size.size != 2
                or sampled.size != 2
                or not np.all(np.isfinite(center))
                or not np.all(np.isfinite(size))
                or not np.all(np.isfinite(sampled))
                or np.any(size <= 0.0)
            ):
                raise ValueError(f"invalid WA-05 service region {slot}")
            lower = center - 0.5 * size
            upper = center + 0.5 * size
            if np.any(sampled < lower) or np.any(sampled > upper):
                raise ValueError(f"sampled WA-05 service target escaped {slot}")
            resolved_regions[slot] = {
                "center_xy": center.astype(float).tolist(),
                "size_xy": size.astype(float).tolist(),
                "x_bounds": [float(lower[0]), float(upper[0])],
                "y_bounds": [float(lower[1]), float(upper[1])],
                "sampled_relative_xy": sampled.astype(float).tolist(),
            }
        self.service_slot_regions = resolved_regions
        self.service_slot_poses = {
            slot: self._counter_surface_pose(
                tuple(resolved_regions[slot]["sampled_relative_xy"]),
                surface_z_offset=surface_z_offset,
            )
            for slot in SERVICE_SLOTS
        }

    def record_verified_fill(self, cup_id: str, temperature: str) -> bool:
        """Commit a fill only after execution code has validated it physically."""

        if str(cup_id) not in CUP_IDS:
            return False
        applied = record_verified_fill(self._semantic_state, temperature)
        if applied:
            self.cup_temperatures[str(cup_id)] = str(temperature)
            self.cup_filled[str(cup_id)] = True
            self.set_cup_fill_progress(str(cup_id), 1.0)
        return bool(
            applied
            or self._semantic_state.filled_temperature == str(temperature)
        )

    def record_verified_staging(
        self, cup_id: str, slot: str, *, currently_ready: bool | None = None
    ) -> bool:
        if str(cup_id) not in CUP_IDS or str(slot) not in SERVICE_SLOTS:
            return False
        if currently_ready is None:
            currently_ready = self.are_cups_physically_staged()
        return record_verified_staging(
            self._semantic_state, currently_ready=bool(currently_ready)
        )

    # Physical queries ---------------------------------------------------

    def update_state(self) -> None:
        """刷新物理状态,并把已验证的物理变化提交为共享任务事件。

        The update runs after every RoboSuite env.step(), so physical evidence
        (fill / staging / answer-button edges) is latched for both the expert
        and the recorded-action / learned-policy evaluation path. Semantic events
        are committed here exactly once per cup; execution handlers may still
        record verified fill/staging during dry-run (deferred) planning without
        duplicating events.
        """

        super().update_state()
        step_id = int(getattr(self, "timestep", 0))
        if self._last_task_event_update_step == step_id:
            return
        self._last_task_event_update_step = step_id
        self._refresh_sensor_faucets()
        self._update_task_events_from_physics()
        # Only interpret a button qpos edge as an intentional answer when the
        # end effector is physically at the button station. Arm motion near
        # the service slot can otherwise create a transient contact edge.
        try:
            eef_pos = np.asarray(
                self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]],
                dtype=float,
            )
            station_pos = np.mean(
                np.stack([self.answer_button_position(temp) for temp in TEMPERATURES]),
                axis=0,
            )
            at_button_station = bool(np.linalg.norm(eef_pos - station_pos) <= 0.12)
        except Exception:
            at_button_station = True
        for temperature in (self._answer_button_press_edges() if at_button_station else ()):
            applied = record_button_press(self._semantic_state, temperature)
            self._physical_event_diagnostics["answer_button"] = {"temperature": temperature, "applied": applied}

    def object_position(self, cup_id: str) -> np.ndarray:
        return np.asarray(self.sim.data.body_xpos[self.obj_body_id[str(cup_id)]], dtype=float)

    def water_site_position(self, temperature: str | None = None) -> np.ndarray | None:
        """Return one sensor-faucet stream center, or their midpoint."""

        try:
            if temperature is None:
                positions = [self.water_site_position(value) for value in TEMPERATURES]
                if any(value is None for value in positions):
                    return None
                return np.mean(np.vstack(positions), axis=0)
            temp = _temperature(temperature)
            site_id = self.sim.model.site_name2id(
                wa05_sensor_faucet_stream_name(temp)
            )
            return np.asarray(self.sim.data.site_xpos[site_id], dtype=float).copy()
        except Exception:
            return None

    def service_slot_position(self, slot: str) -> np.ndarray:
        value = self.service_slot_poses.get(str(slot))
        if value is None:
            raise RuntimeError(f"WA-05 service slot {slot!r} has not been configured")
        return np.asarray(value, dtype=float)

    def answer_button_position(self, temperature: str) -> np.ndarray:
        temp = _temperature(temperature)
        try:
            geom_id = int(self.sim.model.geom_name2id(ANSWER_BUTTON_GEOM_NAMES[temp]))
            return np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float).copy()
        except Exception:
            return np.asarray(self._answer_button_positions()[temp], dtype=float) + np.asarray([0.0, 0.0, 0.075])

    def answer_button_press_depth(self, temperature: str) -> float:
        temp = _temperature(temperature)
        try:
            return max(0.0, min(0.012, -float(self.sim.data.get_joint_qpos(ANSWER_BUTTON_JOINT_NAMES[temp]))))
        except Exception:
            return 0.0

    def service_slot_region(self, slot: str) -> dict[str, list[float]]:
        value = self.service_slot_regions.get(str(slot))
        if value is None:
            raise RuntimeError(f"WA-05 service region {slot!r} is not configured")
        return deepcopy(value)

    def cup_is_upright(self, cup_id: str) -> bool:
        rotation = np.asarray(
            self.sim.data.body_xmat[self.obj_body_id[str(cup_id)]], dtype=float
        ).reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        return bool(tilt <= self.cup_upright_threshold_deg)

    def cup_is_at_service_slot(self, cup_id: str, slot: str) -> bool:
        """Validate the original service-slot placement used by the expert."""

        target = self.service_slot_position(slot)
        distance = float(np.linalg.norm(self.object_position(cup_id)[:2] - target[:2]))
        try:
            contact = bool(OU.check_obj_fixture_contact(self, str(cup_id), self.counter))
        except Exception:
            contact = False
        return bool(
            distance <= self.service_threshold
            and self.cup_is_in_service_region(cup_id, slot)
            and contact
            and self.cup_is_upright(str(cup_id))
        )

    def cup_placement_checks(
        self, cup_id: str, *, release_distance: float = 0.12
    ) -> dict[str, bool]:
        """Require upright, released support anywhere on the task counter."""

        from bench.execution.manipulation.object_interactions import is_object_released

        cup = str(cup_id)
        try:
            counter_contact = bool(OU.check_obj_fixture_contact(self, cup, self.counter))
        except Exception:
            counter_contact = False
        return {
            "counter_contact": counter_contact,
            "cup_upright": self.cup_is_upright(cup),
            "cup_released": bool(is_object_released(self, cup, distance=release_distance)),
        }

    def cup_is_in_service_region(self, cup_id: str, slot: str) -> bool:
        """Return whether the cup center lies inside its service region."""

        try:
            region = self.service_slot_region(slot)
            center_world = np.asarray(
                self.object_position(str(cup_id)), dtype=float
            ).reshape(3)
            local_xy = np.asarray(
                OU.get_fixture_to_point_rel_offset(self.counter, center_world),
                dtype=float,
            )[:2]
            lower = np.asarray(
                [region["x_bounds"][0], region["y_bounds"][0]], dtype=float
            )
            upper = np.asarray(
                [region["x_bounds"][1], region["y_bounds"][1]], dtype=float
            )
            return bool(
                np.all(local_xy >= lower - 1.0e-6)
                and np.all(local_xy <= upper + 1.0e-6)
            )
        except Exception:
            return False

    def are_cups_physically_staged(self) -> bool:
        if self._episode_plan is None or not self.service_slot_poses:
            return False
        return all(
            self._cup_physically_staged(
                cup,
                self._episode_plan.service_slot_by_cup[cup],
            )
            for cup in CUP_IDS
        )

    def faucet_matches(self, *, mode: str, temperature: str | None = None) -> bool:
        """Report the derived touchless state; no handle state is consulted."""

        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"on", "off"}:
            return False
        self._refresh_sensor_faucets()
        if temperature is None:
            active = bool(self._active_sensor_faucets)
        else:
            active = _temperature(temperature) in self._active_sensor_faucets
        return active if normalized_mode == "on" else not active

    def cup_is_under_sensor_faucet(self, cup_id: str, temperature: str) -> bool:
        cup = str(cup_id)
        if cup not in CUP_IDS:
            return False
        temp = _temperature(temperature)
        return bool(
            self._cup_is_in_sensor_zone(cup, temp)
            and temp in self._active_sensor_faucets
        )

    def deactivate_sensor_faucets(self) -> dict[str, Any]:
        """Turn both streams off and restore their full hidden geometry."""

        self._active_sensor_faucets.clear()
        self._sensor_cup_by_temperature = {
            temperature: None for temperature in TEMPERATURES
        }
        self._sensor_zone_entry_steps = {
            temperature: 0 for temperature in TEMPERATURES
        }
        result = self.set_water_stream_intercept(None)
        for temperature in TEMPERATURES:
            self._set_sensor_stream_visibility(temperature, visible=False)
        return result

    def set_sensor_faucet_affinity(
        self, cup_id: str, temperature: str
    ) -> dict[str, Any]:
        """Limit one held cup to its intended faucet until sink exit.

        This prevents a cold-water cup from triggering the neighbouring hot
        sensor while it traverses the two-nozzle workspace. It changes only
        physical sensor eligibility and never writes the remembered label.
        """

        cup = str(cup_id)
        if cup not in CUP_IDS:
            raise ValueError(f"invalid WA-05 cup id: {cup_id!r}")
        temp = _temperature(temperature)
        self._sensor_faucet_affinity_by_cup[cup] = temp
        for other in TEMPERATURES:
            if other == temp:
                continue
            self._sensor_zone_entry_steps[other] = 0
            if self._sensor_cup_by_temperature.get(other) == cup:
                self._active_sensor_faucets.discard(other)
                self._sensor_cup_by_temperature[other] = None
                self._restore_sensor_stream_geometry(other)
                self._set_sensor_stream_visibility(other, visible=False)
        self.sim.forward()
        return {"cup_id": cup, "temperature": temp, "applied": True}

    def clear_sensor_faucet_affinity(self, cup_id: str) -> dict[str, Any]:
        cup = str(cup_id)
        previous = self._sensor_faucet_affinity_by_cup.pop(cup, None)
        return {
            "cup_id": cup,
            "previous_temperature": previous,
            "cleared": previous is not None,
        }

    # Diagnostics and task result --------------------------------------

    def _check_success(self) -> bool:
        self.update_state()
        return evaluate_success(self._semantic_state).success

    def evaluation_snapshot(self) -> WA05EvaluationSnapshot:
        """Expose verified fill, staging, and answer facts to the evaluator."""

        details = self.task_diagnostics()
        state = self._semantic_state
        report = evaluate_success(state)
        plan = self.episode_plan
        return WA05EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            cup_id=plan.cup_id,
            service_slot=plan.service_slot,
            filled_temperature=state.filled_temperature,
            staged_before_answer=state.staged_before_answer,
            pressed_temperature=state.pressed_temperature,
            domain_success=report.success,
            absorbing_failure=state.absorbing_failure,
            failure_code=state.failure_code,
            details=details,
        )

    def task_diagnostics(self) -> dict[str, object]:
        """Return oracle diagnostics for the recorder, not policy observations."""

        self.update_state()
        report = evaluate_success(self._semantic_state)
        slots = {
            slot: np.asarray(position, dtype=float).tolist()
            for slot, position in self.service_slot_poses.items()
        }
        return {
            "success": report.success,
            "success_checks": report.checks(),
            "physical_events": dict(self._physical_event_diagnostics),
            "oracle": {"expert_fill_temperature": None if self._episode_plan is None else self._episode_plan.expert_fill_temperature, "filled_temperature": self._semantic_state.filled_temperature, "pressed_temperature": self._semantic_state.pressed_temperature, "failure_code": self._semantic_state.failure_code},
            "fill": {
                "water_control_mode": "twin_proximity_sensor_faucets",
                "active_sensor_faucets": sorted(self._active_sensor_faucets),
                "sensor_cup_by_temperature": dict(
                    self._sensor_cup_by_temperature
                ),
                "cup_filled": dict(self.cup_filled),
                "cup_fill_progress": dict(self.cup_fill_progress),
                "cup_temperatures": dict(self.cup_temperatures),
                "recorded_temperature": self._semantic_state.filled_temperature,
                "water_stream_intercept": dict(self._water_stream_intercept_trace),
            },
            "service": {
                "slot_positions_world": slots,
                "placement_regions": deepcopy(self.service_slot_regions),
                "placement_rule": (
                    "service_region"
                    if self.require_service_region_for_staging
                    else "upright_released_on_counter"
                ),
                "physically_staged": self.are_cups_physically_staged(),
                "staged_before_answer": self._semantic_state.staged_before_answer,
            },
            "answer_buttons": {temperature: {"position": self.answer_button_position(temperature).tolist(), "press_depth_m": self.answer_button_press_depth(temperature)} for temperature in TEMPERATURES},
        }

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.wa05.prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        return metadata

    # Private helpers ----------------------------------------------------

    def _counter_surface_pose(
        self,
        relative_xy: tuple[float, float],
        *,
        surface_z_offset: float,
    ) -> list[float]:
        try:
            _, _, _, top = self.counter.get_ext_sites(relative=True)
            relative_z = float(np.asarray(top, dtype=float)[2])
        except Exception:
            relative_z = float(np.asarray(self.counter.size, dtype=float)[2]) * 0.5
        world = np.asarray(
            OU.get_pos_after_rel_offset(
                self.counter,
                np.asarray([relative_xy[0], relative_xy[1], relative_z], dtype=float),
            ),
            dtype=float,
        )
        world[2] += float(surface_z_offset)
        return world[:3].astype(float).tolist()

    def _set_water_liquid(self, cup_id: str, *, visible: bool) -> None:
        self.set_cup_fill_progress(str(cup_id), 1.0 if visible else 0.0)

    def set_cup_fill_progress(self, cup_id: str, progress: float) -> dict[str, Any]:
        """Move the cup-owned liquid disk continuously from bottom to full."""

        cup = str(cup_id)
        if cup not in CUP_IDS:
            raise ValueError(f"invalid WA-05 cup id: {cup_id!r}")
        value = float(np.clip(float(progress), 0.0, 1.0))
        self.cup_fill_progress[cup] = value
        try:
            model = self.sim.model
            liquid_id = int(model.geom_name2id(f"{cup}_liquid"))
            if cup not in self._cup_liquid_full_local_pos:
                self._cup_liquid_full_local_pos[cup] = np.asarray(
                    model.geom_pos[liquid_id], dtype=float
                ).copy()
            if cup not in self._cup_liquid_original_size:
                self._cup_liquid_original_size[cup] = np.asarray(
                    model.geom_size[liquid_id], dtype=float
                ).copy()

            full_pos = np.asarray(
                self._cup_liquid_full_local_pos[cup], dtype=float
            )
            liquid_size = np.asarray(
                self._cup_liquid_original_size[cup], dtype=float
            )
            interior_id = int(model.geom_name2id(f"{cup}_reg_int"))
            interior_center_z = float(model.geom_pos[interior_id][2])
            interior_half_height = float(model.geom_size[interior_id][1])
            empty_center_z = (
                interior_center_z - interior_half_height + float(liquid_size[1])
            )
            local_pos = full_pos.copy()
            local_pos[2] = empty_center_z + value * (
                float(full_pos[2]) - empty_center_z
            )
            model.geom_pos[liquid_id] = local_pos
            model.geom_size[liquid_id] = liquid_size
            alpha = 0.0 if value <= 0.0 else 0.12 + 0.33 * value
            model.geom_rgba[liquid_id] = [*WATER_LIQUID_RGBA[:3], alpha]
            self.sim.forward()
            return {
                "applied": True,
                "cup_id": cup,
                "progress": value,
                "liquid_local_z": float(local_pos[2]),
                "empty_local_z": float(empty_center_z),
                "full_local_z": float(full_pos[2]),
            }
        except Exception:
            # During early model construction object liquid sites may not exist yet.
            return {"applied": False, "cup_id": cup, "progress": value}

    def set_water_stream_intercept(
        self,
        cup_id: str | None,
        temperature: str | None = None,
    ) -> dict[str, Any]:
        """Restore both streams or end one active stream at the cup rim."""

        try:
            if cup_id is None:
                streams = {
                    temp: self._restore_sensor_stream_geometry(temp)
                    for temp in TEMPERATURES
                }
                self.sim.forward()
                result = {
                    "restored": True,
                    "intercepted": False,
                    "streams": streams,
                }
                self._water_stream_intercept_trace = result
                return result

            cup = str(cup_id)
            if cup not in CUP_IDS:
                raise ValueError(f"invalid WA-05 cup id: {cup_id!r}")
            if temperature is None:
                candidates = [
                    temp
                    for temp in TEMPERATURES
                    if self._sensor_cup_by_temperature.get(temp) == cup
                ]
                if len(candidates) != 1:
                    raise ValueError("WA-05 stream interception requires one temperature")
                temp = candidates[0]
            else:
                temp = _temperature(temperature)

            model = self.sim.model
            site_id, original_pos, original_half_height = (
                self._sensor_stream_geometry(temp)
            )
            model.site_pos[site_id] = original_pos
            model.site_size[site_id][1] = original_half_height
            self.sim.forward()
            body_id = int(self.obj_body_id[cup])
            body_position = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
            body_rotation = np.asarray(
                self.sim.data.body_xmat[body_id], dtype=float
            ).reshape(3, 3)
            rim_world = body_position + body_rotation @ np.asarray(
                self.objects[cup].top_offset, dtype=float
            )
            site_center_world = np.asarray(
                self.sim.data.site_xpos[site_id], dtype=float
            )
            site_axis_world = np.asarray(
                self.sim.data.site_xmat[site_id], dtype=float
            ).reshape(3, 3)[:, 2]
            axis_z = float(site_axis_world[2])
            if axis_z <= 0.5:
                raise RuntimeError("WA-05 sensor-faucet stream is not vertical")
            stream_top_world_z = float(
                site_center_world[2] + original_half_height * axis_z
            )
            requested_half_height = (
                stream_top_world_z - float(rim_world[2])
            ) / (2.0 * axis_z)
            truncated_half_height = float(
                np.clip(requested_half_height, 0.002, original_half_height)
            )
            truncated_pos = original_pos.copy()
            truncated_pos[2] += original_half_height - truncated_half_height
            model.site_pos[site_id] = truncated_pos
            model.site_size[site_id][1] = truncated_half_height
            model.site_rgba[site_id][3] = 0.52
            self.sim.forward()
            intercepted = bool(
                truncated_half_height < original_half_height - 1.0e-6
            )
            if intercepted:
                self._sensor_stream_intercepted_for[temp] = cup
            result = {
                "restored": False,
                "intercepted": intercepted,
                "control_mode": "proximity_sensor",
                "temperature": temp,
                "cup_id": cup,
                "cup_rim_world_z": float(rim_world[2]),
                "stream_top_world_z": stream_top_world_z,
                "original_half_height": original_half_height,
                "truncated_half_height": truncated_half_height,
                "stream_bottom_world_z": float(
                    stream_top_world_z - 2.0 * truncated_half_height * axis_z
                ),
            }
            self._water_stream_intercept_trace = result
            return result
        except Exception as exc:
            result = {
                "restored": cup_id is None,
                "intercepted": False,
                "cup_id": None if cup_id is None else str(cup_id),
                "temperature": temperature,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._water_stream_intercept_trace = result
            return result

    def water_stream_intercept_callback(
        self,
        cup_id: str,
        temperature: str | None = None,
    ):
        """Drive sensor activation and stream truncation during cup motion."""

        cup = str(cup_id)
        if cup not in CUP_IDS:
            raise ValueError(f"invalid WA-05 cup id: {cup_id!r}")
        temp = (
            _temperature(temperature)
            if temperature is not None
            else str(self._water_stream_intercept_trace.get("temperature") or "")
        )
        temp = _temperature(temp)
        active = False

        def callback(obs: dict[str, Any]) -> dict[str, Any] | None:
            del obs
            nonlocal active
            in_zone, activated = self._advance_sensor_zone(cup, temp)
            if activated:
                self._set_sensor_stream_visibility(temp, visible=True)
                result = self.set_water_stream_intercept(cup, temp)
                active = bool(result.get("intercepted", False))
                return self._get_observations(force_update=True)
            if in_zone:
                # 杯子已进入感应区但 settle 尚未完成:继续累积停留计数,
                # 暂不激活龙头,避免持杯快速掠过另一龙头时误触发。
                return None
            if active or temp in self._active_sensor_faucets:
                self._active_sensor_faucets.discard(temp)
                self._sensor_cup_by_temperature[temp] = None
                self._restore_sensor_stream_geometry(temp)
                self._set_sensor_stream_visibility(temp, visible=False)
                self.sim.forward()
                active = False
                return self._get_observations(force_update=True)
            return None

        return callback

    def _sensor_zone_settle_steps(self) -> int:
        """Steps a cup must stay inside a sensor zone before its faucet turns on."""

        threshold = self._sensor_faucet_layout.get("zone_settle_steps", 0)
        try:
            return max(0, int(threshold))
        except Exception:
            return 0

    def _advance_sensor_zone(self, cup_id: str, temperature: str) -> tuple[bool, bool]:
        """Advance the cup dwell counter inside one sensor zone.

        Returns ``(in_zone, activated)``.  A faucet activates only after the cup
        has stayed inside its zone for ``zone_settle_steps`` consecutive steps,
        so a fast pass through the neighbouring faucet does not turn it on.
        """

        temp = _temperature(temperature)
        allowed = self._sensor_faucet_affinity_by_cup.get(str(cup_id))
        if allowed is not None and allowed != temp:
            self._sensor_zone_entry_steps[temp] = 0
            self._active_sensor_faucets.discard(temp)
            if self._sensor_cup_by_temperature.get(temp) == str(cup_id):
                self._sensor_cup_by_temperature[temp] = None
            self._restore_sensor_stream_geometry(temp)
            self._set_sensor_stream_visibility(temp, visible=False)
            return False, False
        in_zone = self._cup_is_in_sensor_zone(cup_id, temp)
        if not in_zone:
            self._sensor_zone_entry_steps[temp] = 0
            return False, False
        count = int(self._sensor_zone_entry_steps.get(temp, 0)) + 1
        self._sensor_zone_entry_steps[temp] = count
        activated = count >= self._sensor_zone_settle_steps()
        if activated:
            self._active_sensor_faucets.add(temp)
            self._sensor_cup_by_temperature[temp] = str(cup_id)
        return True, activated

    def _refresh_sensor_faucets(self) -> None:
        threshold = self._sensor_zone_settle_steps()
        for temp in TEMPERATURES:
            in_zone_cup = None
            for candidate in CUP_IDS:
                if not self._cup_is_in_sensor_zone(candidate, temp):
                    continue
                allowed = self._sensor_faucet_affinity_by_cup.get(candidate)
                if allowed is not None and allowed != temp:
                    continue
                in_zone_cup = candidate
                break
            was_active = temp in self._active_sensor_faucets
            if in_zone_cup is not None:
                count = int(self._sensor_zone_entry_steps.get(temp, 0)) + 1
                self._sensor_zone_entry_steps[temp] = count
                settled = count >= threshold
                if settled:
                    self._active_sensor_faucets.add(temp)
                    self._sensor_cup_by_temperature[temp] = in_zone_cup
                    self._set_sensor_stream_visibility(temp, visible=True)
                    if self._sensor_stream_intercepted_for.get(temp) != in_zone_cup:
                        self.set_water_stream_intercept(in_zone_cup, temp)
                elif was_active:
                    self._active_sensor_faucets.discard(temp)
                    self._sensor_cup_by_temperature[temp] = None
                    self._restore_sensor_stream_geometry(temp)
                    self._set_sensor_stream_visibility(temp, visible=False)
            else:
                self._sensor_zone_entry_steps[temp] = 0
                self._active_sensor_faucets.discard(temp)
                self._sensor_cup_by_temperature[temp] = None
                if was_active:
                    self._restore_sensor_stream_geometry(temp)
                self._set_sensor_stream_visibility(temp, visible=False)

    def _cup_is_in_sensor_zone(self, cup_id: str, temperature: str) -> bool:
        try:
            cup_position = self.object_position(cup_id)
            stream_position = self.water_site_position(temperature)
            if stream_position is None:
                return False
            layout = self._sensor_faucet_layout
            xy_distance = float(
                np.linalg.norm(cup_position[:2] - stream_position[:2])
            )
            stream_top_z = float(
                stream_position[2] + float(layout["stream_half_height"])
            )
            return bool(
                xy_distance <= float(layout["sensor_xy_radius"])
                and float(cup_position[2]) >= float(layout["sensor_min_cup_z"])
                and float(cup_position[2])
                <= stream_top_z + float(layout["sensor_max_cup_z_above_nozzle"])
            )
        except Exception:
            return False

    # Physical task-event derivation ------------------------------------

    def _cup_physically_staged(self, cup_id: str, slot: str) -> bool:
        """Apply expert slot constraints or the policy evaluation counter rule."""

        cup = str(cup_id)
        if cup not in CUP_IDS or slot not in SERVICE_SLOTS:
            return False
        try:
            if getattr(self, "require_service_region_for_staging", True):
                from bench.execution.manipulation.object_interactions import is_object_released

                return bool(
                    self.cup_is_at_service_slot(cup, slot)
                    and is_object_released(self, cup, distance=0.12)
                )
            return all(self.cup_placement_checks(cup).values())
        except Exception:
            return False

    def _reset_physical_event_tracking(self) -> None:
        """Clear per-episode event latching between resets.

        Called from ``_setup_scene`` and ``configure_episode`` so that replay
        / policy rollouts on the same env object start from a clean history.
        """

        self._staging_detected_at = {cup: None for cup in CUP_IDS}
        self._physical_event_diagnostics = {
            "staging_detected_by_cup": {},
        }
        self._task_event_locked = False
        self._last_task_event_update_step = None
        self._water_marker_gate_pending = {cup: None for cup in CUP_IDS}

    def _update_task_events_from_physics(self) -> None:
        """Lock shared semantic history from physical evidence each env step.

        This is the recorded-action / learned-policy evaluation path that the
        expert workflow already reached through its execution handlers.  It
        never infers hidden labels (temperature / cup identity) from images or
        prompts; it only commits what the physical state provably shows:

        * a cup has water once it is observed under an active sensor faucet;
        * staging requires upright, released counter placement; expert
          recording additionally requires the planned service region;
        * a later active faucet of the other temperature is a mixed fill.
        """

        if self._task_event_locked:
            return
        plan = self._episode_plan
        if plan is None:
            return
        # The sensor association already includes the configured activation
        # debounce. Its first verified water contact commits the semantic fill;
        # the synthetic liquid level continues to animate independently and is
        # not an evaluation threshold.
        for cup in CUP_IDS:
            temperature = self._sensor_cup_by_temperature.get("hot")
            if temperature == cup:
                temperature = "hot"
            else:
                temperature = self._sensor_cup_by_temperature.get("cold")
                temperature = "cold" if temperature == cup else None
            if temperature is None:
                continue
            if not self.cup_is_under_sensor_faucet(cup, temperature):
                continue
            filled_temperature = self._semantic_state.filled_temperature
            if (
                filled_temperature is not None
                and temperature != filled_temperature
            ):
                record_verified_fill(self._semantic_state, temperature)
                self._physical_event_diagnostics["mixed_fill_detected"] = {
                    "cup_id": cup,
                    "first_temperature": filled_temperature,
                    "second_temperature": temperature,
                }
                continue

            fill_steps = max(
                1,
                int(getattr(self, "fill_under_water_steps_required", 40)),
            )
            progress = float(self.cup_fill_progress.get(cup, 0.0))
            progress = min(1.0, progress + 1.0 / fill_steps)
            self.set_cup_fill_progress(cup, progress)
            if filled_temperature is not None:
                continue

            applied = record_verified_fill(self._semantic_state, temperature)
            if applied:
                self.cup_temperatures[cup] = temperature
                self.cup_filled[cup] = True
                # Pure action replay does not execute the expert handler that
                # installs faucet affinity. Once physical dwell has identified
                # the first completed fill, lock this cup to that faucet so its
                # sink-exit path cannot activate the neighbouring sensor and be
                # misclassified as a mixed-temperature refill.
                self._sensor_faucet_affinity_by_cup[cup] = temperature
                for other in TEMPERATURES:
                    if other == temperature:
                        continue
                    self._sensor_zone_entry_steps[other] = 0
                    if self._sensor_cup_by_temperature.get(other) == cup:
                        self._active_sensor_faucets.discard(other)
                        self._sensor_cup_by_temperature[other] = None
                        self._restore_sensor_stream_geometry(other)
                        self._set_sensor_stream_visibility(other, visible=False)
                self._physical_event_diagnostics.setdefault(
                    "fill_detected_by_cup", {}
                )[cup] = temperature
            continue

        # Staging uses the active recording / evaluation placement rule.
        for cup in CUP_IDS:
            if self._semantic_state.staged_before_answer:
                continue
            slot = str(plan.service_slot_by_cup[cup])
            if not self._cup_physically_staged(cup, slot):
                continue
            applied = record_verified_staging(self._semantic_state, currently_ready=True)
            if applied:
                self._staging_detected_at[cup] = np.asarray(
                    self.object_position(cup), dtype=float
                ).copy()
                self._physical_event_diagnostics.setdefault(
                    "staging_detected_by_cup", {}
                )[cup] = slot

        if self._semantic_state.filled_temperature is not None and self._semantic_state.staged_before_answer and self._semantic_state.pressed_temperature is not None:
            self._task_event_locked = True

    def _sensor_stream_geometry(
        self, temperature: str
    ) -> tuple[int, np.ndarray, float]:
        temp = _temperature(temperature)
        model = self.sim.model
        site_id = int(model.site_name2id(wa05_sensor_faucet_stream_name(temp)))
        if temp not in self._water_stream_original_local_pos:
            self._water_stream_original_local_pos[temp] = np.asarray(
                model.site_pos[site_id], dtype=float
            ).copy()
        if temp not in self._water_stream_original_half_height:
            self._water_stream_original_half_height[temp] = float(
                model.site_size[site_id][1]
            )
        return (
            site_id,
            np.asarray(self._water_stream_original_local_pos[temp], dtype=float),
            float(self._water_stream_original_half_height[temp]),
        )

    def _restore_sensor_stream_geometry(self, temperature: str) -> dict[str, Any]:
        site_id, original_pos, original_half_height = self._sensor_stream_geometry(
            temperature
        )
        self.sim.model.site_pos[site_id] = original_pos
        self.sim.model.site_size[site_id][1] = original_half_height
        self._sensor_stream_intercepted_for[_temperature(temperature)] = None
        return {
            "temperature": _temperature(temperature),
            "restored": True,
            "half_height": original_half_height,
        }

    def _set_sensor_stream_visibility(
        self, temperature: str, *, visible: bool
    ) -> None:
        try:
            site_id, _, _ = self._sensor_stream_geometry(temperature)
            self.sim.model.site_rgba[site_id][3] = 0.52 if visible else 0.0
        except Exception:
            pass

    def _answer_button_press_edges(self) -> tuple[str, ...]:
        edges: list[str] = []
        for temperature in TEMPERATURES:
            depth = self.answer_button_press_depth(temperature)
            if self._answer_button_armed[temperature] and depth >= self.button_pressed_threshold_m:
                edges.append(temperature)
                self._answer_button_armed[temperature] = False
                self._answer_button_release_steps[temperature] = 0
            elif not self._answer_button_armed[temperature]:
                if depth <= self.button_released_threshold_m:
                    self._answer_button_release_steps[temperature] += 1
                    if self._answer_button_release_steps[temperature] >= self.button_release_dwell_steps:
                        self._answer_button_armed[temperature] = True
                        self._answer_button_release_steps[temperature] = 0
                else:
                    self._answer_button_release_steps[temperature] = 0
        return tuple(edges)

    def _answer_button_positions(self) -> dict[str, tuple[float, float, float]]:
        defaults = {"hot": (1.856, -2.994, 0.93), "cold": (1.856, -3.074, 0.93)}
        path = DEFAULT_ANSWER_BUTTON_POSITIONS_FILE if self.answer_button_positions_file is None else Path(self.answer_button_positions_file)
        if not path.is_file():
            return defaults
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("positions", payload)
        for temperature in TEMPERATURES:
            value = np.asarray(raw[temperature], dtype=float).reshape(-1)
            if value.size != 3 or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid WA-05 {temperature} answer-button position")
            defaults[temperature] = tuple(float(item) for item in value)
        return defaults


def _default_semantic_state() -> SemanticState:
    return SemanticState()


def _semantic_for_plan(plan: EpisodePlan | None) -> SemanticState:
    del plan
    return SemanticState()


def _load_sensor_faucet_layout(value: str | None) -> dict[str, Any]:
    path = DEFAULT_SENSOR_FAUCET_LAYOUT_FILE if value is None else Path(str(value))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("config_kind") != "wa05_sensor_faucet_layout":
        raise ValueError(f"invalid WA-05 sensor-faucet resource: {path}")
    nozzles = payload.get("nozzle_positions")
    if not isinstance(nozzles, Mapping) or set(nozzles) != set(TEMPERATURES):
        raise ValueError("WA-05 sensor-faucet layout requires hot and cold nozzles")
    resolved_nozzles: dict[str, list[float]] = {}
    for temperature in TEMPERATURES:
        position = np.asarray(nozzles[temperature], dtype=float).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError(f"invalid WA-05 {temperature} sensor-faucet position")
        resolved_nozzles[temperature] = position.astype(float).tolist()
    raw_offsets = payload.get("spout_local_x_offsets")
    if not isinstance(raw_offsets, Mapping) or set(raw_offsets) != set(TEMPERATURES):
        raise ValueError("WA-05 sensor-faucet layout requires hot and cold spout offsets")
    offsets = {temperature: float(raw_offsets[temperature]) for temperature in TEMPERATURES}
    if not all(np.isfinite(item) for item in offsets.values()):
        raise ValueError("WA-05 sensor-faucet spout offsets must be finite")
    raw_yaws = payload.get("spout_local_yaw_degrees")
    if not isinstance(raw_yaws, Mapping) or set(raw_yaws) != set(TEMPERATURES):
        raise ValueError("WA-05 sensor-faucet layout requires hot and cold spout yaws")
    yaws = {temperature: float(raw_yaws[temperature]) for temperature in TEMPERATURES}
    if not all(np.isfinite(item) for item in yaws.values()):
        raise ValueError("WA-05 sensor-faucet spout yaws must be finite")
    numeric = {
        name: float(payload[name])
        for name in (
            "sensor_xy_radius",
            "sensor_min_cup_z",
            "sensor_max_cup_z_above_nozzle",
            "stream_half_height",
        )
    }
    if not all(np.isfinite(item) for item in numeric.values()):
        raise ValueError("WA-05 sensor-faucet layout values must be finite")
    if numeric["sensor_xy_radius"] <= 0.0 or numeric["stream_half_height"] <= 0.0:
        raise ValueError("WA-05 sensor radius and stream height must be positive")
    zone_settle = int(payload.get("zone_settle_steps", 0) or 0)
    if zone_settle < 0:
        raise ValueError("WA-05 zone_settle_steps must be non-negative")
    return {
        **numeric,
        "zone_settle_steps": zone_settle,
        "nozzle_positions": resolved_nozzles,
        "spout_local_x_offsets": offsets,
        "spout_local_yaw_degrees": yaws,
        "source": str(path),
    }


def _hide_legacy_sink_controls(
    worldbody: ET.Element,
    *,
    water_site_name: str,
    handle_joint_name: str,
) -> None:
    """Keep RoboSuite's named nodes while hiding the superseded controls.

    RoboSuite builds its fixture-id tables from the pre-composition model and
    expects every named geom to survive in the final MJCF.  Removing the old
    spout / handle mesh geoms therefore makes reset fail even though MuJoCo can
    compile the edited XML.  Alpha-zero geoms preserve those identifiers while
    leaving only the two bench-owned faucet clones visible.
    """

    targets: list[ET.Element] = []
    for body in worldbody.iter("body"):
        if any(site.get("name") == water_site_name for site in body.findall("site")):
            targets.append(body)
        if any(joint.get("name") == handle_joint_name for joint in body.findall("joint")):
            targets.append(body)
    for body in targets:
        for geom in body.iter("geom"):
            geom.set("rgba", "0 0 0 0")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def _temperature(value: str) -> str:
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURES:
        raise ValueError(f"invalid WA-05 water temperature: {value!r}")
    return temperature


__all__ = [
    "IDENTICAL_WATER_CUP_XML",
    "WATER_CUP_OBJECT_SCALE",
    "WA05FillOneCupAndMatchTemperatureButton",
]
