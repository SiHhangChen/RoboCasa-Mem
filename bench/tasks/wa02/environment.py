"""RoboCasa scene and physical-state boundary for WA-02."""

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

from bench.tasks.wa02_button_asset import add_wa02_button_asset
from bench.tasks.wa02.domain.contracts import CUP_IDS, EpisodePlan, SERVICE_SLOTS
from bench.tasks.wa02.domain.geometry import (
    CUP_PLACEMENT_SALTS,
    cup_placement_rng,
    resolve_cup_region_constraints,
    resolve_cup_sampling_regions,
    validate_cup_region_layout,
)
from bench.tasks.wa02.domain.semantics import (
    SemanticState,
    evaluate_success,
    latch_cups_staged,
    record_final_take,
    record_verified_fill,
    record_verified_staging,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_BUTTON_POSITIONS_FILE = (
    PACKAGE_ROOT / "resources" / "poses" / "water_button_positions_layout15_style34_seed0.json"
)
IDENTICAL_WATER_CUP_XML = f"{assets_root}/objects/objaverse/cup/cup_2/model.xml"
WATER_BUTTON_GEOM_NAMES = {
    "hot": "bench_wa02_hot_water_button",
    "cold": "bench_wa02_cold_water_button",
}
WATER_BUTTON_ASSET_SCALE = 0.5
WATER_BUTTON_DEFAULT_POSITIONS = {
    # The calibrated plunger centers sit 8 cm apart on the counter strip. At
    # half scale, the complete legacy housings retain a 2 cm visual gap; hot is
    # on the robot's left (+world Y), cold on its right (-world Y).
    "hot": (1.856, -2.994, 0.93),
    "cold": (1.856, -3.074, 0.93),
}
DEFAULT_CUP_SAMPLING_REGIONS = {
    "cup_A": {
        "region_center_xy": (0.5410, -0.5000),
        "region_size_xy": (0.2000, 0.1800),
        "yaw_rad": -0.6302,
    },
    "cup_B": {
        "region_center_xy": (0.8110, -0.5000),
        "region_size_xy": (0.2000, 0.1800),
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


class WA02ServeHotColdWaterCups(Kitchen):
    """Physical WA-02 world; episode choices and workflow live outside this class."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        water_button_positions_file: str | Path | None = None,
        service_threshold: float = 0.12,
        cup_upright_threshold_deg: float = 15.0,
        cup_placement_seed: int = 0,
        cup_sampling_regions: Mapping[str, Any] | None = None,
        cup_region_constraints: Mapping[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.water_button_positions_file = (
            None if water_button_positions_file is None else str(water_button_positions_file)
        )
        self.service_threshold = float(service_threshold)
        self.cup_upright_threshold_deg = float(cup_upright_threshold_deg)
        self._cup_placement_seed = int(cup_placement_seed)
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
        self._water_stream_original_local_pos: np.ndarray | None = None
        self._water_stream_original_half_height: float | None = None
        self._water_stream_intercept_trace: dict[str, Any] = {}
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
            raise RuntimeError("WA-02 cup reset surface was not resolved")
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
                        f"WA-02 placement initializer is missing {sampler_name}"
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
                "WA-02 sampled placement is missing cups: "
                + ", ".join(sorted(missing))
            )
        self.object_placements = placements

    def _resolve_cup_counter_reset_region(self) -> dict[str, Any]:
        """Find the one counter surface containing both declared regions."""

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
                "WA-02 expected exactly one counter reset surface for both cup "
                f"regions, found {len(matches)}"
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
        """Install complete bench-owned hot/cold button assemblies."""

        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_str
        existing = {geom.get("name") for geom in worldbody.iter("geom") if geom.get("name")}
        positions = self._water_button_positions()
        for temperature, geom_name in WATER_BUTTON_GEOM_NAMES.items():
            if geom_name not in existing:
                _add_water_button_asset(
                    worldbody,
                    geom_name=geom_name,
                    pos=positions[temperature],
                    rgba="0.88 0.31 0.24 1" if temperature == "hot" else "0.24 0.54 0.90 1",
                    glow_rgba=(
                        "1.00 0.36 0.24 1"
                        if temperature == "hot"
                        else "0.32 0.66 1.00 1"
                    ),
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
        self.set_water_stream_intercept(None)
        for cup in CUP_IDS:
            self._set_water_liquid(cup, visible=False)

    # Episode and semantic events ---------------------------------------

    def configure_episode(self, episode: EpisodePlan) -> None:
        """Install a pre-resolved plan without adding labels to observations."""

        self._episode_plan = episode
        self._semantic_state = _semantic_for_plan(episode)
        self.cup_temperatures = {cup: None for cup in CUP_IDS}
        self.cup_filled = {cup: False for cup in CUP_IDS}
        self.cup_fill_progress = {cup: 0.0 for cup in CUP_IDS}
        self.set_water_stream_intercept(None)
        for cup in CUP_IDS:
            self._set_water_liquid(cup, visible=False)

    @property
    def episode_plan(self) -> EpisodePlan:
        if self._episode_plan is None:
            raise RuntimeError("WA-02 episode was not configured after reset")
        return self._episode_plan

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
            raise ValueError("WA-02 requires exactly two service placement regions")
        if set(sampled_relative_xy_by_slot) != set(SERVICE_SLOTS):
            raise ValueError("WA-02 requires one sampled target per service region")
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
                raise ValueError(f"invalid WA-02 service region {slot}")
            lower = center - 0.5 * size
            upper = center + 0.5 * size
            if np.any(sampled < lower) or np.any(sampled > upper):
                raise ValueError(f"sampled WA-02 service target escaped {slot}")
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

        applied = record_verified_fill(self._semantic_state, cup_id, temperature)
        if applied:
            self.cup_temperatures[str(cup_id)] = str(temperature)
            self.cup_filled[str(cup_id)] = True
            self.set_cup_fill_progress(str(cup_id), 1.0)
        return applied

    def record_verified_staging(self, cup_id: str, slot: str) -> bool:
        return record_verified_staging(self._semantic_state, cup_id, slot)

    def latch_cups_staged_before_final_pick(self) -> bool:
        return latch_cups_staged(
            self._semantic_state,
            currently_ready=self.are_cups_physically_staged(),
        )

    def record_final_take(self, cup_id: str) -> None:
        record_final_take(self._semantic_state, cup_id)

    # Physical queries ---------------------------------------------------

    def update_state(self) -> None:
        """Refresh only derived physical facts; it never writes hidden labels."""

        super().update_state()

    def object_position(self, cup_id: str) -> np.ndarray:
        return np.asarray(self.sim.data.body_xpos[self.obj_body_id[str(cup_id)]], dtype=float)

    def water_site_position(self) -> np.ndarray | None:
        try:
            site_name = self.sink.water_site.get("name")
            site_id = self.sim.model.site_name2id(site_name)
            return np.asarray(self.sim.data.site_xpos[site_id], dtype=float)
        except Exception:
            return None

    def water_button_position(self, temperature: str) -> np.ndarray:
        temp = str(temperature).strip().lower()
        if temp not in WATER_BUTTON_GEOM_NAMES:
            raise ValueError(f"invalid WA-02 water-button temperature: {temperature!r}")
        geom_name = WATER_BUTTON_GEOM_NAMES[temp]
        try:
            geom_id = self.sim.model.geom_name2id(geom_name)
            return np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
        except Exception:
            return np.asarray(self._water_button_positions()[temp], dtype=float)

    def service_slot_position(self, slot: str) -> np.ndarray:
        value = self.service_slot_poses.get(str(slot))
        if value is None:
            raise RuntimeError(f"WA-02 service slot {slot!r} has not been configured")
        return np.asarray(value, dtype=float)

    def service_slot_region(self, slot: str) -> dict[str, list[float]]:
        value = self.service_slot_regions.get(str(slot))
        if value is None:
            raise RuntimeError(f"WA-02 service region {slot!r} is not configured")
        return deepcopy(value)

    def cup_is_upright(self, cup_id: str) -> bool:
        rotation = np.asarray(
            self.sim.data.body_xmat[self.obj_body_id[str(cup_id)]], dtype=float
        ).reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        return bool(tilt <= self.cup_upright_threshold_deg)

    def cup_is_at_service_slot(self, cup_id: str, slot: str) -> bool:
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
            self.cup_is_at_service_slot(cup, self._episode_plan.service_slot_by_cup[cup])
            for cup in CUP_IDS
        )

    def set_verified_faucet_state(self, *, mode: str, temperature: str | None) -> None:
        """Apply fixture state after a verified physical button/faucet operation."""

        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"on", "off"}:
            raise ValueError("WA-02 faucet mode must be on or off")
        temp = None if temperature is None else str(temperature).strip().lower()
        if normalized_mode == "on" and temp not in {"hot", "cold"}:
            raise ValueError("WA-02 opening faucet requires hot or cold temperature")
        if temp == "hot":
            self.sink.set_temp_state(env=self, rng=self.rng, min_temp=0.65, max_temp=0.75)
        elif temp == "cold":
            self.sink.set_temp_state(env=self, rng=self.rng, min_temp=0.25, max_temp=0.35)
        self.sink.set_handle_state(env=self, rng=self.rng, mode=normalized_mode)
        try:
            self.sim.data.set_joint_qpos(f"{self.sink.naming_prefix}spout_joint", 0.0)
            self.sim.forward()
        except Exception:
            pass
        self.update_state()

    def faucet_matches(self, *, mode: str, temperature: str | None = None) -> bool:
        state = dict(self.sink.get_handle_state(env=self))
        expected_on = str(mode).strip().lower() == "on"
        if bool(state.get("water_on", False)) != expected_on:
            return False
        if temperature is None:
            return True
        temp = str(temperature).strip().lower()
        return bool(
            str(state.get("water_temp_state", "")).lower() == temp
            or (temp == "hot" and float(state.get("water_temp", 0.0)) >= 0.60)
            or (temp == "cold" and float(state.get("water_temp", 1.0)) <= 0.40)
        )

    # Diagnostics and task result --------------------------------------

    def _check_success(self) -> bool:
        self.update_state()
        return evaluate_success(self._semantic_state).success

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
            "oracle": {
                "hot_cup_id": self._semantic_state.hot_cup_id,
                "cold_cup_id": self._semantic_state.cold_cup_id,
                "requested_temperature": self._semantic_state.requested_temperature,
                "requested_cup_id": self._semantic_state.requested_cup_id,
                "service_slot_by_cup": dict(self._semantic_state.service_slot_by_cup),
            },
            "fill": {
                "cup_filled": dict(self.cup_filled),
                "cup_fill_progress": dict(self.cup_fill_progress),
                "cup_temperatures": dict(self.cup_temperatures),
                "recorded_temperature_by_cup": dict(
                    self._semantic_state.filled_temperature_by_cup
                ),
                "water_stream_intercept": dict(self._water_stream_intercept_trace),
            },
            "service": {
                "slot_positions_world": slots,
                "placement_regions": deepcopy(self.service_slot_regions),
                "recorded_slot_by_cup": dict(self._semantic_state.staged_slot_by_cup),
                "physically_staged": self.are_cups_physically_staged(),
                "staged_before_final_pick": self._semantic_state.cups_staged_before_final_pick,
            },
            "selection": {"final_taken_cup_id": self._semantic_state.final_taken_cup_id},
        }

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.wa02.prompts import resolve_prompt_spec

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(
            requested_temperature=self.requested_temperature,
            variant="memory",
        )
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
            raise ValueError(f"invalid WA-02 cup id: {cup_id!r}")
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

    def set_water_stream_intercept(self, cup_id: str | None) -> dict[str, Any]:
        """Restore the full stream or end its visible cylinder at a cup rim."""

        try:
            model = self.sim.model
            site_name = str(self.sink.water_site.get("name"))
            site_id = int(model.site_name2id(site_name))
            if self._water_stream_original_local_pos is None:
                self._water_stream_original_local_pos = np.asarray(
                    model.site_pos[site_id], dtype=float
                ).copy()
            if self._water_stream_original_half_height is None:
                self._water_stream_original_half_height = float(
                    model.site_size[site_id][1]
                )
            original_pos = np.asarray(
                self._water_stream_original_local_pos, dtype=float
            )
            original_half_height = float(
                self._water_stream_original_half_height
            )
            model.site_pos[site_id] = original_pos
            model.site_size[site_id][1] = original_half_height
            if cup_id is None:
                self.sim.forward()
                result = {
                    "restored": True,
                    "intercepted": False,
                    "half_height": original_half_height,
                }
                self._water_stream_intercept_trace = result
                return result

            cup = str(cup_id)
            if cup not in CUP_IDS:
                raise ValueError(f"invalid WA-02 cup id: {cup_id!r}")
            self.sim.forward()
            body_id = int(self.obj_body_id[cup])
            body_position = np.asarray(
                self.sim.data.body_xpos[body_id], dtype=float
            )
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
                raise RuntimeError("WA-02 sink water stream is not vertically aligned")
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
            self.sim.forward()
            intercepted = bool(
                truncated_half_height < original_half_height - 1.0e-6
            )
            result = {
                "restored": False,
                "intercepted": intercepted,
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
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._water_stream_intercept_trace = result
            return result

    def water_stream_intercept_callback(self, cup_id: str):
        """Update the stream before each recorded frame as a cup enters/exits."""

        cup = str(cup_id)
        if cup not in CUP_IDS:
            raise ValueError(f"invalid WA-02 cup id: {cup_id!r}")
        active = bool(self._water_stream_intercept_trace.get("intercepted", False))

        def callback(obs: dict[str, Any]) -> dict[str, Any] | None:
            nonlocal active
            try:
                under_water = bool(self.sink.check_obj_under_water(self, cup))
            except Exception:
                under_water = False
            if under_water:
                result = self.set_water_stream_intercept(cup)
                active = bool(result.get("intercepted", False))
                return self._get_observations(force_update=True)
            if active:
                self.set_water_stream_intercept(None)
                active = False
                return self._get_observations(force_update=True)
            return None

        return callback

    def _water_button_positions(self) -> dict[str, tuple[float, float, float]]:
        positions = dict(WATER_BUTTON_DEFAULT_POSITIONS)
        path = _button_positions_path(self.water_button_positions_file)
        if path is None or not path.is_file():
            return positions
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("positions", payload)
        if not isinstance(raw, dict):
            raise ValueError(f"invalid WA-02 water button resource: {path}")
        for temperature in WATER_BUTTON_GEOM_NAMES:
            value = raw.get(temperature)
            array = np.asarray(value, dtype=float).reshape(-1) if value is not None else None
            if array is None:
                continue
            if array.size < 3 or not np.all(np.isfinite(array[:3])):
                raise ValueError(f"invalid WA-02 {temperature} button position in {path}")
            positions[temperature] = tuple(float(item) for item in array[:3])
        return positions


def _default_semantic_state() -> SemanticState:
    return SemanticState(
        hot_cup_id="cup_A",
        cold_cup_id="cup_B",
        requested_temperature="hot",
        requested_cup_id="cup_A",
        service_slot_by_cup={"cup_A": "slot_1", "cup_B": "slot_2"},
    )


def _semantic_for_plan(plan: EpisodePlan | None) -> SemanticState:
    if plan is None:
        return _default_semantic_state()
    return SemanticState(
        hot_cup_id=plan.hot_cup_id,
        cold_cup_id=plan.cold_cup_id,
        requested_temperature=plan.requested_temperature,
        requested_cup_id=plan.requested_cup_id,
        service_slot_by_cup=dict(plan.service_slot_by_cup),
    )


def _button_positions_path(value: str | None) -> Path | None:
    if value is None:
        return DEFAULT_BUTTON_POSITIONS_FILE
    text = str(value).strip()
    if text.lower() in {"", "none", "off", "false"}:
        return None
    return Path(text)


def _add_water_button_asset(
    parent: ET.Element,
    *,
    geom_name: str,
    pos: tuple[float, float, float],
    rgba: str,
    glow_rgba: str,
) -> None:
    """Install the complete legacy WA02 housing around the live plunger."""

    add_wa02_button_asset(
        parent,
        geom_name=geom_name,
        pos=pos,
        color_rgba=rgba,
        glow_rgba=glow_rgba,
        scale=WATER_BUTTON_ASSET_SCALE,
    )


__all__ = [
    "IDENTICAL_WATER_CUP_XML",
    "WATER_BUTTON_ASSET_SCALE",
    "WATER_BUTTON_GEOM_NAMES",
    "WA02ServeHotColdWaterCups",
]
