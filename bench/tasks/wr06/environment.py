"""Independent RoboCasa scene and physical-state boundary for WR-06.

One real ketchup bottle starts on one open-shelf level, is staged beside a
fixed serving plate on the dining island (plate_left or plate_right, seed
decided), and must be returned to its remembered shelf level. The plate asset
and its world position follow the WR-05 plate-staging scene.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.utils import object_utils as OU

from bench.tasks.wr06.asset_poses import (
    SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M,
    SHELF_UPRIGHT_ROTATION_AXIS,
    SHELF_UPRIGHT_ROTATION_RAD,
)
from bench.tasks.wr06.domain.contracts import (
    HOME_LEVELS,
    ITEM_ID,
    PLATE_SIDE_WORLD_OFFSETS_M,
    PLATE_SIDES,
)
from bench.tasks.wr06.domain.episode import validate_level, validate_plate_stage_region
from bench.tasks.wr06.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_memory_barrier,
    record_initial_level,
    record_transport_posture_normalized,
)
from bench.tasks.wr06.physical_events import PhysicalSample, WR06PhysicalObserver


STORAGE_FIXTURE_NAME = "opencabinet_1_main_group_1"
STORAGE_FIXTURE_CENTER_Z = 1.58
MEAL_FIXTURE_NAME = "island_island_group_1"
MEAL_PAPER_TOWEL_FIXTURE_NAME = "papertowel_island_group_1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ITEM_ASSETS = {
    ITEM_ID: str(
        _REPOSITORY_ROOT
        / "robocasa/robocasa/models/assets/objects/aigen_objs/ketchup/ketchup_3/model.xml"
    )
}
DEFAULT_FIXED_PLATE_ASSET = str(
    _REPOSITORY_ROOT
    / "robocasa/robocasa/models/assets/objects/objaverse/plate/plate_14/model.xml"
)
# Legacy HS-WR-02 calibrated serving plate center on the dining island.
DEFAULT_FIXED_PLATE_WORLD_POS = (1.6902007516630673, -2.701823059087081, 0.9291613871832717)
DEFAULT_HOME_REGION_PLACEMENTS = {
    "shelf_lower": {
        "shelf_level": 0,
        # Layout 32 has a wall beside the shelf's low-world-X edge.  Shift and
        # narrow the span so shelf approaches retain arm clearance from it.
        "size": (0.58, 0.13),
        "pos": (0.25, -1.0),
        "margin": 0.04,
    },
    "shelf_upper": {
        "shelf_level": 1,
        "size": (0.58, 0.13),
        "pos": (0.25, -1.0),
        "margin": 0.04,
    },
}


class WR06ReturnSingleBottleToOriginalLevel(Kitchen):
    """One ketchup bottle starts on one shelf level and must return there."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        *args: Any,
        original_level: str = "shelf_lower",
        plate_stage_region: str = "plate_left",
        placement_pair_seed: int = 0,
        home_region_placements: dict[str, dict[str, Any]] | None = None,
        plate_stage_threshold: float = 0.20,
        return_region_xy_margin: float = 0.04,
        return_region_height_tolerance: float = 0.10,
        physical_event_dwell_steps: int = 3,
        terminal_verification_steps: int = 100,
        **kwargs: Any,
    ) -> None:
        self.demo_original_level = validate_level(original_level)
        self.plate_stage_region = validate_plate_stage_region(plate_stage_region)
        self.placement_pair_seed = int(placement_pair_seed)
        self.home_region_placements = _validate_home_region_placements(
            home_region_placements
        )
        self.plate_stage_threshold = float(plate_stage_threshold)
        self.return_region_xy_margin = float(return_region_xy_margin)
        self.return_region_height_tolerance = float(return_region_height_tolerance)
        self.physical_event_dwell_steps = int(physical_event_dwell_steps)
        self.terminal_verification_steps = int(terminal_verification_steps)
        if self.physical_event_dwell_steps < 1:
            raise ValueError("WR-06 physical-event dwell must be positive")
        if self.terminal_verification_steps < 1:
            raise ValueError("WR-06 terminal verification must be positive")
        self._semantic_state = SemanticState()
        self._physical_observer: WR06PhysicalObserver | None = None
        self._physical_event_diagnostics: dict[str, Any] = {}
        self._object_center_surface_offset_z = 0.0
        kwargs.setdefault("obj_registries", ("objaverse", "aigen", "lightwheel"))
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.item_shelf = self.register_fixture_ref(
            "item_shelf", dict(id=STORAGE_FIXTURE_NAME, full_name_check=True)
        )
        shelf_pos = np.asarray(self.item_shelf.pos, dtype=float).copy()
        shelf_pos[2] = STORAGE_FIXTURE_CENTER_Z
        self.item_shelf.set_pos(shelf_pos)
        self.meal_counter = self.register_fixture_ref(
            "meal_counter",
            dict(id=MEAL_FIXTURE_NAME, full_name_check=True, full_depth_region=True),
        )
        self.meal_paper_towel = self.register_fixture_ref(
            "meal_paper_towel",
            dict(id=MEAL_PAPER_TOWEL_FIXTURE_NAME, full_name_check=True),
        )
        if self.item_shelf is None or self.meal_counter is None:
            raise ValueError(
                "WR-06 requires layout 32 with its open shelf and dining island"
            )
        self.plate_stage_fixture = self.meal_counter
        self.init_robot_base_ref = self.item_shelf

    def _configure_condiment_contact_physics(self) -> None:
        """Tune contact friction for the real ketchup bottle.

        The recorded action stream closes the Panda fingers on the bottle and
        carries it through long mobile-base motions.  The imported mesh asset
        uses a low tangential friction default on some collision geoms, so the
        bottle slips out despite genuine finger contacts.  Raising tangential
        friction changes only MuJoCo's contact solve; it never attaches,
        teleports, or otherwise kinematically drives the object.
        """
        model = getattr(getattr(self, "sim", None), "model", None)
        if model is None or not hasattr(model, "geom_bodyid"):
            return
        for name in (ITEM_ID,):
            body_id = self.obj_body_id.get(str(name))
            if body_id is None:
                continue
            for geom_id, geom_body_id in enumerate(model.geom_bodyid):
                if int(geom_body_id) != int(body_id):
                    continue
                # Keep rolling/torsional terms from the source asset while
                # raising sliding friction to a stable, physically plausible
                # rubber-like bottle/pad contact.
                model.geom_friction[geom_id, 0] = max(
                    float(model.geom_friction[geom_id, 0]), 2.5
                )
                model.geom_friction[geom_id, 1] = max(
                    float(model.geom_friction[geom_id, 1]), 0.5
                )

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        placement = self.home_region_placements[self.demo_original_level]
        cfgs = [
            dict(
                name=ITEM_ID,
                obj_groups=DEFAULT_ITEM_ASSETS[ITEM_ID],
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.item_shelf,
                    sample_region_kwargs=dict(shelf_level=int(placement["shelf_level"])),
                    size=tuple(placement["size"]),
                    margin=float(placement["margin"]),
                    rotation=SHELF_UPRIGHT_ROTATION_RAD,
                    rotation_axis=SHELF_UPRIGHT_ROTATION_AXIS,
                    offset=(0.0, 0.0, SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M[ITEM_ID]),
                    pos=tuple(placement["pos"]),
                ),
            )
        ]
        cfgs.append(
            dict(
                name="plate",
                obj_groups=DEFAULT_FIXED_PLATE_ASSET,
                placement=dict(
                    fixture=self.plate_stage_fixture,
                    sample_region_kwargs=dict(
                        ref=self.meal_paper_towel, loc="right"
                    ),
                    size=(0.46, 0.42),
                    pos=(1.0, 0.0),
                    rotation=0.0,
                ),
            )
        )
        return cfgs

    def _reset_internal(self) -> None:
        self._resample_initial_item()
        super()._reset_internal()
        if "plate" in self.obj_body_id:
            self.set_object_body_position("plate", DEFAULT_FIXED_PLATE_WORLD_POS)

    def _resample_initial_item(self) -> None:
        """Pair lower/upper cases with the same task-owned placement RNG."""
        initializer = getattr(self, "placement_initializer", None)
        if initializer is None:
            return
        children = getattr(initializer, "samplers", {})
        child_samplers = (
            list(children.values()) if hasattr(children, "values") else list(children)
        )
        samplers = [initializer, *child_samplers]
        previous_rngs = [getattr(sampler, "rng", None) for sampler in samplers]
        placement_rng = np.random.default_rng(self.placement_pair_seed)
        try:
            for sampler in samplers:
                sampler.rng = placement_rng
            placements = initializer.sample(placed_objects=self.fxtr_placements)
        finally:
            for sampler, previous_rng in zip(samplers, previous_rngs):
                sampler.rng = previous_rng
        if ITEM_ID not in placements:
            raise RuntimeError(f"WR-06 sampled placement is missing {ITEM_ID!r}")
        self.object_placements = placements

    def _setup_scene(self) -> None:
        # A reset may reuse the same environment instance.  Do not let the
        # previous episode's observer consume partially reset physics.
        self._physical_observer = None
        super()._setup_scene()
        self._configure_condiment_contact_physics()
        self._semantic_state = SemanticState()
        placement = getattr(self, "object_placements", {}).get(ITEM_ID)
        if placement is None:
            raise RuntimeError("WR-06 reset is missing the sampled bottle placement")
        sampled_pos = np.asarray(placement[0], dtype=float).reshape(-1)[:3]
        actual_level = self.detect_actual_level(position=sampled_pos)
        if actual_level != self.demo_original_level:
            raise RuntimeError(
                f"WR-06 physical initial level {actual_level!r} differs from episode "
                f"level {self.demo_original_level!r}"
            )
        self._object_center_surface_offset_z = float(
            sampled_pos[2] - self.home_region_surface_z(actual_level)
        )
        # The sampled reset XY is the physically stable home spot on the shelf
        # level; returning to the shelf-region center can land the bottle on an
        # unstable slope, so the return target follows the original home XY.
        self._original_home_xy = np.asarray(sampled_pos[:2], dtype=float).copy()
        physically_verified = bool(
            self._position_in_level(sampled_pos, actual_level)
            and not self._position_in_level(sampled_pos, _other_level(actual_level))
        )
        if not record_initial_level(
            self._semantic_state,
            actual_level,
            physically_verified=physically_verified,
        ):
            raise RuntimeError(
                "WR-06 failed to record the physically verified initial level: "
                f"actual={actual_level}, "
                f"in_actual={self._position_in_level(sampled_pos, actual_level)}, "
                f"in_other={self._position_in_level(sampled_pos, _other_level(actual_level))}, "
                f"sampled_pos={sampled_pos.tolist()}, "
                f"surface_offset={self._object_center_surface_offset_z}"
            )
        self._physical_observer = WR06PhysicalObserver(
            self._semantic_state,
            dwell_steps=self.physical_event_dwell_steps,
            terminal_verification_steps=self.terminal_verification_steps,
        )
        update = self._physical_observer.reset(self._physical_event_sample())
        self._physical_event_diagnostics = dict(update.diagnostics)
        self._sync_diagnostics()

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.wr06.prompts import resolve_prompt_spec

        meta = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        meta["lang"] = prompt.policy_prompt
        meta["initial_prompt"] = prompt.initial_prompt
        meta.setdefault("refs", {}).update(
            {
                "item_id": ITEM_ID,
                "item_label": "ketchup bottle",
                "home_levels": list(HOME_LEVELS),
                "plate_stage_region": self.plate_stage_region,
            }
        )
        return meta

    def update_state(self) -> None:
        super().update_state()
        self._update_task_state()
        self._sync_diagnostics()

    def configure_episode(self, *, original_level: str) -> None:
        expected = validate_level(original_level)
        if expected != self.demo_original_level:
            raise ValueError("WR-06 environment and EpisodePlan original levels differ")
        actual = self.detect_actual_level()
        if actual != expected or self._semantic_state.initial_level != actual:
            raise ValueError("WR-06 initial physical level verification failed")

    # Semantic event adapters ------------------------------------------

    def record_transport_posture_normalized(self, *, physically_normalized: bool) -> bool:
        return record_transport_posture_normalized(
            self._semantic_state, physically_normalized=physically_normalized
        )

    def latch_original_level_hidden(self) -> bool:
        physical = self.physical_state()
        return latch_memory_barrier(
            self._semantic_state,
            item_at_fixed_plate=physical.item_at_fixed_plate,
            item_outside_shelves=physical.item_outside_shelves,
        )

    def begin_return_placement(self) -> bool:
        observer = self._physical_observer
        return bool(observer is not None and observer.begin_return_placement())

    def arm_return_verification(self) -> bool:
        observer = self._physical_observer
        return bool(observer is not None and observer.arm_return_verification())

    # Physical observation boundary -----------------------------------

    def _obj_pos(self, object_name: str) -> np.ndarray:
        return np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[str(object_name)]], dtype=float
        )

    def detect_actual_level(self, *, position: Any | None = None) -> str:
        """Detect the nearest shelf surface from the simulated object pose."""
        pos = (
            self._obj_pos(ITEM_ID)
            if position is None
            else np.asarray(position, dtype=float).reshape(-1)[:3]
        )
        pos_z = float(pos[2])
        distances = {
            level: abs(pos_z - self.home_region_surface_z(level)) for level in HOME_LEVELS
        }
        ordered = sorted(distances, key=distances.get)
        if len(ordered) != 2 or distances[ordered[1]] - distances[ordered[0]] < 0.08:
            raise RuntimeError(f"WR-06 initial shelf level is ambiguous: {distances}")
        return ordered[0]

    def plate_stage_pose(self) -> np.ndarray:
        return np.asarray(self._obj_pos("plate"), dtype=float) + np.asarray(
            PLATE_SIDE_WORLD_OFFSETS_M[self.plate_stage_region], dtype=float
        )

    def use_station_pose(self, object_name: str) -> np.ndarray:
        if str(object_name) != ITEM_ID:
            raise KeyError(f"unknown WR-06 object {object_name!r}")
        return self.plate_stage_pose()

    def return_region_pose(self, level: str) -> np.ndarray:
        resolved = validate_level(level)
        center_xy, _ = self._home_region_xy_geometry(resolved)
        original_xy = getattr(self, "_original_home_xy", None)
        if original_xy is not None and resolved == self.demo_original_level:
            center_xy = original_xy
        return np.array(
            [
                float(center_xy[0]),
                float(center_xy[1]),
                self.home_region_surface_z(resolved) + self._object_center_surface_offset_z,
            ],
            dtype=float,
        )

    def home_region_center(self, level: str) -> np.ndarray:
        resolved = validate_level(level)
        center_xy, _ = self._home_region_xy_geometry(resolved)
        return np.array(
            [
                float(center_xy[0]),
                float(center_xy[1]),
                self.home_region_surface_z(resolved) + self._object_center_surface_offset_z,
            ],
            dtype=float,
        )

    def home_region_surface_z(self, level: str) -> float:
        settings = self.home_region_placements[validate_level(level)]
        regions = self.item_shelf.get_reset_regions(
            shelf_level=int(settings["shelf_level"])
        )
        if len(regions) != 1:
            raise ValueError(f"WR-06 {level} must resolve to one shelf region")
        region = next(iter(regions.values()))
        return float(np.asarray(self.item_shelf.pos, dtype=float)[2] + region["offset"][2])

    def item_in_level(self, level: str) -> bool:
        return self._position_in_level(self._obj_pos(ITEM_ID), level)

    def _position_in_level(self, position: Any, level: str) -> bool:
        resolved = validate_level(level)
        center, size = self._home_region_xy_geometry(resolved)
        pos = np.asarray(position, dtype=float).reshape(-1)[:3]
        rot = float(self.item_shelf.rot)
        inverse = np.array(
            [[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]], dtype=float
        )
        relative = inverse @ (pos[:2] - center)
        expected_z = self.home_region_surface_z(resolved) + self._object_center_surface_offset_z
        return bool(
            np.all(np.abs(relative) <= 0.5 * size + self.return_region_xy_margin)
            and abs(float(pos[2]) - expected_z) <= self.return_region_height_tolerance
        )

    def item_at_fixed_plate(self) -> bool:
        """Whether the bottle rests at the episode's staged plate side."""
        return self._item_at_plate_side(self.plate_stage_region)

    def _item_at_plate_side(self, region: str) -> bool:
        resolved = validate_plate_stage_region(region)
        pos = self._obj_pos(ITEM_ID)
        distance = float(
            np.linalg.norm(pos[:2] - self.plate_stage_pose()[:2])
        )
        if distance > self.plate_stage_threshold:
            return False
        supported = bool(
            OU.check_obj_fixture_contact(self, ITEM_ID, self.plate_stage_fixture)
        )
        if supported:
            return True
        # The generic fixture-contact helper can briefly report false while a
        # gripper approaches a resting bottle (observed at the plate re-pick).
        # The bottle bottom resting at the plate surface height is the physical
        # staging fact; use it as a fallback so a transient contact flicker
        # does not look like the bottle left the plate.
        try:
            from bench.execution.planning import object_reference_pos

            bottom_z = float(
                object_reference_pos(
                    self,
                    ITEM_ID,
                    placement_reference="bbox_bottom_center",
                )[2]
            )
        except Exception:
            return False
        return abs(bottom_z - float(self.plate_stage_pose()[2])) <= 0.03

    def _item_supported_by_shelf(self) -> bool:
        level = self._semantic_state.initial_level
        if level is None:
            return False
        return self._item_supported_by_level(level)

    def _item_supported_by_level(self, level: str) -> bool:
        shelf_index = int(self.home_region_placements[validate_level(level)]["shelf_level"])
        # RoboCasa's OpenCabinet does not register its dynamically created
        # level shelves in ``fixture.contact_geoms``.  Check the actual level
        # geom explicitly; the generic fixture-contact helper is therefore
        # false even for a valid reset placement.
        shelf_geom = f"{self.item_shelf.name}_level{shelf_index}_shelf"
        if bool(self.check_contact(self.objects[ITEM_ID], shelf_geom)):
            return True
        # The released bottle can settle on the flat shelf for a few steps
        # after the open TCP retreats, during which the geom contact check can
        # flicker.  The bottle bottom resting at the shelf surface height is
        # the physical support fact; use it as a fallback.
        try:
            from bench.execution.planning import object_reference_pos

            bottom_z = float(
                object_reference_pos(
                    self,
                    ITEM_ID,
                    placement_reference="bbox_bottom_center",
                )[2]
            )
        except Exception:
            return False
        return abs(bottom_z - self.home_region_surface_z(level)) <= 0.03

    def physical_state(self) -> PhysicalState:
        return PhysicalState(
            item_at_fixed_plate=self.item_at_fixed_plate(),
            item_in_lower=self.item_in_level("shelf_lower"),
            item_in_upper=self.item_in_level("shelf_upper"),
            item_supported_by_lower=self._item_supported_by_level("shelf_lower"),
            item_supported_by_upper=self._item_supported_by_level("shelf_upper"),
        )

    def _item_grasped(self) -> bool:
        """Return strict grasp evidence for the plate-side departure gate."""
        try:
            # Contact alone is insufficient: an open gripper can bump the
            # bottle off the plate. Require the same 0.6 threshold as pick
            # checks and fingers that have actually closed around the item.
            robot = self.robots[0]
            qpos = np.asarray(
                robot.get_gripper_joint_positions("right"),
                dtype=float,
            ).reshape(-1)
            if qpos.size < 2 or not np.all(np.isfinite(qpos)):
                return False
            # Panda's open limits are +/-0.04 m. A 0.01 m inward margin
            # distinguishes an open collision from a closed/loaded grasp.
            gripper_closed = bool(np.max(np.abs(qpos[:2])) <= 0.03)
            if not gripper_closed:
                return False
            return bool(
                OU.check_obj_grasped(
                    self,
                    ITEM_ID,
                    threshold=0.6,
                )
            )
        except Exception:
            return False

    def _check_success(self) -> bool:
        self._update_task_state()
        return self._success_report().success

    def _success_report(self):
        return evaluate_success(self._semantic_state, self.physical_state())

    def task_diagnostics(self) -> dict[str, Any]:
        self._update_task_state()
        report = self._success_report()
        physical = self.physical_state()
        violation_reason = self._physical_event_diagnostics.get("violation_reason")
        return {
            "success": report.success,
            "terminal": bool(report.success or self._semantic_state.protocol_violation),
            "failure_code": (
                None
                if not self._semantic_state.protocol_violation
                else str(violation_reason or "protocol_violation")
            ),
            **report.checks(),
            "current_item_position": self._obj_pos(ITEM_ID).astype(float).tolist(),
            "current_item_at_fixed_plate": physical.item_at_fixed_plate,
            "current_item_in_lower": physical.item_in_lower,
            "current_item_in_upper": physical.item_in_upper,
            "wrong_return_level_attempted": self._semantic_state.wrong_return_level_attempted,
            "protocol_violation": self._semantic_state.protocol_violation,
            "stage_regressions": list(self._semantic_state.stage_regressions),
            "physical_events": dict(self._physical_event_diagnostics),
        }

    def _update_task_state(self) -> None:
        observer = self._physical_observer
        if observer is None:
            return
        update = observer.observe(self._physical_event_sample())
        self._physical_event_diagnostics = {
            **dict(update.diagnostics),
            "last_step": int(update.step_id),
            "last_events": list(update.semantic_events),
            "duplicate_step": bool(update.duplicate_step),
        }

    def _physical_event_sample(self) -> PhysicalSample:
        physical = self.physical_state()
        return PhysicalSample(
            step_id=int(getattr(self, "timestep", 0)),
            item_at_fixed_plate=physical.item_at_fixed_plate,
            item_in_lower=physical.item_in_lower,
            item_in_upper=physical.item_in_upper,
            item_supported_by_lower=physical.item_supported_by_lower,
            item_supported_by_upper=physical.item_supported_by_upper,
            item_grasped=self._item_grasped(),
        )

    def _sync_diagnostics(self) -> None:
        state = self._semantic_state
        self.initial_level_physically_verified = state.initial_level_physically_verified
        self.item_staged_at_fixed_plate = state.item_staged_at_plate
        self.memory_barrier_latched = state.memory_barrier_latched
        self.wrong_return_level_attempted = state.wrong_return_level_attempted
        self.protocol_violation = state.protocol_violation
        self.stage_regressions = list(state.stage_regressions)

    def set_object_pose(self, obj_name: str, pos: Any, quat_xyzw: Any | None = None) -> None:
        import robosuite.utils.transform_utils as T

        obj = self.objects[str(obj_name)]
        pos_arr = np.array(pos, dtype=float).reshape(-1)
        if pos_arr.size < 3 or not np.all(np.isfinite(pos_arr[:3])):
            raise ValueError(f"invalid object position for {obj_name!r}")
        if quat_xyzw is None:
            quat_wxyz = np.array(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float)[3:7]
        else:
            quat_wxyz = T.convert_quat(np.array(quat_xyzw, dtype=float).reshape(4), to="wxyz")
        self.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([pos_arr[:3], quat_wxyz]),
        )
        self.sim.forward()

    def set_object_body_position(self, obj_name: str, pos: Any) -> None:
        """Set a free object's body center while preserving its joint-frame offset."""
        obj = self.objects[str(obj_name)]
        target = np.asarray(pos, dtype=float).reshape(-1)
        if target.size < 3 or not np.all(np.isfinite(target[:3])):
            raise ValueError(f"invalid body position for {obj_name!r}")
        joint_qpos = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).copy()
        joint_qpos[:3] += target[:3] - self._obj_pos(str(obj_name))[:3]
        self.sim.data.set_joint_qpos(obj.joints[0], joint_qpos)
        self.sim.forward()

    def _home_region_xy_geometry(self, level: str) -> tuple[np.ndarray, np.ndarray]:
        settings = self.home_region_placements[validate_level(level)]
        regions = self.item_shelf.get_reset_regions(
            shelf_level=int(settings["shelf_level"])
        )
        if len(regions) != 1:
            raise ValueError(f"WR-06 {level} must resolve to one shelf region")
        region = next(iter(regions.values()))
        margin = float(settings["margin"])
        outer_size = np.asarray(region["size"], dtype=float)[:2] - margin
        inner_size = np.minimum(outer_size, np.asarray(settings["size"], dtype=float)[:2])
        normalized_pos = np.asarray(settings["pos"], dtype=float)[:2]
        local_center = np.asarray(region["offset"], dtype=float)[:2] + 0.5 * (
            outer_size - inner_size
        ) * normalized_pos
        rot = float(self.item_shelf.rot)
        rotation = np.array(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]], dtype=float
        )
        center = np.asarray(self.item_shelf.pos, dtype=float)[:2] + rotation @ local_center
        return center, inner_size


def _validate_home_region_placements(
    value: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    raw = DEFAULT_HOME_REGION_PLACEMENTS if value is None else value
    if set(raw) != set(HOME_LEVELS):
        raise ValueError("WR-06 home-region placements must define lower and upper")
    result: dict[str, dict[str, Any]] = {}
    for level in HOME_LEVELS:
        settings = dict(raw[level])
        result[level] = {
            "shelf_level": int(settings["shelf_level"]),
            "size": tuple(float(v) for v in settings["size"]),
            "pos": tuple(float(v) for v in settings["pos"]),
            "margin": float(settings.get("margin", 0.04)),
        }
    if result["shelf_lower"]["shelf_level"] >= result["shelf_upper"]["shelf_level"]:
        raise ValueError("WR-06 shelf_lower must be below shelf_upper")
    return result


def _other_level(level: str) -> str:
    return "shelf_upper" if validate_level(level) == "shelf_lower" else "shelf_lower"


__all__ = ["WR06ReturnSingleBottleToOriginalLevel"]
