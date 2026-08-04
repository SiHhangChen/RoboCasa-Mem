"""Independent RoboCasa scene and physical-state boundary for WR-04."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.utils import object_utils as OU

from bench.tasks.wr04.asset_poses import (
    SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M,
    SHELF_UPRIGHT_ROTATION_AXIS,
    SHELF_UPRIGHT_ROTATION_RAD,
)
from bench.tasks.wr04.domain.contracts import SINK_SIDE_WORLD_OFFSETS_M
from bench.tasks.wr04.domain.contracts import HOME_LEVELS, ITEM_ID, SINK_STAGE_REGION
from bench.tasks.wr04.domain.episode import validate_level
from bench.tasks.wr04.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_memory_barrier,
    record_initial_level,
    record_return_attempt,
    record_return_completion,
    record_transport_posture_normalized,
)


STORAGE_FIXTURE_NAME = "opencabinet_1_main_group_1"
STORAGE_FIXTURE_CENTER_Z = 1.58
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ITEM_ASSETS = {
    ITEM_ID: str(
        _REPOSITORY_ROOT
        / "robocasa/robocasa/models/assets/objects/objaverse/sponge/sponge_3/model.xml"
    )
}
DEFAULT_HOME_REGION_PLACEMENTS = {
    "shelf_lower": {
        "shelf_level": 0,
        "size": (0.72, 0.13),
        "pos": (0.0, -1.0),
        "margin": 0.04,
    },
    "shelf_upper": {
        "shelf_level": 1,
        "size": (0.72, 0.13),
        "pos": (0.0, -1.0),
        "margin": 0.04,
    },
}


class WR04ReturnSingleObjectToOriginalLevel(Kitchen):
    """One sponge randomly starts on one of two levels and must return there."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        *args: Any,
        original_level: str = "shelf_lower",
        placement_pair_seed: int = 0,
        home_region_placements: dict[str, dict[str, Any]] | None = None,
        sink_stage_threshold: float = 0.20,
        return_region_xy_margin: float = 0.04,
        return_region_height_tolerance: float = 0.10,
        release_distance: float = 0.15,
        **kwargs: Any,
    ) -> None:
        self.demo_original_level = validate_level(original_level)
        self.placement_pair_seed = int(placement_pair_seed)
        self.home_region_placements = _validate_home_region_placements(
            home_region_placements
        )
        self.sink_stage_threshold = float(sink_stage_threshold)
        self.return_region_xy_margin = float(return_region_xy_margin)
        self.return_region_height_tolerance = float(return_region_height_tolerance)
        self.release_distance = float(release_distance)
        self._semantic_state = SemanticState()
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
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.sink_counter = self.register_fixture_ref(
            "sink_counter",
            dict(id=FixtureType.COUNTER, ref=self.sink, full_depth_region=True),
        )
        if self.item_shelf is None or self.sink is None or self.sink_counter is None:
            raise ValueError("WR-04 requires layout 32 with open shelf, sink, and counter")
        self.sink_stage_fixture = self.sink_counter
        self.init_robot_base_ref = self.item_shelf

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        placement = self.home_region_placements[self.demo_original_level]
        return [
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

    def _reset_internal(self) -> None:
        self._resample_initial_item()
        super()._reset_internal()

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
            raise RuntimeError(f"WR-04 sampled placement is missing {ITEM_ID!r}")
        self.object_placements = placements

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._semantic_state = SemanticState()
        placement = getattr(self, "object_placements", {}).get(ITEM_ID)
        if placement is None:
            raise RuntimeError("WR-04 reset is missing the sampled sponge placement")
        sampled_pos = np.asarray(placement[0], dtype=float).reshape(-1)[:3]
        actual_level = self.detect_actual_level(position=sampled_pos)
        if actual_level != self.demo_original_level:
            raise RuntimeError(
                f"WR-04 physical initial level {actual_level!r} differs from episode "
                f"level {self.demo_original_level!r}"
            )
        self._object_center_surface_offset_z = float(
            sampled_pos[2] - self.home_region_surface_z(actual_level)
        )
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
                "WR-04 failed to record the physically verified initial level: "
                f"actual={actual_level}, "
                f"in_actual={self._position_in_level(sampled_pos, actual_level)}, "
                f"in_other={self._position_in_level(sampled_pos, _other_level(actual_level))}, "
                f"sampled_pos={sampled_pos.tolist()}, "
                f"surface_offset={self._object_center_surface_offset_z}"
            )
        self._sync_diagnostics()

    def get_ep_meta(self) -> dict[str, Any]:
        from bench.tasks.wr04.prompts import resolve_prompt_spec

        meta = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        meta["lang"] = prompt.policy_prompt
        meta["initial_prompt"] = prompt.initial_prompt
        meta.setdefault("refs", {}).update(
            {
                "item_id": ITEM_ID,
                "item_label": "sponge",
                "home_levels": list(HOME_LEVELS),
                "sink_stage_region": SINK_STAGE_REGION,
            }
        )
        return meta

    def update_state(self) -> None:
        super().update_state()
        self._sync_diagnostics()

    def configure_episode(self, *, original_level: str) -> None:
        expected = validate_level(original_level)
        if expected != self.demo_original_level:
            raise ValueError("WR-04 environment and EpisodePlan original levels differ")
        actual = self.detect_actual_level()
        if actual != expected or self._semantic_state.initial_level != actual:
            raise ValueError("WR-04 initial physical level verification failed")

    # Semantic event adapters ------------------------------------------

    def record_transport_posture_normalized(self, *, physically_normalized: bool) -> bool:
        return record_transport_posture_normalized(
            self._semantic_state, physically_normalized=physically_normalized
        )

    def latch_original_level_hidden(self) -> bool:
        physical = self.physical_state()
        return latch_memory_barrier(
            self._semantic_state,
            item_at_fixed_sink=physical.item_at_fixed_sink,
            item_released=physical.item_released,
            item_outside_shelves=physical.item_outside_shelves,
            posture_normalized=self._semantic_state.transport_posture_normalized,
        )

    def record_return_attempt(self, level: str) -> bool:
        return record_return_attempt(self._semantic_state, level)

    def record_return_completion(self, item_id: str) -> bool:
        original = self._semantic_state.initial_level
        return record_return_completion(
            self._semantic_state,
            item_id,
            item_in_original_level=bool(
                original is not None and self.item_in_level(original) and self._item_supported_by_shelf()
            ),
            item_released=self.item_released(),
        )

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
            raise RuntimeError(f"WR-04 initial shelf level is ambiguous: {distances}")
        return ordered[0]

    def sink_stage_pose(self) -> np.ndarray:
        return np.asarray(self.sink.pos, dtype=float) + np.asarray(
            SINK_SIDE_WORLD_OFFSETS_M[SINK_STAGE_REGION], dtype=float
        )

    def use_station_pose(self, object_name: str) -> np.ndarray:
        if str(object_name) != ITEM_ID:
            raise KeyError(f"unknown WR-04 object {object_name!r}")
        return self.sink_stage_pose()

    def return_region_pose(self, level: str) -> np.ndarray:
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

    def home_region_center(self, level: str) -> np.ndarray:
        return self.return_region_pose(level)

    def home_region_surface_z(self, level: str) -> float:
        settings = self.home_region_placements[validate_level(level)]
        regions = self.item_shelf.get_reset_regions(
            shelf_level=int(settings["shelf_level"])
        )
        if len(regions) != 1:
            raise ValueError(f"WR-04 {level} must resolve to one shelf region")
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

    def item_at_fixed_sink(self) -> bool:
        supported = any(
            OU.check_obj_fixture_contact(self, ITEM_ID, fixture)
            for fixture in (self.sink, self.sink_counter)
        )
        distance = float(np.linalg.norm(self._obj_pos(ITEM_ID)[:2] - self.sink_stage_pose()[:2]))
        return bool(distance <= self.sink_stage_threshold and supported)

    def item_released(self) -> bool:
        return bool(OU.gripper_obj_far(self, ITEM_ID, th=self.release_distance))

    def _item_supported_by_shelf(self) -> bool:
        level = self._semantic_state.initial_level
        if level is None:
            return False
        shelf_index = int(
            self.home_region_placements[validate_level(level)]["shelf_level"]
        )
        # RoboCasa's OpenCabinet does not register its dynamically created
        # level shelves in ``fixture.contact_geoms``.  Check the actual level
        # geom explicitly; the generic fixture-contact helper is therefore
        # false even for a valid reset placement.
        shelf_geom = f"{self.item_shelf.name}_level{shelf_index}_shelf"
        return bool(self.check_contact(self.objects[ITEM_ID], shelf_geom))

    def physical_state(self) -> PhysicalState:
        return PhysicalState(
            item_at_fixed_sink=self.item_at_fixed_sink(),
            item_in_lower=self.item_in_level("shelf_lower"),
            item_in_upper=self.item_in_level("shelf_upper"),
            item_released=self.item_released(),
            item_supported_by_shelf=self._item_supported_by_shelf(),
        )

    def _check_success(self) -> bool:
        return self._success_report().success

    def _success_report(self):
        return evaluate_success(self._semantic_state, self.physical_state())

    def task_diagnostics(self) -> dict[str, Any]:
        report = self._success_report()
        physical = self.physical_state()
        return {
            "success": report.success,
            **report.checks(),
            "current_item_position": self._obj_pos(ITEM_ID).astype(float).tolist(),
            "current_item_at_fixed_sink": physical.item_at_fixed_sink,
            "current_item_in_lower": physical.item_in_lower,
            "current_item_in_upper": physical.item_in_upper,
            "transport_posture_normalized": self._semantic_state.transport_posture_normalized,
            "wrong_return_level_attempted": self._semantic_state.wrong_return_level_attempted,
            "protocol_violation": self._semantic_state.protocol_violation,
        }

    def _sync_diagnostics(self) -> None:
        state = self._semantic_state
        self.initial_level_physically_verified = state.initial_level_physically_verified
        self.item_staged_at_fixed_sink = state.item_staged_at_sink
        self.memory_barrier_latched = state.memory_barrier_latched
        self.wrong_return_level_attempted = state.wrong_return_level_attempted
        self.protocol_violation = state.protocol_violation

    def _home_region_xy_geometry(self, level: str) -> tuple[np.ndarray, np.ndarray]:
        settings = self.home_region_placements[validate_level(level)]
        regions = self.item_shelf.get_reset_regions(
            shelf_level=int(settings["shelf_level"])
        )
        if len(regions) != 1:
            raise ValueError(f"WR-04 {level} must resolve to one shelf region")
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
        raise ValueError("WR-04 home-region placements must define lower and upper")
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
        raise ValueError("WR-04 shelf_lower must be below shelf_upper")
    return result


def _other_level(level: str) -> str:
    return "shelf_upper" if validate_level(level) == "shelf_lower" else "shelf_lower"


__all__ = ["WR04ReturnSingleObjectToOriginalLevel"]
