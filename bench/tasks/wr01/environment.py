"""RoboCasa scene and physical-observation boundary for WR-01."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models import assets_root
from robocasa.models.fixtures.fixture import Fixture
from robocasa.utils.object_utils import (
    get_fixture_to_point_rel_offset,
    get_pos_after_rel_offset,
)

from bench.tasks.wr01.domain.contracts import CABINET_LEVELS, ITEM_IDS
from bench.tasks.wr01.domain.geometry import (
    STORAGE_INTERIOR_LOCAL_CENTER,
    level_world_xyz,
    point_in_storage_interior,
    validate_item_level_mapping,
)
from bench.tasks.wr01.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_storage_hidden,
    record_retrieval_door_open,
    record_retrieval_pick,
    record_storage,
)
from bench.tasks.wr01.prompts import resolve_prompt_spec


RED_CAN_XML = f"{assets_root}/objects/objaverse/canned_food/canned_food_15/model.xml"
BLUE_CAN_XML = f"{assets_root}/objects/objaverse/canned_food/canned_food_11/model.xml"
ITEM_COUNTER_NAME = "counter_1_main_group"
LOWER_CABINET_NAME = "wr01_lower_storage_cabinet"
UPPER_CABINET_NAME = "wr01_upper_storage_cabinet"
SLIDE_CABINET_ASSET = (
    Path(__file__).resolve().parent / "resources" / "assets" / "slidecabinet.xml"
)
INTERIOR_LOCAL_X = STORAGE_INTERIOR_LOCAL_CENTER[0]
INTERIOR_LOCAL_Y = STORAGE_INTERIOR_LOCAL_CENTER[1]
INTERIOR_FLOOR_LOCAL_Z = -0.15
TOP_STAGING_LOCAL_Y = -0.20
TOP_STAGING_LOCAL_Z = 0.22
ITEM_SAMPLING_REGIONS: dict[str, dict[str, Any]] = {
    "red_can": {
        "size": (0.18, 0.14),
        "pos": (-0.71, -0.10),
        "offset": (-0.11, -0.05),
        "rotation": 0.0,
    },
    "blue_can": {
        "size": (0.18, 0.14),
        "pos": (-0.71, -0.10),
        "offset": (0.11, -0.05),
        "rotation": 0.0,
    },
}


class WR01SlideStorageCabinet(Fixture):
    """Task-owned sliding cabinet used only by the WR-01 environment."""

    def __init__(self, *, name: str, rng: Any | None = None) -> None:
        super().__init__(
            xml=str(SLIDE_CABINET_ASSET),
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
            [INTERIOR_LOCAL_X, INTERIOR_LOCAL_Y, INTERIOR_FLOOR_LOCAL_Z],
            dtype=float,
        )
        return np.asarray(get_pos_after_rel_offset(self, local), dtype=float)[:3]

    def interior_pose(self) -> np.ndarray:
        pose = self.interior_floor_pose()
        pose[2] += 0.08
        return pose

    def top_staging_pose(self) -> np.ndarray:
        local = np.asarray(
            [0.0, TOP_STAGING_LOCAL_Y, TOP_STAGING_LOCAL_Z],
            dtype=float,
        )
        return np.asarray(get_pos_after_rel_offset(self, local), dtype=float)[:3]

    def contains_point(self, point: Any, *, margin: float = 0.035) -> bool:
        local = np.asarray(
            get_fixture_to_point_rel_offset(self, point), dtype=float
        ).reshape(-1)[:3]
        return point_in_storage_interior(local, margin=margin)


class WR01RetrieveRememberedCan(Kitchen):
    """Two visible cans, two stacked cabinets, and one hidden relation memory test."""

    def __init__(
        self,
        *args: Any,
        cabinet_world_xyz: tuple[float, float, float] = (3.3599999, -0.3500018, 1.1363333),
        cabinet_yaw_deg: float = 0.0,
        upper_cabinet_z_offset: float = 0.40,
        object_scale: float = 0.90,
        item_placement_seed: int = 0,
        final_closed_threshold: float = 0.08,
        cabinet_contains_margin: float = 0.04,
        **kwargs: Any,
    ) -> None:
        self._cabinet_world_xyz = tuple(float(value) for value in cabinet_world_xyz)
        self._cabinet_yaw = float(np.deg2rad(float(cabinet_yaw_deg)))
        self._upper_cabinet_z_offset = float(upper_cabinet_z_offset)
        self._object_scale = float(object_scale)
        self._item_placement_seed = int(item_placement_seed)
        self.final_closed_threshold = float(final_closed_threshold)
        self.cabinet_contains_margin = float(cabinet_contains_margin)
        self._semantic_state = SemanticState("red_can", "lower")
        self.item_by_level = {"lower": "red_can", "upper": "blue_can"}
        self.level_by_item = {
            item: level for level, item in self.item_by_level.items()
        }
        kwargs.setdefault("obj_registries", ("objaverse", "aigen", "lightwheel"))
        super().__init__(*args, **kwargs)

    # Scene construction -------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()
        self.lower_storage_cabinet = WR01SlideStorageCabinet(
            name=LOWER_CABINET_NAME,
            rng=self.rng,
        )
        self.upper_storage_cabinet = WR01SlideStorageCabinet(
            name=UPPER_CABINET_NAME,
            rng=self.rng,
        )
        for name, fixture in (
            (LOWER_CABINET_NAME, self.lower_storage_cabinet),
            (UPPER_CABINET_NAME, self.upper_storage_cabinet),
        ):
            self.fixtures[name] = fixture
            self.fixture_cfgs.append({"name": name, "model": fixture, "type": "fixture"})
        self.model.merge_objects([self.lower_storage_cabinet, self.upper_storage_cabinet])

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=ITEM_COUNTER_NAME),
        )
        self.init_robot_base_ref = self.counter
        positions = level_world_xyz(
            self._cabinet_world_xyz,
            upper_z_offset=self._upper_cabinet_z_offset,
        )
        self.lower_storage_cabinet.set_world_pose(
            world_xyz=positions["lower"], yaw=self._cabinet_yaw
        )
        self.upper_storage_cabinet.set_world_pose(
            world_xyz=positions["upper"], yaw=self._cabinet_yaw
        )
        self.storage_cabinets = {
            "lower": self.lower_storage_cabinet,
            "upper": self.upper_storage_cabinet,
        }
        self.fixture_refs["lower_storage_cabinet"] = (self.lower_storage_cabinet, False)
        self.fixture_refs["upper_storage_cabinet"] = (self.upper_storage_cabinet, False)

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        return [
            dict(
                name="red_can",
                obj_groups=RED_CAN_XML,
                object_scale=self._object_scale,
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.counter),
                    **ITEM_SAMPLING_REGIONS["red_can"],
                ),
            ),
            dict(
                name="blue_can",
                obj_groups=BLUE_CAN_XML,
                object_scale=self._object_scale,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.counter),
                    **ITEM_SAMPLING_REGIONS["blue_can"],
                ),
            ),
        ]

    def _reset_internal(self) -> None:
        """Install episode-seeded item samples before RoboCasa settles the reset."""

        self._resample_initial_items()
        super()._reset_internal()

    def _resample_initial_items(self) -> None:
        """Resample the declared regions without advancing the fixed-scene RNG."""

        initializer = getattr(self, "placement_initializer", None)
        if initializer is None:
            return
        children = getattr(initializer, "samplers", {})
        child_samplers = list(children.values()) if hasattr(children, "values") else list(children)
        samplers = [initializer, *child_samplers]
        previous_rngs = [getattr(sampler, "rng", None) for sampler in samplers]
        placement_rng = np.random.default_rng(self._item_placement_seed)
        try:
            for sampler in samplers:
                sampler.rng = placement_rng
            placements = initializer.sample(placed_objects=self.fxtr_placements)
        finally:
            for sampler, previous_rng in zip(samplers, previous_rngs):
                sampler.rng = previous_rng
        missing = set(ITEM_IDS) - set(placements)
        if missing:
            raise RuntimeError(
                "WR-01 sampled placement is missing items: " + ", ".join(sorted(missing))
            )
        self.object_placements = placements

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._semantic_state = SemanticState("red_can", "lower")
        self.item_by_level = {"lower": "red_can", "upper": "blue_can"}
        self.level_by_item = {
            item: level for level, item in self.item_by_level.items()
        }

    # Episode and semantic events ---------------------------------------

    @property
    def target_item_id(self) -> str:
        return self._semantic_state.target_item_id

    @property
    def target_level(self) -> str:
        return self._semantic_state.target_level

    @property
    def distractor_item_id(self) -> str:
        return "blue_can" if self.target_item_id == "red_can" else "red_can"

    def configure_episode(
        self,
        *,
        item_by_level: Mapping[str, str],
        target_item_id: str,
    ) -> None:
        """Install frozen private episode facts after the environment is reset."""

        assignment = validate_item_level_mapping(item_by_level)
        target = str(target_item_id)
        if target not in ITEM_IDS:
            raise ValueError(f"invalid WR-01 target item: {target!r}")
        self.item_by_level = assignment
        self.level_by_item = {item: level for level, item in assignment.items()}
        self._semantic_state = SemanticState(target, self.level_by_item[target])
        self._ep_meta.setdefault("refs", {})["requested_item_id"] = target

    def record_storage(self, item_id: str, level: str) -> bool:
        return record_storage(self._semantic_state, item_id, level)

    def latch_storage_hidden(self) -> bool:
        return latch_storage_hidden(
            self._semantic_state,
            all_stored=all(
                self.is_item_in_level(item, self.level_by_item[item])
                for item in ITEM_IDS
            ),
            cabinets_closed=self.are_cabinets_closed(),
        )

    def record_retrieval_door_open(self, level: str) -> bool:
        return record_retrieval_door_open(self._semantic_state, level)

    def record_retrieval_pick(self, item_id: str) -> bool:
        return record_retrieval_pick(self._semantic_state, item_id)

    # Physical observation boundary -------------------------------------

    def cabinet_for_level(self, level: str) -> WR01SlideStorageCabinet:
        if level not in CABINET_LEVELS:
            raise KeyError(f"unknown WR-01 cabinet level: {level!r}")
        return self.storage_cabinets[level]

    def set_level_door_open_fraction(self, level: str, fraction: float) -> None:
        self.cabinet_for_level(level).set_door_open_fraction(self, fraction)
        self.sim.forward()

    def door_open_fraction(self, level: str) -> float:
        return self.cabinet_for_level(level).door_open_fraction(self)

    def is_door_open(self, level: str, *, threshold: float = 0.70) -> bool:
        return self.door_open_fraction(level) >= float(threshold)

    def is_door_closed(self, level: str, *, threshold: float | None = None) -> bool:
        return self.door_open_fraction(level) <= float(
            self.final_closed_threshold if threshold is None else threshold
        )

    def are_cabinets_closed(self) -> bool:
        return all(self.is_door_closed(level) for level in CABINET_LEVELS)

    def get_object_world_position(self, object_name: str) -> np.ndarray:
        return np.asarray(self.sim.data.body_xpos[self.obj_body_id[str(object_name)]], dtype=float)

    def get_object_world_quaternion_xyzw(self, object_name: str) -> np.ndarray:
        quat_wxyz = np.asarray(
            self.sim.data.body_xquat[self.obj_body_id[str(object_name)]],
            dtype=float,
        )
        return quat_wxyz[[1, 2, 3, 0]]

    @property
    def item_placement_seed(self) -> int:
        return self._item_placement_seed

    def item_sampling_regions(self) -> dict[str, dict[str, Any]]:
        """Return the public, task-owned counter-region declarations."""

        return {
            item_id: {
                "reference_fixture": str(self.counter.name),
                "coordinate_frame": "counter_reset_region",
                "size": [float(value) for value in values["size"]],
                "pos": [values["pos"][0], float(values["pos"][1])],
                "offset": [float(value) for value in values["offset"]],
                "rotation": float(values["rotation"]),
            }
            for item_id, values in ITEM_SAMPLING_REGIONS.items()
        }

    def is_item_in_level(self, item_id: str, level: str) -> bool:
        if item_id not in ITEM_IDS:
            return False
        return bool(
            self.cabinet_for_level(level).contains_point(
                self.get_object_world_position(item_id),
                margin=self.cabinet_contains_margin,
            )
        )

    def physical_state(self) -> PhysicalState:
        return PhysicalState(
            distractor_still_stored=self.is_item_in_level(
                self.distractor_item_id,
                self.level_by_item[self.distractor_item_id],
            ),
        )

    def set_object_pose(
        self,
        object_name: str,
        position: Any,
        quat_xyzw: Any | None = None,
    ) -> None:
        """Task-owned pose helper used by reset and validated transport following."""

        obj = self.objects[str(object_name)]
        qpos = np.asarray(self.sim.data.get_joint_qpos(obj.joints[0]), dtype=float).reshape(-1)
        target = np.asarray(position, dtype=float).reshape(-1)
        if qpos.size < 7 or target.size < 3 or not np.all(np.isfinite(target[:3])):
            raise ValueError(f"invalid WR-01 reset pose for {object_name!r}")
        updated = qpos.copy()
        updated[:3] = target[:3]
        if quat_xyzw is not None:
            quat = np.asarray(quat_xyzw, dtype=float).reshape(-1)
            if quat.size != 4 or not np.all(np.isfinite(quat)):
                raise ValueError(f"invalid WR-01 reset quaternion for {object_name!r}")
            norm = float(np.linalg.norm(quat))
            if norm <= 1e-9:
                raise ValueError(f"zero WR-01 reset quaternion for {object_name!r}")
            quat = quat / norm
            updated[3:7] = quat[[3, 0, 1, 2]]
        self.sim.data.set_joint_qpos(obj.joints[0], updated)
        self.sim.forward()

    # Metadata and final predicate --------------------------------------

    def get_ep_meta(self) -> dict[str, Any]:
        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(
            target_item_id=self.target_item_id,
            variant="memory",
        )
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        metadata.setdefault("refs", {}).update(
            {
                "item_ids": list(ITEM_IDS),
                "cabinet_levels": list(CABINET_LEVELS),
                "requested_item_id": self.target_item_id,
            }
        )
        return metadata

    def _check_success(self) -> bool:
        return evaluate_success(self._semantic_state, self.physical_state()).success

    def task_diagnostics(self) -> dict[str, Any]:
        report = evaluate_success(self._semantic_state, self.physical_state())
        return {
            "success": report.success,
            "success_checks": report.checks(),
            "storage": {
                "stored_levels_by_item": dict(
                    self._semantic_state.stored_levels_by_item
                ),
                "storage_hidden": bool(self._semantic_state.storage_hidden),
                "doors": {level: self.door_open_fraction(level) for level in CABINET_LEVELS},
                "closed": self.are_cabinets_closed(),
            },
            "retrieval": {
                "target_item_id": self.target_item_id,
                "opened_level": self._semantic_state.retrieval_level_opened,
                "wrong_level_opened": self._semantic_state.wrong_retrieval_level_opened,
                "retrieved_item_id": self._semantic_state.retrieved_item_id,
            },
            "physical": {
                "distractor_still_stored": self.physical_state().distractor_still_stored,
            },
        }


__all__ = ["WR01RetrieveRememberedCan"]
