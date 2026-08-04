"""RoboCasa environment and physical observations for WX-01."""

from __future__ import annotations

from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.wx01.domain.geometry import bottom_center_xy_distance
from bench.tasks.wx01.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    record_wipe_completion,
)
from bench.tasks.wx01.prompts import resolve_prompt_spec


CLEANING_TOOL_XML = f"{assets_root}/objects/objaverse/sponge/sponge_0/model.xml"
BLOCKING_OBJECT_XML = f"{assets_root}/objects/objaverse/cup/cup_2/model.xml"


class WX01ReturnObjectToOriginalPlace(Kitchen):
    """Six-stage task: wipe a remembered spot and return its displaced cup."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        *,
        origin_threshold: float = 0.02,
        wipe_radius: float = 0.20,
        contact_separation: float = 0.02,
        minimum_contact_count: int = 2,
        minimum_wipe_sweep: float = 0.07,
        release_distance: float = 0.15,
        blocking_object_scale: float = 0.85,
        blocking_object_placement_size: tuple[float, float] = (0.16, 0.20),
        blocking_object_placement_pos: tuple[float, float] = (1.0, 1.0),
        blocking_object_rotation: tuple[float, float] = (0.0, 0.0),
        **kwargs: Any,
    ) -> None:
        self.origin_threshold = _positive(origin_threshold, "origin_threshold")
        self.wipe_radius = _positive(wipe_radius, "wipe_radius")
        self.contact_separation = _positive(contact_separation, "contact_separation")
        self.minimum_contact_count = max(1, int(minimum_contact_count))
        self.minimum_wipe_sweep = _positive(minimum_wipe_sweep, "minimum_wipe_sweep")
        self.release_distance = _positive(release_distance, "release_distance")
        self.blocking_object_scale = float(blocking_object_scale)
        self.blocking_object_placement_size = _pair(
            blocking_object_placement_size,
            "blocking_object_placement_size",
        )
        self.blocking_object_placement_pos = _pair(
            blocking_object_placement_pos,
            "blocking_object_placement_pos",
        )
        self.blocking_object_rotation = _pair(
            blocking_object_rotation,
            "blocking_object_rotation",
        )
        super().__init__(**kwargs)

    def _setup_kitchen_references(self) -> None:
        """Register the two fixture anchors required by this task."""

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
        self.init_robot_base_ref = self.dining_table

    def _setup_scene(self) -> None:
        """Reset physical wipe tracking and the episode semantic latch."""

        super()._setup_scene()
        self._semantic_state = SemanticState()
        self.table_contact_positions: list[np.ndarray] = []
        self.table_contact_timer = 0
        self.wipe_sweep_range = 0.0
        self.original_area_wiped = False
        self.original_tool_pose = self._placement_pose("cleaning_tool")
        self.original_blocking_object_pose = self._placement_pose("blocking_object")

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        """Create one fixed sponge and one fixed cup without host-specific paths."""

        return [
            dict(
                name="cleaning_tool",
                obj_groups=CLEANING_TOOL_XML,
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.12, 0.12),
                    pos=(-0.0874406352692412, 0.7814444247774983),
                    rotation=(0.0, 0.0),
                ),
            ),
            dict(
                name="blocking_object",
                obj_groups=BLOCKING_OBJECT_XML,
                graspable=True,
                object_scale=self.blocking_object_scale,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=self.blocking_object_placement_size,
                    margin=0.01,
                    pos=self.blocking_object_placement_pos,
                    rotation=self.blocking_object_rotation,
                ),
            ),
        ]

    def set_original_object_poses(
        self,
        *,
        cleaning_tool: Any,
        blocking_object: Any,
    ) -> None:
        """Install the original poses sampled before the short-start displacement."""

        tool = _pose3(cleaning_tool, "cleaning_tool original pose")
        blocking = _pose3(blocking_object, "blocking_object original pose")
        self.original_tool_pose = tool.tolist()
        self.original_blocking_object_pose = blocking.tolist()
        refs = self._ep_meta.setdefault("refs", {})
        refs["original_tool_pose"] = tool.tolist()
        refs["original_blocking_object_pose"] = blocking.tolist()

    def record_wipe_completed(self) -> bool:
        """Latch the historical wipe event after an accepted physical sweep."""

        self.original_area_wiped = True
        return record_wipe_completion(self._semantic_state)

    def update_state(self) -> None:
        """Refresh physical wipe progress and synchronize its semantic latch."""

        super().update_state()
        self._update_wipe_progress()
        if self.original_area_wiped:
            record_wipe_completion(self._semantic_state)

    def get_ep_meta(self) -> dict[str, Any]:
        """Expose the short-start instruction and remembered target bindings."""

        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        refs = metadata.setdefault("refs", {})
        refs["cleaning_tool"] = "cleaning_tool"
        refs["blocking_object"] = "blocking_object"
        if self.original_tool_pose is not None:
            refs["original_tool_pose"] = list(self.original_tool_pose)
        if self.original_blocking_object_pose is not None:
            refs["original_blocking_object_pose"] = list(
                self.original_blocking_object_pose
            )
        return metadata

    def _check_success(self) -> bool:
        """Delegate the final decision to the domain success predicate."""

        self.update_state()
        return evaluate_success(self._semantic_state, self._physical_state()).success

    def task_diagnostics(self) -> dict[str, object]:
        """Return complete evidence for every authoritative success condition."""

        self.update_state()
        physical = self._physical_state()
        report = evaluate_success(self._semantic_state, physical)
        return {
            "success": report.success,
            "success_checks": report.checks(),
            "original_area_wiped": bool(self.original_area_wiped),
            "table_contact_timer": int(self.table_contact_timer),
            "wipe_sweep_range": float(self.wipe_sweep_range),
            "original_tool_pose": self.original_tool_pose,
            "original_blocking_object_pose": self.original_blocking_object_pose,
            "blocking_object_origin_distance": self.blocking_object_origin_distance(),
            "blocking_object_on_dining_table": physical.blocking_object_on_dining_table,
            "objects_released": bool(
                physical.cleaning_tool_released and physical.blocking_object_released
            ),
        }

    def get_object_world_position(self, object_name: str) -> np.ndarray:
        """Read a task object body position in world coordinates."""

        return np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[object_name]],
            dtype=float,
        )

    def blocking_object_origin_distance(self) -> float | None:
        """Return strict bottom-center XY distance to the remembered pose."""

        if self.original_blocking_object_pose is None:
            return None
        return bottom_center_xy_distance(
            self._object_bottom_center("blocking_object"),
            self.original_blocking_object_pose,
        )

    def _physical_state(self) -> PhysicalState:
        distance = self.blocking_object_origin_distance()
        return PhysicalState(
            blocking_object_at_origin=bool(
                distance is not None and distance < self.origin_threshold
            ),
            blocking_object_on_dining_table=bool(
                OU.check_obj_fixture_contact(
                    self,
                    "blocking_object",
                    self.dining_table,
                )
            ),
            cleaning_tool_released=bool(
                OU.gripper_obj_far(self, "cleaning_tool", th=self.release_distance)
            ),
            blocking_object_released=bool(
                OU.gripper_obj_far(self, "blocking_object", th=self.release_distance)
            ),
        )

    def _update_wipe_progress(self) -> None:
        sponge_grasped = OU.check_obj_grasped(
            self,
            "cleaning_tool",
            threshold=0.6,
        )
        sponge_contact_table = OU.check_obj_fixture_contact(
            self,
            "cleaning_tool",
            self.dining_table,
        )
        sponge_xy = self.get_object_world_position("cleaning_tool")[:2]
        origin_xy = (
            None
            if self.original_blocking_object_pose is None
            else np.asarray(self.original_blocking_object_pose, dtype=float)[:2]
        )
        near_original_area = bool(
            origin_xy is not None
            and np.linalg.norm(sponge_xy - origin_xy) <= self.wipe_radius
        )
        if sponge_contact_table and sponge_grasped and near_original_area:
            new_contact = all(
                np.linalg.norm(sponge_xy - recorded_xy) >= self.contact_separation
                for recorded_xy in self.table_contact_positions
            )
            if new_contact:
                self.table_contact_positions.append(
                    np.array(sponge_xy, dtype=float, copy=True)
                )
                self.table_contact_timer += 1
        if self.table_contact_positions:
            positions = np.asarray(self.table_contact_positions, dtype=float)
            self.wipe_sweep_range = float(
                np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
            )
        if self.original_blocking_object_pose is None:
            return
        assert origin_xy is not None
        covered = any(
            np.linalg.norm(contact[:2] - origin_xy) <= self.wipe_radius
            for contact in self.table_contact_positions
        )
        self.original_area_wiped = bool(
            self.original_area_wiped
            or (
                covered
                and self.table_contact_timer >= self.minimum_contact_count
                and self.wipe_sweep_range >= self.minimum_wipe_sweep
            )
        )

    def _placement_pose(self, object_name: str) -> list[float] | None:
        placement = getattr(self, "object_placements", {}).get(object_name)
        if placement is not None:
            return np.asarray(placement[0], dtype=float).tolist()
        if object_name in getattr(self, "obj_body_id", {}):
            return self.get_object_world_position(object_name).tolist()
        return None

    def _object_bottom_center(self, object_name: str) -> np.ndarray:
        body_id = self.obj_body_id[object_name]
        body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
        body_rotation = np.asarray(
            self.sim.data.body_xmat[body_id],
            dtype=float,
        ).reshape(3, 3)
        bottom_offset = np.asarray(
            self.objects[object_name].bottom_offset,
            dtype=float,
        )
        return body_pos + body_rotation @ bottom_offset


def _positive(value: Any, label: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"WX-01 {label} must be positive")
    return resolved


def _pair(value: Any, label: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"WX-01 {label} must contain two finite values")
    return float(array[0]), float(array[1])


def _pose3(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        raise ValueError(f"WX-01 {label} must contain a finite XYZ pose")
    return array[:3]


__all__ = [
    "BLOCKING_OBJECT_XML",
    "CLEANING_TOOL_XML",
    "WX01ReturnObjectToOriginalPlace",
]
