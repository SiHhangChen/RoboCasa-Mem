"""RoboCasa environment and physical observations for WX-02."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.wx02.domain.contracts import CUP_NAMES, PRIMARY_CUP, SECONDARY_CUP
from bench.tasks.wx02.domain.geometry import bottom_center_xy_distance
from bench.tasks.wx02.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    record_object_moved,
    record_object_returned,
    record_stain_cleaned,
    record_tool_acquired,
    record_tool_left_return_region,
    record_tool_returned,
)
from bench.tasks.wx02.evaluation import WX02EvaluationSnapshot, WX02Evaluator
from bench.tasks.wx02.prompts import resolve_prompt_spec


CLEANING_TOOL_XML = f"{assets_root}/objects/objaverse/sponge/sponge_0/model.xml"
CUP_XML = f"{assets_root}/objects/objaverse/cup/cup_5/model.xml"
STAIN_GEOM_PREFIX_BY_OBJECT = {
    PRIMARY_CUP: "wx02_primary_cup_stain_spot_",
    SECONDARY_CUP: "wx02_secondary_cup_stain_spot_",
}
CLEANING_TOOL_RETURN_REGION_GEOM = "bench_wx02_cleaning_tool_return_region"


class WX02CleanUnderCupAndReturnIt(Kitchen):
    """Clean two hidden stains in order and restore both cups and the sponge."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        *,
        origin_threshold: float = 0.02,
        tool_return_region_tolerance: float = 0.02,
        moved_threshold: float = 0.10,
        wipe_radius: float = 0.14,
        contact_separation: float = 0.02,
        minimum_contact_count: int = 2,
        minimum_wipe_sweep: float = 0.08,
        release_distance: float = 0.15,
        stain_surface_z_offset: float = 0.0015,
        stain_color_rgba: tuple[float, float, float, float] = (
            0.20,
            0.055,
            0.01,
            0.92,
        ),
        stain_spots: tuple[tuple[float, float, float, float, float], ...] = (),
        stain_center_xy_by_object: Mapping[str, Any] | None = None,
        wipe_direction_xy_by_object: Mapping[str, Any] | None = None,
        cup_scale: float = 0.9,
        cup_placement_size: tuple[float, float] = (0.16, 0.20),
        primary_cup_placement_pos: tuple[float, float] = (1.0, 1.0),
        secondary_cup_placement_pos: tuple[float, float] = (1.0, 0.55),
        cup_rotation: tuple[float, float] = (0.0, 0.0),
        cleaning_tool_return_region_center_xy: tuple[float, float] = (
            2.50,
            -4.62,
        ),
        cleaning_tool_return_region_size_xy: tuple[float, float] = (0.22, 0.20),
        cleaning_tool_return_region_rgba: tuple[float, float, float, float] = (
            0.96,
            0.72,
            0.16,
            0.72,
        ),
        **kwargs: Any,
    ) -> None:
        self.origin_threshold = _positive(origin_threshold, "origin_threshold")
        self.tool_return_region_tolerance = _positive(
            tool_return_region_tolerance, "tool_return_region_tolerance"
        )
        self.moved_threshold = _positive(moved_threshold, "moved_threshold")
        self.wipe_radius = _positive(wipe_radius, "wipe_radius")
        self.contact_separation = _positive(
            contact_separation, "contact_separation"
        )
        self.minimum_contact_count = max(1, int(minimum_contact_count))
        self.minimum_wipe_sweep = _positive(
            minimum_wipe_sweep, "minimum_wipe_sweep"
        )
        self.release_distance = _positive(release_distance, "release_distance")
        self.stain_surface_z_offset = _positive(
            stain_surface_z_offset, "stain_surface_z_offset"
        )
        self.stain_color_rgba = _rgba(stain_color_rgba, "stain_color_rgba")
        self.stain_spots = _stain_spots(stain_spots)
        centers = dict(stain_center_xy_by_object or {})
        directions = dict(wipe_direction_xy_by_object or {})
        if set(centers) != set(CUP_NAMES) or set(directions) != set(CUP_NAMES):
            raise ValueError(
                "WX-02 stain centers and wipe directions must define both cups"
            )
        self.stain_center_xy_by_object = {
            name: np.asarray(_pair(centers[name], f"{name}.stain_center_xy"))
            for name in CUP_NAMES
        }
        self.wipe_direction_xy_by_object = {}
        for name in CUP_NAMES:
            direction = np.asarray(
                _pair(directions[name], f"{name}.wipe_direction_xy"), dtype=float
            )
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-9:
                raise ValueError(f"WX-02 {name} wipe direction must be non-zero")
            self.wipe_direction_xy_by_object[name] = direction / norm
        self.cup_scale = float(cup_scale)
        self.cup_placement_size = _pair(cup_placement_size, "cup_placement_size")
        self.cup_placement_pos = {
            PRIMARY_CUP: _pair(
                primary_cup_placement_pos, "primary_cup_placement_pos"
            ),
            SECONDARY_CUP: _pair(
                secondary_cup_placement_pos, "secondary_cup_placement_pos"
            ),
        }
        self.cup_rotation = _pair(cup_rotation, "cup_rotation")
        self.cleaning_tool_return_region_center_xy = np.asarray(
            _pair(
                cleaning_tool_return_region_center_xy,
                "cleaning_tool_return_region_center_xy",
            ),
            dtype=float,
        )
        self.cleaning_tool_return_region_size_xy = np.asarray(
            _positive_pair(
                cleaning_tool_return_region_size_xy,
                "cleaning_tool_return_region_size_xy",
            ),
            dtype=float,
        )
        self.cleaning_tool_return_region_rgba = _rgba(
            cleaning_tool_return_region_rgba,
            "cleaning_tool_return_region_rgba",
        )
        self._stain_initial_alphas_by_object = {
            name: tuple(
                self.stain_color_rgba[3] * spot[4] for spot in self.stain_spots
            )
            for name in CUP_NAMES
        }
        self._wx02_evaluator: WX02Evaluator | None = None
        super().__init__(**kwargs)

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
        self._validate_cleaning_tool_return_region()
        self.init_robot_base_ref = self.dining_table

    def edit_model_xml(self, xml_str: str) -> str:
        """Add two independent irregular, non-colliding stains."""

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
        if CLEANING_TOOL_RETURN_REGION_GEOM not in existing:
            region_z = self._dining_table_surface_z() + 0.002
            half = self.cleaning_tool_return_region_size_xy * 0.5
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": CLEANING_TOOL_RETURN_REGION_GEOM,
                    "type": "box",
                    "pos": (
                        f"{self.cleaning_tool_return_region_center_xy[0]:.8f} "
                        f"{self.cleaning_tool_return_region_center_xy[1]:.8f} "
                        f"{region_z:.8f}"
                    ),
                    "size": f"{half[0]:.6f} {half[1]:.6f} 0.002",
                    "rgba": " ".join(
                        f"{value:.6f}"
                        for value in self.cleaning_tool_return_region_rgba
                    ),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
        surface_z = self._dining_table_surface_z() + self.stain_surface_z_offset
        for object_name in CUP_NAMES:
            direction = self.wipe_direction_xy_by_object[object_name]
            perpendicular = np.asarray([-direction[1], direction[0]], dtype=float)
            yaw = math.atan2(float(direction[1]), float(direction[0]))
            quat = (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))
            center = self.stain_center_xy_by_object[object_name]
            prefix = STAIN_GEOM_PREFIX_BY_OBJECT[object_name]
            for index, spot in enumerate(self.stain_spots):
                along, across, radius_x, radius_y, relative_alpha = spot
                name = f"{prefix}{index:02d}"
                if name in existing:
                    continue
                xy = center + along * direction + across * perpendicular
                rgba = (
                    *self.stain_color_rgba[:3],
                    self.stain_color_rgba[3] * relative_alpha,
                )
                ET.SubElement(
                    worldbody,
                    "geom",
                    {
                        "name": name,
                        "type": "ellipsoid",
                        "pos": f"{xy[0]:.8f} {xy[1]:.8f} {surface_z:.8f}",
                        "size": f"{radius_x:.6f} {radius_y:.6f} 0.0006",
                        "quat": " ".join(f"{value:.8f}" for value in quat),
                        "rgba": " ".join(f"{value:.6f}" for value in rgba),
                        "contype": "0",
                        "conaffinity": "0",
                        "group": "1",
                    },
                )
        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._semantic_state = SemanticState()
        self._wx02_evaluator = None
        self.table_contact_positions_by_object = {
            name: [] for name in CUP_NAMES
        }
        self.table_contact_timer_by_object = {name: 0 for name in CUP_NAMES}
        self.wipe_sweep_range_by_object = {name: 0.0 for name in CUP_NAMES}
        self.stain_progress_by_object = {name: 0.0 for name in CUP_NAMES}
        self.original_area_wiped_by_object = {name: False for name in CUP_NAMES}
        self.original_tool_pose = self._placement_pose("cleaning_tool")
        self.original_object_poses = {
            name: self._placement_pose(name) for name in CUP_NAMES
        }
        for name, pose in self.original_object_poses.items():
            setattr(self, f"original_{name}_pose", pose)
        self._tool_stably_in_return_region_previous: bool | None = None

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        configs = [
            dict(
                name="cleaning_tool",
                obj_groups=CLEANING_TOOL_XML,
                graspable=True,
                init_robot_here=False,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.12, 0.12),
                    pos=(-0.0874406352692412, 0.7814444247774983),
                    rotation=(0.0, 0.0),
                ),
            )
        ]
        for object_name in CUP_NAMES:
            configs.append(
                dict(
                    name=object_name,
                    obj_groups=CUP_XML,
                    graspable=True,
                    object_scale=self.cup_scale,
                    placement=dict(
                        fixture=self.dining_table,
                        sample_region_kwargs=dict(ref=self.stool),
                        size=self.cup_placement_size,
                        margin=0.01,
                        pos=self.cup_placement_pos[object_name],
                        rotation=self.cup_rotation,
                    ),
                )
            )
        return configs

    def set_original_object_poses(
        self,
        *,
        cleaning_tool: Any,
        object_poses: Mapping[str, Any],
    ) -> None:
        if set(object_poses) != set(CUP_NAMES):
            raise ValueError("WX-02 original object poses must define both cups")
        tool = _pose3(cleaning_tool, "cleaning_tool original pose")
        objects = {
            name: _pose3(object_poses[name], f"{name} original pose")
            for name in CUP_NAMES
        }
        self.original_tool_pose = tool.tolist()
        self.original_object_poses = {
            name: pose.tolist() for name, pose in objects.items()
        }
        refs = self._ep_meta.setdefault("refs", {})
        refs["original_tool_pose"] = tool.tolist()
        refs["original_object_poses"] = dict(self.original_object_poses)
        refs["stain_center_poses"] = dict(self.original_object_poses)
        for name, pose in objects.items():
            value = pose.tolist()
            setattr(self, f"original_{name}_pose", value)
            refs[f"original_{name}_pose"] = value

    def record_cleaning_tool_acquired(self) -> bool:
        return record_tool_acquired(self._semantic_state)

    def record_object_moved(self, object_name: str) -> bool:
        return record_object_moved(self._semantic_state, object_name)

    def record_wipe_completed(self, object_name: str) -> bool:
        if object_name not in CUP_NAMES:
            raise KeyError(f"unknown WX-02 stain: {object_name!r}")
        changed = record_stain_cleaned(self._semantic_state, object_name)
        if not changed and not self._stain_cleaned(object_name):
            return False
        self.original_area_wiped_by_object[object_name] = True
        self.stain_progress_by_object[object_name] = 1.0
        self._sync_stain_visual(object_name)
        return changed

    def update_state(self) -> None:
        super().update_state()
        if OU.check_obj_grasped(self, "cleaning_tool", threshold=0.6):
            record_tool_acquired(self._semantic_state)
        for object_name in CUP_NAMES:
            distance = self.object_origin_distance(object_name)
            if distance is not None and distance >= self.moved_threshold:
                record_object_moved(self._semantic_state, object_name)
            self._update_wipe_progress(object_name)
            if self.original_area_wiped_by_object[object_name]:
                record_stain_cleaned(self._semantic_state, object_name)
        physical = self._physical_state()
        for object_name in CUP_NAMES:
            if (
                self._cup_physical(physical, object_name, "at_origin")
                and self._cup_physical(physical, object_name, "on_dining_table")
                and self._cup_physical(physical, object_name, "released")
            ):
                record_object_returned(self._semantic_state, object_name)

        tool_stable = bool(
            physical.cleaning_tool_in_return_region
            and physical.cleaning_tool_on_dining_table
            and physical.cleaning_tool_released
        )
        if self._semantic_state.cleaning_tool_acquired and not tool_stable:
            record_tool_left_return_region(self._semantic_state)
        previous = self._tool_stably_in_return_region_previous
        if previous is False and tool_stable:
            record_tool_returned(self._semantic_state)
        self._tool_stably_in_return_region_previous = tool_stable

    def get_ep_meta(self) -> dict[str, Any]:
        metadata = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        refs = metadata.setdefault("refs", {})
        refs["cleaning_tool"] = "cleaning_tool"
        refs["cups"] = list(CUP_NAMES)
        refs["stain_geom_prefix_by_object"] = dict(STAIN_GEOM_PREFIX_BY_OBJECT)
        refs["cleaning_tool_return_region_geom"] = (
            CLEANING_TOOL_RETURN_REGION_GEOM
        )
        refs["cleaning_tool_return_region_center_xy"] = (
            self.cleaning_tool_return_region_center_xy.tolist()
        )
        refs["cleaning_tool_return_region_size_xy"] = (
            self.cleaning_tool_return_region_size_xy.tolist()
        )
        if self.original_tool_pose is not None:
            refs["original_tool_pose"] = list(self.original_tool_pose)
        refs["original_object_poses"] = dict(self.original_object_poses)
        refs["stain_center_poses"] = dict(self.original_object_poses)
        return metadata

    def _check_success(self) -> bool:
        self.update_state()
        return evaluate_success(self._semantic_state, self._physical_state()).success

    def task_diagnostics(self) -> dict[str, object]:
        snapshot = self.evaluation_snapshot()
        if self._wx02_evaluator is not None:
            return self._wx02_evaluator.observe(snapshot).to_diagnostics()
        return {
            **dict(snapshot.details),
            "success": bool(snapshot.domain_success),
            "terminal": bool(snapshot.domain_success),
            "failure_code": None,
        }

    def evaluation_snapshot(self) -> WX02EvaluationSnapshot:
        self.update_state()
        physical = self._physical_state()
        state = self._semantic_state
        report = evaluate_success(state, physical)
        details = {
            "success_checks": report.checks(),
            "release_checks": {
                "cleaning_tool_released": physical.cleaning_tool_released,
                "primary_cup_released": physical.primary_cup_released,
                "secondary_cup_released": physical.secondary_cup_released,
            },
            "cleaning_tool_left_return_region": (
                state.cleaning_tool_left_return_region
            ),
            "stain_progress_by_object": dict(self.stain_progress_by_object),
            "table_contact_timer_by_object": dict(
                self.table_contact_timer_by_object
            ),
            "wipe_sweep_range_by_object": dict(
                self.wipe_sweep_range_by_object
            ),
            "original_tool_pose": self.original_tool_pose,
            "original_object_poses": dict(self.original_object_poses),
            "object_origin_distance": {
                name: self.object_origin_distance(name) for name in CUP_NAMES
            },
            "cleaning_tool_return_region_distance": (
                self.cleaning_tool_return_region_distance()
            ),
        }
        return WX02EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            cleaning_tool_acquired=state.cleaning_tool_acquired,
            primary_cup_moved=state.primary_cup_moved,
            primary_stain_cleaned=state.primary_stain_cleaned,
            primary_cup_returned=state.primary_cup_returned,
            secondary_cup_moved=state.secondary_cup_moved,
            secondary_stain_cleaned=state.secondary_stain_cleaned,
            secondary_cup_returned=state.secondary_cup_returned,
            cleaning_tool_returned=state.cleaning_tool_returned,
            cleaning_tool_released=physical.cleaning_tool_released,
            primary_cup_at_origin=physical.primary_cup_at_origin,
            primary_cup_on_dining_table=physical.primary_cup_on_dining_table,
            secondary_cup_at_origin=physical.secondary_cup_at_origin,
            secondary_cup_on_dining_table=physical.secondary_cup_on_dining_table,
            cleaning_tool_in_return_region=(
                physical.cleaning_tool_in_return_region
            ),
            cleaning_tool_on_dining_table=physical.cleaning_tool_on_dining_table,
            primary_cup_released=physical.primary_cup_released,
            secondary_cup_released=physical.secondary_cup_released,
            domain_success=report.success,
            details=details,
        )

    def get_object_world_position(self, object_name: str) -> np.ndarray:
        return np.asarray(
            self.sim.data.body_xpos[self.obj_body_id[object_name]], dtype=float
        )

    def object_origin_distance(self, object_name: str) -> float | None:
        origin = self.original_object_poses.get(object_name)
        if origin is None:
            return None
        return bottom_center_xy_distance(
            self._object_bottom_center(object_name),
            origin,
        )

    def cleaning_tool_return_region_distance(self) -> float:
        return bottom_center_xy_distance(
            self._object_bottom_center("cleaning_tool"),
            self.cleaning_tool_return_region_center_xy,
        )

    def cleaning_tool_in_return_region(self) -> bool:
        center = self._object_bottom_center("cleaning_tool")[:2]
        half = self.cleaning_tool_return_region_size_xy * 0.5
        return bool(
            np.all(
                np.abs(center - self.cleaning_tool_return_region_center_xy)
                <= half + self.tool_return_region_tolerance
            )
        )

    def _physical_state(self) -> PhysicalState:
        distances = {name: self.object_origin_distance(name) for name in CUP_NAMES}
        return PhysicalState(
            primary_cup_at_origin=bool(
                distances[PRIMARY_CUP] is not None
                and distances[PRIMARY_CUP] < self.origin_threshold
            ),
            primary_cup_on_dining_table=bool(
                OU.check_obj_fixture_contact(self, PRIMARY_CUP, self.dining_table)
            ),
            secondary_cup_at_origin=bool(
                distances[SECONDARY_CUP] is not None
                and distances[SECONDARY_CUP] < self.origin_threshold
            ),
            secondary_cup_on_dining_table=bool(
                OU.check_obj_fixture_contact(self, SECONDARY_CUP, self.dining_table)
            ),
            cleaning_tool_in_return_region=self.cleaning_tool_in_return_region(),
            cleaning_tool_on_dining_table=bool(
                OU.check_obj_fixture_contact(
                    self, "cleaning_tool", self.dining_table
                )
            ),
            cleaning_tool_released=bool(
                OU.gripper_obj_far(self, "cleaning_tool", th=self.release_distance)
            ),
            primary_cup_released=bool(
                OU.gripper_obj_far(self, PRIMARY_CUP, th=self.release_distance)
            ),
            secondary_cup_released=bool(
                OU.gripper_obj_far(self, SECONDARY_CUP, th=self.release_distance)
            ),
        )

    def _update_wipe_progress(self, object_name: str) -> None:
        origin = self.original_object_poses.get(object_name)
        origin_xy = None if origin is None else np.asarray(origin, dtype=float)[:2]
        if origin_xy is None or not self._cup_semantic(object_name, "moved"):
            self._sync_stain_visual(object_name)
            return
        sponge_grasped = OU.check_obj_grasped(
            self, "cleaning_tool", threshold=0.6
        )
        sponge_contact_table = OU.check_obj_fixture_contact(
            self, "cleaning_tool", self.dining_table
        )
        sponge_xy = self.get_object_world_position("cleaning_tool")[:2]
        near_stain = bool(np.linalg.norm(sponge_xy - origin_xy) <= self.wipe_radius)
        contacts = self.table_contact_positions_by_object[object_name]
        if sponge_contact_table and sponge_grasped and near_stain:
            new_contact = all(
                np.linalg.norm(sponge_xy - recorded_xy) >= self.contact_separation
                for recorded_xy in contacts
            )
            if new_contact:
                contacts.append(sponge_xy.copy())
                self.table_contact_timer_by_object[object_name] += 1
        if contacts:
            positions = np.asarray(contacts, dtype=float)
            self.wipe_sweep_range_by_object[object_name] = float(
                np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
            )
        contact_progress = min(
            1.0,
            self.table_contact_timer_by_object[object_name]
            / float(self.minimum_contact_count),
        )
        sweep_progress = min(
            1.0,
            self.wipe_sweep_range_by_object[object_name]
            / self.minimum_wipe_sweep,
        )
        self.stain_progress_by_object[object_name] = max(
            self.stain_progress_by_object[object_name],
            min(contact_progress, sweep_progress),
        )
        covered = any(
            np.linalg.norm(contact - origin_xy) <= self.wipe_radius
            for contact in contacts
        )
        if (
            covered
            and self.table_contact_timer_by_object[object_name]
            >= self.minimum_contact_count
            and self.wipe_sweep_range_by_object[object_name]
            >= self.minimum_wipe_sweep
        ):
            changed = record_stain_cleaned(self._semantic_state, object_name)
            if changed or self._stain_cleaned(object_name):
                self.original_area_wiped_by_object[object_name] = True
                self.stain_progress_by_object[object_name] = 1.0
        self._sync_stain_visual(object_name)

    def _sync_stain_visual(self, object_name: str) -> None:
        model = getattr(getattr(self, "sim", None), "model", None)
        if model is None:
            return
        remaining = max(
            0.0, 1.0 - float(self.stain_progress_by_object[object_name])
        )
        prefix = STAIN_GEOM_PREFIX_BY_OBJECT[object_name]
        initial_alphas = self._stain_initial_alphas_by_object[object_name]
        for index, initial_alpha in enumerate(initial_alphas):
            try:
                geom_id = int(model.geom_name2id(f"{prefix}{index:02d}"))
            except Exception:
                continue
            if geom_id >= 0:
                model.geom_rgba[geom_id][3] = float(initial_alpha * remaining)

    def _cup_semantic(self, object_name: str, fact: str) -> bool:
        return bool(getattr(self._semantic_state, f"{object_name}_{fact}"))

    def _stain_cleaned(self, object_name: str) -> bool:
        prefix = object_name.removesuffix("_cup")
        return bool(getattr(self._semantic_state, f"{prefix}_stain_cleaned"))

    @staticmethod
    def _cup_physical(
        physical: PhysicalState, object_name: str, fact: str
    ) -> bool:
        return bool(getattr(physical, f"{object_name}_{fact}"))

    def _dining_table_surface_z(self) -> float:
        try:
            points = np.asarray(self.dining_table.get_bbox_points(), dtype=float)
            if points.ndim == 2 and points.shape[1] >= 3:
                return float(np.max(points[:, 2]))
        except Exception:
            pass
        position = np.asarray(
            getattr(self.dining_table, "pos", (0, 0, 0)), dtype=float
        )
        size = np.asarray(
            getattr(self.dining_table, "size", (0, 0, 0)), dtype=float
        )
        return float(position[2] + 0.5 * size[2])

    def _validate_cleaning_tool_return_region(self) -> None:
        points = np.asarray(self.dining_table.get_bbox_points(), dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            raise RuntimeError("WX-02 dining-table bounds are unavailable")
        lower = np.min(points[:, :2], axis=0)
        upper = np.max(points[:, :2], axis=0)
        half = self.cleaning_tool_return_region_size_xy * 0.5
        region_lower = self.cleaning_tool_return_region_center_xy - half
        region_upper = self.cleaning_tool_return_region_center_xy + half
        if np.any(region_lower < lower) or np.any(region_upper > upper):
            raise ValueError(
                "WX-02 cleaning-tool return region must stay inside the dining table"
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
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        bottom_offset = np.asarray(
            self.objects[object_name].bottom_offset, dtype=float
        )
        return body_pos + body_rotation @ bottom_offset


def _positive(value: Any, label: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"WX-02 {label} must be positive")
    return resolved


def _pair(value: Any, label: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"WX-02 {label} must contain two finite values")
    return float(array[0]), float(array[1])


def _positive_pair(value: Any, label: str) -> tuple[float, float]:
    pair = _pair(value, label)
    if any(item <= 0.0 for item in pair):
        raise ValueError(f"WX-02 {label} must contain positive values")
    return pair


def _rgba(value: Any, label: str) -> tuple[float, float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 4 or not np.all(np.isfinite(array)):
        raise ValueError(f"WX-02 {label} must contain four finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"WX-02 {label} values must be in [0, 1]")
    return tuple(float(item) for item in array)


def _stain_spots(value: Any) -> tuple[tuple[float, float, float, float, float], ...]:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        array = np.asarray(
            [
                (-0.028, -0.003, 0.022, 0.016, 0.72),
                (-0.015, 0.008, 0.028, 0.019, 0.86),
                (-0.002, -0.006, 0.030, 0.021, 1.00),
                (0.014, 0.006, 0.027, 0.019, 0.90),
                (0.027, -0.004, 0.020, 0.015, 0.76),
            ],
            dtype=float,
        )
    if array.ndim != 2 or array.shape[1] != 5 or not np.all(np.isfinite(array)):
        raise ValueError("WX-02 stain_spots must be finite five-value records")
    if np.any(array[:, 2:4] <= 0.0) or np.any(array[:, 4] < 0.0) or np.any(
        array[:, 4] > 1.0
    ):
        raise ValueError("WX-02 stain spot radii/alphas are invalid")
    return tuple(tuple(float(item) for item in row) for row in array)


def _pose3(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        raise ValueError(f"WX-02 {label} must contain a finite XYZ pose")
    return array[:3]


__all__ = [
    "CLEANING_TOOL_RETURN_REGION_GEOM",
    "CLEANING_TOOL_XML",
    "CUP_XML",
    "STAIN_GEOM_PREFIX_BY_OBJECT",
    "WX02CleanUnderCupAndReturnIt",
]
