"""RoboCasa environment and physical observations for WX-01."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.models import assets_root
from robocasa.utils import object_utils as OU

from bench.tasks.wx01 import object_randomization
from bench.tasks.wx01.domain.geometry import bottom_center_xy_distance
from bench.tasks.wx01.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    record_wipe_completion,
)
from bench.tasks.wx01.evaluation import WX01EvaluationSnapshot, WX01Evaluator
from bench.tasks.wx01.prompts import resolve_prompt_spec


def _rgba(value: Any, label: str) -> tuple[float, float, float, float]:
    """Module-level helper to validate RGBA tuples for stains etc."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 4 or not np.all(np.isfinite(arr)):
        raise ValueError(f"WX-01 {label} must be 4 finite values")
    if not (0.0 <= arr[3] <= 1.0):
        raise ValueError(f"WX-01 {label} alpha must be in [0,1]")
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])


# Base cup asset (material will be overridden per-episode via apply_cup_material)
BLOCKING_OBJECT_XML = f"{assets_root}/objects/objaverse/cup/cup_5/model.xml"

# Fallback (will be overridden by self.sponge_type at runtime)
CLEANING_TOOL_XML = f"{assets_root}/objects/objaverse/sponge/sponge_0/model.xml"


class WX01ReturnObjectToOriginalPlace(Kitchen):
    """Six-stage task: wipe a remembered spot and return its displaced cup."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        *,
        origin_threshold: float = 0.01,
        wipe_radius: float = 0.20,
        contact_separation: float = 0.02,
        minimum_contact_count: int = 2,
        minimum_wipe_sweep: float = 0.07,
        release_distance: float = 0.15,
        blocking_object_scale: float = 0.9,
        blocking_object_placement_size: tuple[float, float] = (0.16, 0.20),
        blocking_object_placement_pos: tuple[float, float] = (1.0, 1.0),
        blocking_object_rotation: tuple[float, float] = (0.0, 0.0),
        cup_material_id: str = "ceramic_white",
        sponge_type: str = "sponge_0",
        stain_color_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        stain_circle_radius: float = 0.045,
        stain_z_offset: float = 0.0018,
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
        self.cup_material_id = object_randomization.validate_cup_material(cup_material_id)
        self.sponge_type = object_randomization.validate_sponge_type(sponge_type)
        # Resolve the actual sponge asset path for this episode
        self._cleaning_tool_xml = str(
            object_randomization.resolve_sponge_xml(
                sponge_type=self.sponge_type, asset_root=assets_root
            )
        )
        self.stain_color_rgba = _rgba(stain_color_rgba, "stain_color_rgba")
        self.stain_circle_radius = _positive(stain_circle_radius, "stain_circle_radius")
        self.stain_z_offset = _positive(stain_z_offset, "stain_z_offset")
        self._stain_geom_names = ("wx01_stain_circle",)
        self._wx01_evaluator: WX01Evaluator | None = None
        super().__init__(**kwargs)

    def edit_model_xml(self, xml_str: str) -> str:
        """Overlay one white, non-colliding circle centered on the remembered cup pose."""

        xml_str = super().edit_model_xml(xml_str)
        root = ET.fromstring(xml_str)
        worldbody = root.find("worldbody")
        if worldbody is None:
            return xml_str

        # Avoid duplicate names if re-called
        existing = {g.get("name") for g in worldbody.iter("geom") if g.get("name")}

        surface_z = self._dining_table_surface_z() + self.stain_z_offset

        # Before reset-time episode binding, use a harmless placeholder. The
        # authoritative center is installed by _position_stains_on_remembered_area.
        base_center = np.array([2.55, -3.49], dtype=float)  # good for layout 38 dining area
        for name in self._stain_geom_names:
            if name in existing:
                continue
            xy = base_center
            rgba = self.stain_color_rgba
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": name,
                    "type": "cylinder",
                    "pos": f"{xy[0]:.8f} {xy[1]:.8f} {surface_z:.8f}",
                    "size": f"{self.stain_circle_radius:.6f} 0.0008",
                    "rgba": " ".join(f"{v:.6f}" for v in rgba),
                    "contype": "0",
                    "conaffinity": "0",
                    "group": "1",
                },
            )
        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _dining_table_surface_z(self) -> float:
        try:
            points = np.asarray(self.dining_table.get_bbox_points(), dtype=float)
            if points.ndim == 2 and points.shape[1] >= 3:
                return float(np.max(points[:, 2]))
        except Exception:
            pass
        try:
            pos = np.asarray(getattr(self.dining_table, "pos", (0, 0, 0)), dtype=float)
            size = np.asarray(getattr(self.dining_table, "size", (0, 0, 0)), dtype=float)
            return float(pos[2] + 0.5 * size[2])
        except Exception:
            return 0.78

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
        self._wx01_evaluator = None
        self.table_contact_positions: list[np.ndarray] = []
        self.table_contact_timer = 0
        self.wipe_sweep_range = 0.0
        self.original_area_wiped = False
        self.stain_progress = 0.0
        self.original_tool_pose = self._placement_pose("cleaning_tool")

        # CRITICAL FIX for stain vs wipe target alignment (~5cm mismatch):
        #
        # Do NOT blindly assign self.original_blocking_object_pose from the current
        # sim placement here.
        #
        # In the short-start flow:
        #   - apply_initial_object_plan places the cup at the *remembered*
        #     (pre-displacement) episode-sampled world position.
        #   - It calls set_original_object_poses(..., blocking_object=remembered_pose).
        #     This is the authoritative place that stores the "dirty spot" pose
        #     AND calls _position_stains_on_remembered_area() to put the white
        #     target circle exactly where the cleaning tool will wipe.
        #   - Only *after* that the cup is displaced to the temporary location.
        #
        # When _setup_scene runs (especially on resets for recording), the live
        # placement of "blocking_object" is usually already the temporary displaced pose.
        #
        # Overwriting the remembered pose here is what caused the stains to be
        # generated at a different location from the wipe target.
        #
        # Rule: original_blocking_object_pose (the remembered dirty spot) must
        # only be written by set_original_object_poses. We only take a fallback
        # if it was never set.

        if getattr(self, "original_blocking_object_pose", None) is None:
            self.original_blocking_object_pose = self._placement_pose("blocking_object")

        # Re-position stains using the (hopefully correct) remembered pose.
        if getattr(self, "original_blocking_object_pose", None) is not None:
            self._position_stains_on_remembered_area()

        # Apply per-episode appearance (cup material + correct sponge model is already in _get_obj_cfgs)
        self._sync_stain_visual()  # ensure stains start visible (full alpha)
        if self.cup_material_id and "blocking_object" in self.obj_body_id:
            try:
                model = self.sim.model
                root_body_id = self.obj_body_id["blocking_object"]
                object_randomization.apply_cup_material(
                    model, root_body_id=root_body_id, material_id=self.cup_material_id
                )
            except Exception:
                # Best-effort for rendering / dry runs
                pass

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        """Create one fixed sponge (per-episode type) and one fixed cup (material applied later)."""

        # Use the episode-resolved sponge XML (set in __init__ via sponge_type)
        sponge_xml = getattr(self, "_cleaning_tool_xml", None) or CLEANING_TOOL_XML

        return [
            dict(
                name="cleaning_tool",
                obj_groups=sponge_xml,
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
        """Install the original poses sampled before the short-start displacement.

        Also positions the visual stains exactly on the remembered dirty spot
        (the location the cup occupied before being displaced).
        """

        tool = _pose3(cleaning_tool, "cleaning_tool original pose")
        blocking = _pose3(blocking_object, "blocking_object original pose")
        self.original_tool_pose = tool.tolist()
        self.original_blocking_object_pose = blocking.tolist()
        refs = self._ep_meta.setdefault("refs", {})
        refs["original_tool_pose"] = tool.tolist()
        refs["original_blocking_object_pose"] = blocking.tolist()

        # Now that we have the authoritative remembered position, place the stains there.
        self._position_stains_on_remembered_area()
        self._sync_stain_visual()

    def record_wipe_completed(self) -> bool:
        """Latch the historical wipe event after an accepted physical sweep."""

        self.original_area_wiped = True
        self.stain_progress = 1.0
        self._sync_stain_visual()
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

        snapshot = self.evaluation_snapshot()
        if self._wx01_evaluator is not None:
            return self._wx01_evaluator.observe(snapshot).to_diagnostics()
        return {
            **dict(snapshot.details),
            "success": bool(snapshot.domain_success),
            "terminal": bool(snapshot.domain_success),
            "failure_code": None,
        }

    def evaluation_snapshot(self) -> WX01EvaluationSnapshot:
        """Expose wipe history and current cleanup and return predicates."""

        self.update_state()
        physical = self._physical_state()
        report = evaluate_success(self._semantic_state, physical)
        details = {
            "success_checks": report.checks(),
            "release_checks": {
                "cleaning_tool_released": physical.cleaning_tool_released,
                "blocking_object_released": physical.blocking_object_released,
            },
            "original_area_wiped": bool(self.original_area_wiped),
            "table_contact_timer": int(self.table_contact_timer),
            "wipe_sweep_range": float(self.wipe_sweep_range),
            "original_tool_pose": self.original_tool_pose,
            "original_blocking_object_pose": self.original_blocking_object_pose,
            "target_circle_center": self.original_blocking_object_pose,
            "target_circle_radius": float(self.stain_circle_radius),
            "blocking_object_origin_distance": self.blocking_object_origin_distance(),
            "blocking_object_on_dining_table": physical.blocking_object_on_dining_table,
            "objects_released": bool(
                physical.cleaning_tool_released and physical.blocking_object_released
            ),
        }
        return WX01EvaluationSnapshot(
            step_id=int(getattr(self, "timestep", 0)),
            remembered_area_wiped=self._semantic_state.remembered_area_wiped,
            cleaning_tool_released=physical.cleaning_tool_released,
            blocking_object_at_origin=physical.blocking_object_at_origin,
            blocking_object_on_dining_table=(
                physical.blocking_object_on_dining_table
            ),
            blocking_object_released=physical.blocking_object_released,
            domain_success=report.success,
            details=details,
        )

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
        newly_wiped = (
            covered
            and self.table_contact_timer >= self.minimum_contact_count
            and self.wipe_sweep_range >= self.minimum_wipe_sweep
        )
        if newly_wiped and not self.original_area_wiped:
            self.original_area_wiped = True
            self.stain_progress = 1.0
            self._sync_stain_visual()
        elif self.original_area_wiped:
            self.stain_progress = 1.0
            self._sync_stain_visual()

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

    def _position_stains_on_remembered_area(self) -> None:
        """Place the single white target circle at the remembered cup center."""

        model = getattr(getattr(self, "sim", None), "model", None)
        if model is None:
            return
        if self.original_blocking_object_pose is None:
            return

        origin_xy = np.asarray(self.original_blocking_object_pose, dtype=float)[:2]
        surface_z = self._dining_table_surface_z() + self.stain_z_offset

        # The circle center is the exact placement target; there is no visual
        # offset between the marker and the remembered cup pose.
        for name in self._stain_geom_names:
            try:
                gid = int(model.geom_name2id(name))
            except Exception:
                continue
            if gid < 0:
                continue
            xy = origin_xy
            model.geom_pos[gid, 0] = float(xy[0])
            model.geom_pos[gid, 1] = float(xy[1])
            model.geom_pos[gid, 2] = float(surface_z)

    def _sync_stain_visual(self) -> None:
        """Fade the white target circle after the wipe is accepted."""

        model = getattr(getattr(self, "sim", None), "model", None)
        if model is None:
            return
        progress = float(getattr(self, "stain_progress", 0.0))
        remaining = max(0.0, 1.0 - progress)
        for name in getattr(self, "_stain_geom_names", ()):
            try:
                gid = int(model.geom_name2id(name))
            except Exception:
                continue
            if gid < 0:
                continue
            base = np.asarray(self.stain_color_rgba, dtype=float)
            model.geom_rgba[gid, :] = np.array(
                [*base[:3], base[3] * remaining], dtype=float
            )


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
