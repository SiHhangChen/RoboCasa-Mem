"""RoboCasa scene and physical-state reader for WR-02."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import robosuite.utils.transform_utils as T

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen
from robocasa.utils import object_utils as OU

from bench.tasks.wr02.asset_poses import (
    SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M,
    SHELF_UPRIGHT_ROTATION_AXIS,
    SHELF_UPRIGHT_ROTATION_RAD,
)
from bench.tasks.wr02.domain.contracts import (
    CONDIMENT_IDS,
    HOME_REGIONS,
    SINK_SIDE_WORLD_OFFSETS_M,
    USE_REGIONS,
)
from bench.tasks.wr02.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    record_sink_stage,
)

DEFAULT_LAYOUT_ID = 32
STORAGE_FIXTURE_NAME = "opencabinet_1_main_group_1"
STORAGE_FIXTURE_CENTER_Z = 1.58
DEFAULT_CONDIMENT_OBJECT_GROUPS = {
    "ketchup": "ketchup",
    "mustard": "mustard",
}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXED_CONDIMENT_ASSETS = {
    "ketchup": str(
        ROOT
        / "robocasa/robocasa/models/assets/objects/objaverse/sponge/sponge_3/model.xml"
    ),
    "mustard": str(
        ROOT
        / "robocasa/robocasa/models/assets/objects/aigen_objs/bar_soap/bar_soap_4/model.xml"
    ),
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


class WR02ReturnCondimentPair(Kitchen):
    """Shelf-memory task that stages a sponge and soap beside the sink."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def __init__(
        self,
        home_threshold: float = 0.16,
        sink_stage_threshold: float = 0.20,
        return_region_xy_margin: float = 0.04,
        return_region_height_tolerance: float = 0.10,
        return_level_min_separation: float = 0.20,
        home_assignment: dict[str, str] | None = None,
        home_region_placements: dict[str, dict[str, Any]] | None = None,
        use_assignment: dict[str, str] | None = None,
        episode_case_id: int = 0,
        condiment_object_groups: dict[str, str] | None = None,
        condiment_object_scales: dict[str, float] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        self.home_threshold = float(home_threshold)
        self.sink_stage_threshold = float(sink_stage_threshold)
        self.return_region_xy_margin = float(return_region_xy_margin)
        self.return_region_height_tolerance = float(return_region_height_tolerance)
        self.return_level_min_separation = float(return_level_min_separation)
        self.condiment_names = CONDIMENT_IDS
        self.condiment_object_groups = dict(DEFAULT_FIXED_CONDIMENT_ASSETS)
        if condiment_object_groups is not None:
            self.condiment_object_groups.update(
                {
                    str(name): str(group)
                    for name, group in dict(condiment_object_groups).items()
                }
            )
        self.condiment_object_scales = {
            str(name): float(scale)
            for name, scale in (condiment_object_scales or {}).items()
        }
        self.episode_case_id = int(episode_case_id)
        self.home_assignment = _validate_assignment(
            home_assignment,
            expected_values=HOME_REGIONS,
            label="home_assignment",
        )
        self.home_region_placements = _validate_home_region_placements(
            home_region_placements
        )
        self.use_assignment = _validate_assignment(
            use_assignment,
            expected_values=USE_REGIONS,
            label="use_assignment",
        )
        self._semantic_state = SemanticState()
        kwargs.setdefault("obj_registries", ("objaverse", "aigen", "lightwheel"))
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.condiment_shelf = self.register_fixture_ref(
            "condiment_shelf",
            dict(id=STORAGE_FIXTURE_NAME, full_name_check=True),
        )
        shelf_pos = np.asarray(self.condiment_shelf.pos, dtype=float).copy()
        shelf_pos[2] = STORAGE_FIXTURE_CENTER_Z
        self.condiment_shelf.set_pos(shelf_pos)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.sink_counter = self.register_fixture_ref(
            "sink_counter",
            dict(id=FixtureType.COUNTER, ref=self.sink, full_depth_region=True),
        )
        if self.condiment_shelf is None or self.sink is None or self.sink_counter is None:
            raise ValueError(
                "WR-02 requires layout 32 with its open shelf, sink, and adjacent counter"
            )
        self.sink_stage_fixture = self.sink_counter
        self.init_robot_base_ref = self.condiment_shelf

    def get_ep_meta(self):
        from bench.tasks.wr02.prompts import resolve_prompt_spec

        ep_meta = super().get_ep_meta()
        prompt = resolve_prompt_spec(variant="memory")
        ep_meta["lang"] = prompt.policy_prompt
        ep_meta["initial_prompt"] = prompt.initial_prompt
        ep_meta["refs"] = ep_meta.get("refs", {})
        ep_meta["refs"]["home_assignment"] = dict(getattr(self, "home_assignment", {}))
        ep_meta["refs"]["use_assignment"] = dict(getattr(self, "use_assignment", {}))
        ep_meta["refs"]["episode_case_id"] = self.episode_case_id
        ep_meta["refs"]["condiment_names"] = list(self.condiment_names)
        ep_meta["refs"]["home_regions"] = list(HOME_REGIONS)
        ep_meta["refs"]["shelf_stations"] = ["shelf_lower", "shelf_upper"]
        ep_meta["refs"]["sink_station"] = "sink_place_station"
        return ep_meta

    def _setup_scene(self):
        super()._setup_scene()
        self.original_home_region_by_condiment = dict(self.home_assignment)
        self.original_home_pose_by_condiment = {
            condiment_name: self._placement_pose(condiment_name)
            for condiment_name in self.condiment_names
        }
        self._semantic_state = SemanticState()
        self.current_condiment_home_correct = {
            condiment_name: self._condiment_in_return_shelf(condiment_name)
            for condiment_name in self.condiment_names
        }
        self.condiments_in_return_shelf = all(self.current_condiment_home_correct.values())
        self.condiment_shelf_level_relation_correct = (
            self._shelf_level_relation_correct()
        )
        self.current_condiment_positions = {
            condiment_name: self._obj_pos(condiment_name).astype(float).tolist()
            for condiment_name in self.condiment_names
        }
        self._sync_semantic_diagnostics()

    def _get_obj_cfgs(self):
        cfgs: list[dict[str, Any]] = []
        condiment_by_home = {
            home_name: condiment_name
            for condiment_name, home_name in self.home_assignment.items()
        }

        for home_name in HOME_REGIONS:
            condiment_name = condiment_by_home[home_name]
            home_placement = self.home_region_placements[home_name]
            placement = dict(
                fixture=self.condiment_shelf,
                sample_region_kwargs=dict(
                    shelf_level=int(home_placement["shelf_level"])
                ),
                size=tuple(home_placement["size"]),
                margin=float(home_placement["margin"]),
                rotation=SHELF_UPRIGHT_ROTATION_RAD,
                rotation_axis=SHELF_UPRIGHT_ROTATION_AXIS,
                offset=(
                    0.0,
                    0.0,
                    SHELF_UPRIGHT_PLACEMENT_Z_OFFSETS_M[condiment_name],
                ),
                pos=tuple(home_placement["pos"]),
            )
            cfg = dict(
                name=condiment_name,
                obj_groups=self.condiment_object_groups[condiment_name],
                graspable=True,
                init_robot_here=(condiment_name == "ketchup"),
                placement=placement,
            )
            if condiment_name in self.condiment_object_scales:
                cfg["object_scale"] = self.condiment_object_scales[condiment_name]
            cfgs.append(cfg)

        return cfgs

    def update_state(self):
        super().update_state()
        self._update_task_state()

    def _check_success(self):
        return self._success_report().success

    def _success_report(self):
        self._update_task_state()
        physical = PhysicalState(
            condiments_in_return_shelf=self.condiments_in_return_shelf,
            condiment_shelf_level_relation_correct=(
                self.condiment_shelf_level_relation_correct
            ),
            condiments_released=all(
                OU.gripper_obj_far(self, condiment_name, th=0.15)
                for condiment_name in self.condiment_names
            ),
        )
        return evaluate_success(self._semantic_state, physical)

    def task_diagnostics(self) -> dict[str, Any]:
        """Expose every final WR-02 predicate to the production recorder."""
        report = self._success_report()
        return {
            "success": report.success,
            **report.checks(),
            "current_condiment_home_correct": dict(
                self.current_condiment_home_correct
            ),
            "current_condiment_relation": dict(self.current_condiment_relation),
            "current_condiment_positions": dict(self.current_condiment_positions),
        }

    def _update_task_state(self) -> None:
        for condiment_name in self.condiment_names:
            if self._condiment_at_sink_side(condiment_name):
                record_sink_stage(self._semantic_state, condiment_name)
        self.current_condiment_home_correct = {
            condiment_name: self._condiment_in_return_shelf(condiment_name)
            for condiment_name in self.condiment_names
        }
        self.condiments_in_return_shelf = all(self.current_condiment_home_correct.values())
        self.condiment_shelf_level_relation_correct = (
            self._shelf_level_relation_correct()
        )
        self.current_condiment_relation = self._condiment_relation_diagnostics()
        self.current_condiment_positions = {
            condiment_name: self._obj_pos(condiment_name).astype(float).tolist()
            for condiment_name in self.condiment_names
        }
        self._sync_semantic_diagnostics()

    def _sync_semantic_diagnostics(self) -> None:
        self.ketchup_staged_at_sink = bool(
            self._semantic_state.staged_at_sink["ketchup"]
        )
        self.mustard_staged_at_sink = bool(
            self._semantic_state.staged_at_sink["mustard"]
        )

    def home_region_center(self, region_name: str) -> np.ndarray:
        region_name = str(region_name)
        for condiment_name, home_name in self.original_home_region_by_condiment.items():
            if home_name == region_name:
                pose = self.original_home_pose_by_condiment.get(condiment_name)
                if pose is not None:
                    center_xy, _ = self._home_region_xy_geometry(region_name)
                    return np.array(
                        [center_xy[0], center_xy[1], float(pose[2])],
                        dtype=float,
                    )
        raise KeyError(f"unknown or unavailable home region {region_name!r}")

    def return_region_pose(self, region_name: str) -> np.ndarray:
        """Return the center of the configured lower/upper shelf placement range."""
        if str(region_name) not in HOME_REGIONS:
            raise KeyError(f"unknown WR-02 return region {region_name!r}")
        return self.home_region_center(str(region_name))

    def home_region_surface_z(self, region_name: str) -> float:
        """Return the world Z of the configured shelf surface."""
        settings = self.home_region_placements.get(str(region_name))
        if settings is None:
            raise KeyError(f"unknown WR-02 home region {region_name!r}")
        reset_regions = self.condiment_shelf.get_reset_regions(
            shelf_level=int(settings["shelf_level"])
        )
        if len(reset_regions) != 1:
            raise ValueError(f"WR-02 {region_name} must resolve to one shelf region")
        reset_region = next(iter(reset_regions.values()))
        return float(
            np.asarray(self.condiment_shelf.pos, dtype=float)[2]
            + float(reset_region["offset"][2])
        )

    def use_station_pose(self, condiment_name: str) -> np.ndarray:
        condiment_name = str(condiment_name)
        use_region = self.use_assignment.get(condiment_name)
        if use_region in SINK_SIDE_WORLD_OFFSETS_M:
            return np.asarray(self.sink.pos, dtype=float) + np.asarray(
                SINK_SIDE_WORLD_OFFSETS_M[use_region], dtype=float
            )
        raise KeyError(
            f"unknown WR-02 condiment/use assignment {condiment_name!r}: {use_region!r}"
        )

    def _condiment_at_sink_side(self, condiment_name: str) -> bool:
        # The calibrated left/right targets rest on the sink fixture's outer
        # deck in layout 32. Depending on the exact target, MuJoCo reports that
        # support contact against either the sink geom or its adjacent counter.
        supported = any(
            OU.check_obj_fixture_contact(self, str(condiment_name), fixture)
            for fixture in (self.sink, self.sink_counter)
        )
        return bool(
            self._obj_near_pose(
                condiment_name,
                self.use_station_pose(condiment_name),
                self.sink_stage_threshold,
            )
            and supported
        )

    def set_object_pose(self, obj_name: str, pos: Any, quat_xyzw: Any | None = None) -> None:
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

    def _placement_pose(self, obj_name: str) -> list[float] | None:
        placement = getattr(self, "object_placements", {}).get(str(obj_name))
        if placement is not None:
            return np.array(placement[0], dtype=float).tolist()
        if hasattr(self, "obj_body_id") and obj_name in getattr(self, "obj_body_id", {}):
            return self._obj_pos(str(obj_name)).astype(float).tolist()
        return None

    def _obj_pos(self, obj_name: str) -> np.ndarray:
        return np.array(self.sim.data.body_xpos[self.obj_body_id[str(obj_name)]], dtype=float)

    def _obj_near_pose(self, obj_name: str, pose: Any, threshold: float) -> bool:
        if pose is None:
            return False
        pose_arr = np.array(pose, dtype=float).reshape(-1)
        if pose_arr.size < 2 or not np.all(np.isfinite(pose_arr[:2])):
            return False
        return bool(np.linalg.norm(self._obj_pos(obj_name)[:2] - pose_arr[:2]) <= threshold)

    def _condiment_at_original_home(self, condiment_name: str) -> bool:
        pose = self.original_home_pose_by_condiment.get(str(condiment_name))
        return self._obj_near_pose(str(condiment_name), pose, self.home_threshold)

    def _home_region_xy_geometry(
        self, region_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        region_name = str(region_name)
        settings = self.home_region_placements.get(region_name)
        if settings is None:
            raise KeyError(f"unknown WR-02 home region {region_name!r}")
        reset_regions = self.condiment_shelf.get_reset_regions(
            shelf_level=int(settings["shelf_level"])
        )
        if len(reset_regions) != 1:
            raise ValueError(f"WR-02 {region_name} must resolve to one shelf region")
        reset_region = next(iter(reset_regions.values()))
        margin = float(settings["margin"])
        outer_size = np.asarray(reset_region["size"], dtype=float)[:2] - margin
        requested_size = np.asarray(settings["size"], dtype=float)[:2]
        inner_size = np.minimum(outer_size, requested_size)
        normalized_pos = np.asarray(settings["pos"], dtype=float)[:2]
        intra_offset = 0.5 * (outer_size - inner_size) * normalized_pos
        local_center = (
            np.asarray(reset_region["offset"], dtype=float)[:2] + intra_offset
        )
        rot = float(self.condiment_shelf.rot)
        rotation = np.array(
            [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]],
            dtype=float,
        )
        world_center = (
            np.asarray(self.condiment_shelf.pos, dtype=float)[:2]
            + rotation @ local_center
        )
        return world_center, inner_size

    def _condiment_in_return_shelf(self, condiment_name: str) -> bool:
        try:
            region_name = self.original_home_region_by_condiment[str(condiment_name)]
            center, size = self._home_region_xy_geometry(region_name)
            expected_z = float(self.home_region_center(region_name)[2])
        except (IndexError, KeyError, TypeError, ValueError):
            return False
        pos = self._obj_pos(str(condiment_name))
        relative_world = pos[:2] - center
        rot = float(self.condiment_shelf.rot)
        rotation_inverse = np.array(
            [[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]],
            dtype=float,
        )
        relative = rotation_inverse @ relative_world
        half_extent = 0.5 * size + self.return_region_xy_margin
        height_error = abs(float(pos[2]) - expected_z)
        return bool(
            np.all(np.abs(relative) <= half_extent)
            and height_error <= self.return_region_height_tolerance
        )

    def _shelf_level_condiments(self) -> tuple[str, str]:
        condiment_by_region = {
            region_name: condiment_name
            for condiment_name, region_name in self.original_home_region_by_condiment.items()
        }
        return (
            condiment_by_region["shelf_lower"],
            condiment_by_region["shelf_upper"],
        )

    def _shelf_level_relation_correct(self) -> bool:
        try:
            lower_name, upper_name = self._shelf_level_condiments()
        except (IndexError, KeyError, TypeError, ValueError):
            return False
        separation = float(
            self._obj_pos(upper_name)[2] - self._obj_pos(lower_name)[2]
        )
        return bool(separation >= self.return_level_min_separation)

    def _condiment_relation_diagnostics(self) -> dict[str, Any]:
        try:
            lower_name, upper_name = self._shelf_level_condiments()
            separation = float(
                self._obj_pos(upper_name)[2] - self._obj_pos(lower_name)[2]
            )
        except (IndexError, KeyError, TypeError, ValueError):
            return {"valid": False}
        return {
            "valid": True,
            "expected_lower": lower_name,
            "expected_upper": upper_name,
            "vertical_separation_m": separation,
            "minimum_separation_m": self.return_level_min_separation,
            "correct": bool(separation >= self.return_level_min_separation),
        }

    def _home_swap_detected(self) -> bool:
        return not self._shelf_level_relation_correct()


def _validate_assignment(
    value: dict[str, str] | None,
    *,
    expected_values: tuple[str, str],
    label: str,
) -> dict[str, str]:
    assignment = {str(key): str(item) for key, item in dict(value or {}).items()}
    if set(assignment) != set(CONDIMENT_IDS):
        raise ValueError(f"WR-02 {label} must assign both condiments")
    if set(assignment.values()) != set(expected_values):
        raise ValueError(f"WR-02 {label} must be bijective")
    return assignment


def _validate_home_region_placements(
    value: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    raw = DEFAULT_HOME_REGION_PLACEMENTS if value is None else value
    if set(raw) != set(HOME_REGIONS):
        raise ValueError("WR-02 home-region placements must define lower and upper")
    placements: dict[str, dict[str, Any]] = {}
    for region_name in HOME_REGIONS:
        settings = dict(raw[region_name])
        shelf_level = int(settings["shelf_level"])
        size = tuple(float(item) for item in settings["size"])
        pos = tuple(float(item) for item in settings["pos"])
        margin = float(settings.get("margin", 0.04))
        if len(size) != 2 or min(size) <= 0.0:
            raise ValueError(f"invalid WR-02 placement size for {region_name}")
        if len(pos) != 2 or any(abs(item) > 1.0 for item in pos):
            raise ValueError(f"invalid WR-02 normalized placement position for {region_name}")
        if shelf_level < 0 or margin < 0.0:
            raise ValueError(f"invalid WR-02 shelf placement for {region_name}")
        placements[region_name] = {
            "shelf_level": shelf_level,
            "size": size,
            "pos": pos,
            "margin": margin,
        }
    if (
        placements["shelf_lower"]["shelf_level"]
        >= placements["shelf_upper"]["shelf_level"]
    ):
        raise ValueError("WR-02 shelf_lower must be below shelf_upper")
    return placements


__all__ = [
    "DEFAULT_LAYOUT_ID",
    "DEFAULT_CONDIMENT_OBJECT_GROUPS",
    "DEFAULT_FIXED_CONDIMENT_ASSETS",
    "DEFAULT_HOME_REGION_PLACEMENTS",
    "STORAGE_FIXTURE_NAME",
    "STORAGE_FIXTURE_CENTER_Z",
    "WR02ReturnCondimentPair",
]
