"""TS04 RoboCasa scene and authoritative physical-observation boundary."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen

from bench.tasks.ts04.button_asset import (
    TS04_BUTTON_MAX_TRAVEL_M,
    add_ts04_button_asset,
)
from bench.tasks.ts04.domain.contracts import (
    AXIS,
    FIXTURE_DISPLAY_NAMES,
    FIXTURE_NAMES,
    TARGET_OBJECT_NAME,
)
from bench.tasks.ts04.domain.geometry import (
    DEFAULT_DRAWER_WORLD_XYZS,
    DEFAULT_DRAWER_WORLD_YAW_DEG,
    normalize_drawer_world_xyzs,
)
from bench.tasks.ts04.domain.semantics import (
    PhysicalState,
    SemanticState,
    diagnostics,
    evaluate_success,
    record_close_attempt,
    record_completion_button_attempt,
    record_inspection_attempt,
    record_open_attempt,
)
from bench.tasks.ts04.physical_events import PhysicalSample, TS04PhysicalObserver
from bench.tasks.ts04.single_drawer import SingleTabletopDrawer
from bench.tasks.ts04.target_object import (
    TARGET_OBJECT_GEOM_NAME,
    add_target_red_cube,
)


DRAWER_FIXTURE_NAMES = {fixture: f"ts04_{fixture}_fixture" for fixture in FIXTURE_NAMES}
COMPLETION_BUTTON_GEOM_NAME = "bench_ts04_completion_button"
COMPLETION_BUTTON_JOINT_NAME = f"{COMPLETION_BUTTON_GEOM_NAME}_press_joint"
COMPLETION_BUTTON_MAX_TRAVEL = TS04_BUTTON_MAX_TRAVEL_M
DEFAULT_COMPLETION_BUTTON_WORLD_XYZ = (3.85, -2.06, 0.92)


class TS04InspectTwoDrawers(Kitchen):
    """Inspect two drawers, one containing a high-visibility red cube."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS
    EXPECTED_FIXTURE_NAMES = FIXTURE_NAMES
    FIXTURE_DISPLAY_NAMES = FIXTURE_DISPLAY_NAMES
    DRAWER_FIXTURE_PREFIX = "ts04"
    COMPLETION_BUTTON_GEOM_NAME = COMPLETION_BUTTON_GEOM_NAME
    COMPLETION_BUTTON_JOINT_NAME = COMPLETION_BUTTON_JOINT_NAME
    TARGET_OBJECT_GEOM_NAME = TARGET_OBJECT_GEOM_NAME
    TASK_LABEL = "TS04"
    LANG = (
        "Search the drawers for the red cube, closing each drawer after "
        "inspection. Stop after finding and closing its drawer, then press "
        "the completion button once for each drawer you inspected."
    )
    INITIAL_PROMPT = (
        "Exactly one of the two drawers contains a bright red cube. Search "
        "without revisiting drawers, and close each inspected drawer. Stop "
        "after finding and closing the target drawer. If finding it required "
        "inspecting k drawers, press the completion button exactly k times, "
        "releasing it between successive presses."
    )

    def __init__(
        self,
        *,
        fixture_names: tuple[str, ...] = FIXTURE_NAMES,
        drawer_world_xyzs: tuple[
            tuple[float, float, float], ...
        ] = DEFAULT_DRAWER_WORLD_XYZS,
        drawer_world_yaw_deg: float = DEFAULT_DRAWER_WORLD_YAW_DEG,
        completion_button_world_xyz: tuple[float, float, float] = (
            DEFAULT_COMPLETION_BUTTON_WORLD_XYZ
        ),
        target_fixture: str = FIXTURE_NAMES[0],
        open_threshold: float = 0.70,
        closed_threshold: float = 0.08,
        button_pressed_threshold_m: float = 0.004,
        button_released_threshold_m: float = 0.001,
        button_release_dwell_steps: int = 2,
        **kwargs: Any,
    ) -> None:
        names = tuple(str(name) for name in fixture_names)
        if names != tuple(self.EXPECTED_FIXTURE_NAMES):
            raise ValueError(
                f"{self.TASK_LABEL} fixture_names must equal "
                f"{tuple(self.EXPECTED_FIXTURE_NAMES)}"
            )
        if not 0.0 <= float(closed_threshold) < float(open_threshold) <= 1.0:
            raise ValueError("TS04 requires closed_threshold < open_threshold")
        self.fixture_names = names
        self.drawer_world_xyzs = normalize_drawer_world_xyzs(
            drawer_world_xyzs, expected_count=len(names)
        )
        self.drawer_fixture_names = {
            fixture: f"{self.DRAWER_FIXTURE_PREFIX}_{fixture}_fixture"
            for fixture in names
        }
        self.drawer_world_yaw_deg = float(drawer_world_yaw_deg)
        button_position = np.asarray(completion_button_world_xyz, dtype=float).reshape(-1)
        if button_position.size != 3 or not np.all(np.isfinite(button_position)):
            raise ValueError("TS04 completion_button_world_xyz must be a finite xyz value")
        self.completion_button_world_xyz = tuple(
            float(item) for item in button_position
        )
        self.target_fixture = str(target_fixture)
        if self.target_fixture not in names:
            raise ValueError("TS04 target_fixture must name one of the two drawers")
        self.open_threshold = float(open_threshold)
        self.closed_threshold = float(closed_threshold)
        self._physical_observer = TS04PhysicalObserver(
            fixture_names=names,
            target_fixture=self.target_fixture,
            drawer_open_threshold=self.open_threshold,
            drawer_closed_threshold=self.closed_threshold,
            button_pressed_threshold_m=float(button_pressed_threshold_m),
            button_released_threshold_m=float(button_released_threshold_m),
            button_release_dwell_steps=int(button_release_dwell_steps),
        )
        self._observer_armed = False
        super().__init__(**kwargs)

    # Scene construction -------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()
        self.storage_cabinets = {
            fixture: SingleTabletopDrawer(
                name=self.drawer_fixture_names[fixture],
                rng=self.rng,
            )
            for fixture in self.fixture_names
        }
        for fixture_name, fixture in self.storage_cabinets.items():
            model_name = self.drawer_fixture_names[fixture_name]
            self.fixtures[model_name] = fixture
            self.fixture_cfgs.append(
                {"name": model_name, "model": fixture, "type": "fixture"}
            )
        self.model.merge_objects(list(self.storage_cabinets.values()))

    def edit_model_xml(self, xml_str: str) -> str:
        """Install the TM01-style spring-loaded completion button."""

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
        if self.COMPLETION_BUTTON_GEOM_NAME not in existing:
            _add_completion_button_asset(
                worldbody,
                pos=self.completion_button_world_xyz,
                geom_name=self.COMPLETION_BUTTON_GEOM_NAME,
                joint_name=self.COMPLETION_BUTTON_JOINT_NAME,
            )
        add_target_red_cube(
            root,
            drawer_body_name=(
                f"{self.drawer_fixture_names[self.target_fixture]}_drawer"
            ),
            geom_name=self.TARGET_OBJECT_GEOM_NAME,
        )
        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _setup_kitchen_references(self) -> None:
        super()._setup_kitchen_references()
        self.tabletop_storage_cabinets = self.storage_cabinets
        self.tabletop_storage_cabinet = self.storage_cabinets[self.fixture_names[1]]
        self.storage_cabinet = self.tabletop_storage_cabinet
        self.cabinet = self.storage_cabinet
        self.fixture_refs["cabinet"] = (self.cabinet, False)
        for fixture_name, fixture in self.storage_cabinets.items():
            self.fixture_refs[fixture_name] = (fixture, False)
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, size=(0.95, 0.55)),
        )
        yaw = float(np.deg2rad(self.drawer_world_yaw_deg))
        for fixture_name, world_xyz in zip(
            self.fixture_names,
            self.drawer_world_xyzs,
            strict=True,
        ):
            self.storage_cabinets[fixture_name].set_world_pose(
                world_xyz=world_xyz,
                yaw=yaw,
            )
        self.init_robot_base_ref = self.counter

    def _get_obj_cfgs(self) -> list[dict[str, Any]]:
        return []

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._observer_armed = False
        self.close_all_drawers()
        self.reset_semantic_state()

    def _reset_internal(self) -> None:
        self._observer_armed = False
        super()._reset_internal()
        self.close_all_drawers()
        self.sim.forward()
        self._physical_observer.reset(self._physical_sample())
        self._observer_armed = True

    def update_state(self) -> None:
        super().update_state()
        if self._observer_armed:
            self._physical_observer.observe(self._physical_sample())

    def reward(self, action=None):
        del action
        if self._observer_armed:
            self._physical_observer.observe(self._physical_sample())
        return 1.0 if self._check_success() else 0.0

    # Semantic events ----------------------------------------------------

    @property
    def semantic_state(self) -> SemanticState:
        return self._physical_observer.state

    @property
    def physical_observer(self) -> TS04PhysicalObserver:
        return self._physical_observer

    @property
    def checked_fixture_set(self) -> set[str]:
        return self.semantic_state.checked_fixture_set

    @property
    def completed_fixture_history(self) -> list[str]:
        return self.semantic_state.completed_fixture_history

    @property
    def active_fixture(self) -> str | None:
        return self.semantic_state.active_fixture

    @property
    def num_checked(self) -> int:
        return self.semantic_state.num_checked

    def reset_semantic_state(self) -> None:
        try:
            self._physical_observer.reset(self._physical_sample())
            self._observer_armed = True
        except Exception:
            self._physical_observer.reset_state()

    def record_open_event(
        self,
        fixture_name: str,
        *,
        drawer_open: bool,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        result = record_open_attempt(
            self.semantic_state,
            fixture_name,
            drawer_open=drawer_open,
            failure_reason=failure_reason,
        ).as_dict()
        result["diagnostics"] = self.inspection_diagnostics()
        return result

    def record_inspection_event(
        self,
        fixture_name: str,
        *,
        drawer_open: bool,
    ) -> dict[str, Any]:
        result = record_inspection_attempt(
            self.semantic_state,
            fixture_name,
            drawer_open=drawer_open,
        ).as_dict()
        result["diagnostics"] = self.inspection_diagnostics()
        return result

    def record_close_event(
        self,
        fixture_name: str,
        *,
        drawer_closed: bool,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        report = record_close_attempt(
            self.semantic_state,
            fixture_name,
            drawer_closed=drawer_closed,
            failure_reason=failure_reason,
        )
        if report.passed:
            self._physical_observer.completed_fixture_steps.setdefault(
                str(fixture_name),
                int(getattr(self, "timestep", 0)),
            )
        result = report.as_dict()
        result["diagnostics"] = self.inspection_diagnostics()
        return result

    def record_completion_button_event(
        self,
        *,
        button_physically_pressed: bool,
        all_drawers_closed: bool | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        closed = (
            self.all_drawers_closed()
            if all_drawers_closed is None
            else bool(all_drawers_closed)
        )
        report = record_completion_button_attempt(
            self.semantic_state,
            all_drawers_closed=closed,
            button_physically_pressed=button_physically_pressed,
            failure_reason=failure_reason,
        )
        if report.passed:
            self._physical_observer.valid_button_press_steps.append(
                int(getattr(self, "timestep", 0))
            )
        result = report.as_dict()
        result["diagnostics"] = self.inspection_diagnostics()
        return result

    # Physical drawer observations --------------------------------------

    def cabinet_for_slot(self, fixture_name: str) -> SingleTabletopDrawer:
        self._validate_fixture(fixture_name)
        return self.storage_cabinets[str(fixture_name)]

    def slot_open_fraction(self, fixture_name: str) -> float:
        return float(self.cabinet_for_slot(fixture_name).open_fraction(self))

    def _slot_open_fraction(self, fixture_name: str) -> float:
        return self.slot_open_fraction(fixture_name)

    def set_slot_open_fraction(self, fixture_name: str, fraction: float) -> None:
        self._validate_fixture(fixture_name)
        value = float(fraction)
        if not 0.0 <= value <= 1.0:
            raise ValueError("drawer open fraction must be in [0, 1]")
        fixture = self.cabinet_for_slot(fixture_name)
        if value <= self.closed_threshold:
            fixture.close_drawer(env=self, min=value, max=value)
        else:
            fixture.open_drawer(env=self, min=value, max=value)
        self.sim.forward()

    def close_all_drawers(self, *, fraction: float = 0.0) -> None:
        for fixture_name in self.fixture_names:
            self.set_slot_open_fraction(fixture_name, fraction)

    def physical_state(self) -> PhysicalState:
        return PhysicalState(
            {
                fixture_name: self.slot_open_fraction(fixture_name)
                for fixture_name in self.fixture_names
            }
        )

    def all_drawers_closed(self) -> bool:
        return all(
            self.slot_open_fraction(fixture_name) <= self.closed_threshold
            for fixture_name in self.fixture_names
        )

    def completion_button_position(self) -> np.ndarray:
        """Return the live center of the movable green plunger geom."""

        try:
            geom_id = int(
                self.sim.model.geom_name2id(self.COMPLETION_BUTTON_GEOM_NAME)
            )
            return np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
        except Exception:
            return np.asarray(self.completion_button_world_xyz, dtype=float) + np.asarray(
                [0.0, 0.0, 0.075], dtype=float
            )

    def completion_button_press_depth(self) -> float:
        """Return positive plunger travel in metres (0.012 m at the limit)."""

        try:
            return min(
                COMPLETION_BUTTON_MAX_TRAVEL,
                max(
                    0.0,
                    -float(
                        self.sim.data.get_joint_qpos(
                            self.COMPLETION_BUTTON_JOINT_NAME
                        )
                    ),
                ),
            )
        except Exception:
            return 0.0

    def _physical_sample(self) -> PhysicalSample:
        return PhysicalSample(
            step_id=int(getattr(self, "timestep", 0)),
            drawer_open_fraction={
                fixture_name: self.slot_open_fraction(fixture_name)
                for fixture_name in self.fixture_names
            },
            completion_button_press_depth_m=self.completion_button_press_depth(),
        )

    def slot_pose(self, fixture_name: str) -> np.ndarray:
        fixture = self.cabinet_for_slot(fixture_name)
        fixture.update_state(self)
        return fixture.drawer_pose()

    def slot_floor_pose(self, fixture_name: str) -> np.ndarray:
        fixture = self.cabinet_for_slot(fixture_name)
        fixture.update_state(self)
        return fixture.drawer_floor_pose()

    def robot_base_distance_to_slot(self, fixture_name: str) -> float:
        obs = self._get_observations(force_update=True)
        base_xy = np.asarray(obs["robot0_base_pos"], dtype=float).reshape(-1)[:2]
        return float(np.linalg.norm(base_xy - self.slot_pose(fixture_name)[:2]))

    # Task metadata and final result ------------------------------------

    def get_ep_meta(self) -> dict[str, Any]:
        metadata = super().get_ep_meta()
        prompt = self.resolve_prompt_spec()
        metadata["lang"] = prompt.policy_prompt
        metadata["initial_prompt"] = prompt.initial_prompt
        metadata.setdefault("refs", {}).update(
            {
                "axis": AXIS,
                "fixture_names": list(self.fixture_names),
                "fixture_display_names": dict(self.FIXTURE_DISPLAY_NAMES),
                "memory_variables": [
                    "checked_fixture_set",
                    "active_fixture",
                    "completed_fixture_history",
                    "num_checked",
                    "target_found",
                    "completion_button_press_count",
                    "completion_button_pressed",
                ],
                "completion_button_geom": self.COMPLETION_BUTTON_GEOM_NAME,
                "completion_button_joint": self.COMPLETION_BUTTON_JOINT_NAME,
                "target_fixture": self.target_fixture,
                "target_object_name": TARGET_OBJECT_NAME,
                "target_object_geom": self.TARGET_OBJECT_GEOM_NAME,
            }
        )
        from bench.tasks.ts04.domain.contracts import PARENT_TASK_ID, TASK_ID, TASK_REVISION

        task_values = {
            "task_id": TASK_ID,
            "task_revision": TASK_REVISION,
            "parent_task_id": PARENT_TASK_ID,
            "fixture_count": len(self.fixture_names),
        }
        metadata.update(task_values)
        metadata.setdefault("refs", {}).update(task_values)
        return metadata

    def resolve_prompt_spec(self):
        from bench.tasks.ts04.prompts import resolve_prompt_spec

        return resolve_prompt_spec(variant="memory")

    def _check_success(self) -> bool:
        return evaluate_success(
            self.semantic_state,
            self.physical_state(),
            closed_threshold=self.closed_threshold,
        ).success

    def inspection_diagnostics(self) -> dict[str, Any]:
        return diagnostics(
            self.semantic_state,
            self.physical_state(),
            closed_threshold=self.closed_threshold,
            fixture_display_names=self.FIXTURE_DISPLAY_NAMES,
        )

    def task_diagnostics(self) -> dict[str, Any]:
        return self.inspection_diagnostics()

    def _validate_fixture(self, fixture_name: str) -> None:
        if str(fixture_name) not in self.fixture_names:
            raise KeyError(f"unknown {self.TASK_LABEL} fixture {fixture_name!r}")


def _add_completion_button_asset(
    parent: ET.Element,
    *,
    pos: tuple[float, float, float],
    geom_name: str = COMPLETION_BUTTON_GEOM_NAME,
    joint_name: str = COMPLETION_BUTTON_JOINT_NAME,
) -> None:
    """Install the complete TM01-style button under TS04 identifiers."""

    add_ts04_button_asset(
        parent,
        geom_name=geom_name,
        pos=pos,
        joint_name=joint_name,
    )


__all__ = [
    "COMPLETION_BUTTON_GEOM_NAME",
    "COMPLETION_BUTTON_JOINT_NAME",
    "COMPLETION_BUTTON_MAX_TRAVEL",
    "TS04InspectTwoDrawers",
]
