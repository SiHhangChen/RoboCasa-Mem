"""WR-08 pick safeguards around the shared WR-07 manipulation sequence."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np

from bench.execution.manipulation import ManipulationController
from bench.execution.manipulation.planning_bridge import PlannedStageExecutor
from bench.execution.planning import matrix_from_base_obs, matrix_from_json_pose, pose_from_matrix
from bench.execution.planning.sim_state import preserved_integration_state


def compiled_shelf_contact_geoms(env: Any) -> set[str]:
    """Include procedural shelves absent from the fixture's cached geom list."""
    model = env.sim.model
    roots = {
        int(model.body_name2id(fixture.root_body))
        for name, fixture in env.fixtures.items()
        if "opencabinet" in f"{name} {type(fixture).__name__}".lower()
    }
    result = set()
    for geom_id in range(model.ngeom):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        body = int(model.geom_bodyid[geom_id])
        while body and body not in roots:
            body = int(model.body_parentid[body])
        if body in roots:
            name = model.geom_id2name(geom_id)
            if name is not None:
                result.add(str(name))
    return result


class WorldAnchoredPickExecutor(PlannedStageExecutor):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._pick_base_world: np.ndarray | None = None
        self._pregrasp_plans: dict[tuple[float, ...], dict[str, Any]] = {}

    @contextmanager
    def pick_frame(self, obs: dict[str, Any]):
        previous_frame, previous_plans = self._pick_base_world, self._pregrasp_plans
        self._pick_base_world = matrix_from_base_obs(obs)
        self._pregrasp_plans = {}
        try:
            yield
        finally:
            self._pick_base_world, self._pregrasp_plans = previous_frame, previous_plans

    def run(self, target_pose_base: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if self._pick_base_world is None:
            return super().run(target_pose_base, **kwargs)
        execute = kwargs.get("execute")
        execute = self.config.execute if execute is None else bool(execute)
        stage = kwargs["stage"]
        original_target = matrix_from_json_pose(target_pose_base)
        key = tuple(original_target.reshape(-1))
        if stage == "pregrasp":
            if execute and key in self._pregrasp_plans:
                kwargs["prepared_plan"] = self._pregrasp_plans[key]
            elif not execute:
                kwargs["capture_prepared_plan"] = True
        world_target = None
        if execute and stage in {"grasp", "lift"}:
            self.obs = self.env._get_observations(force_update=True)
            world_target = self._pick_base_world @ original_target
            target_pose_base = pose_from_matrix(
                np.linalg.inv(matrix_from_base_obs(self.obs)) @ world_target
            )
        result = super().run(target_pose_base, **kwargs)
        if stage == "pregrasp" and not execute and result.get("pass"):
            self._pregrasp_plans[key] = result.pop("prepared_plan")
        if world_target is not None:
            result["pick_target_world"] = pose_from_matrix(world_target)
            result["pick_target_frame"] = "world_anchored"
        return result

    def _collision_monitor(self, holding_object, *, collision_options=None):
        if self._pick_base_world is None:
            return super()._collision_monitor(holding_object, collision_options=collision_options)
        with self._virtual_pick_start():
            monitor = super()._collision_monitor(holding_object, collision_options={
                **dict(collision_options or {}),
                "validate_collisions": True,
                "fixture_keywords": "opencabinet",
            })
        shelves = compiled_shelf_contact_geoms(self.env)
        for key in ("obstacle_geoms", "fixture_obstacle_geoms", "held_obstacle_geoms"):
            monitor[key] = sorted(set(monitor[key]) | shelves)
        monitor["compiled_shelf_geoms"] = sorted(shelves)
        return monitor

    @contextmanager
    def _virtual_pick_start(self):
        if self._pick_base_world is None or self._virtual_arm_qpos is None:
            yield
            return
        # Check each dry-run segment from the preceding segment's endpoint.
        right = self.env.robots[0].composite_controller.part_controllers["right"]
        with preserved_integration_state(self.env.sim):
            self.env.sim.data.qpos[right.qpos_index] = self._virtual_arm_qpos
            self.env.sim.forward()
            yield

    def _preflight_if_enabled(self, trajectory, monitor, summary):
        with self._virtual_pick_start():
            result = super()._preflight_if_enabled(trajectory, monitor, summary)
            if (self._pick_base_world is None or not monitor or
                    not monitor.get("enabled") or not result or result.get("abort")):
                return result
            clearance = preflight_shelf_clearance(
                self.env, trajectory, monitor=monitor,
                max_joint_step=self.config.max_joint_step, clearance_m=0.01,
            )
            result["shelf_clearance"] = clearance
            if not clearance["pass"]:
                result.update(collision=True, abort=True,
                              reason="insufficient_shelf_clearance")
            return result


def preflight_shelf_clearance(env, trajectory, *, monitor, max_joint_step, clearance_m):
    """Read geometric distances along the virtual arm path with state restored.

    The margin reserves room for tracking error. It never changes obstacle
    geometry or permits contacts during actual execution.
    """
    import mujoco

    if not np.isfinite(max_joint_step) or max_joint_step <= 0:
        raise ValueError("max_joint_step must be finite and positive")
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("clearance_m must be finite and positive")
    model, data = env.sim.model, env.sim.data
    native_model, native_data = model._model, data._data
    shelves = [model.geom_name2id(name) for name in monitor.get("compiled_shelf_geoms", ())]
    robot_geoms = [model.geom_name2id(name) for name in monitor.get("robot_geoms", ())]
    if not shelves or not robot_geoms:
        return {"pass": False, "required_m": clearance_m, "checked_steps": 0,
                "reason": "missing_shelf_or_robot_collision_geoms"}
    if any(index < 0 or index >= model.ngeom for index in (*shelves, *robot_geoms)):
        raise ValueError("shelf clearance requires valid collision geom IDs")
    right = env.robots[0].composite_controller.part_controllers["right"]
    current = np.array(data.qpos[right.qpos_index], dtype=float)
    trajectory = np.asarray(trajectory, dtype=float)
    if (trajectory.ndim != 2 or trajectory.shape[0] == 0
            or trajectory.shape[1:] != current.shape or not np.all(np.isfinite(trajectory))):
        raise ValueError("shelf clearance requires a nonempty finite arm trajectory")
    checked, min_distance = 0, float(clearance_m)
    with preserved_integration_state(env.sim):
        for waypoint_index, target in enumerate(trajectory):
            target = np.asarray(target, dtype=float)
            delta = target - current
            subdivisions = max(1, int(np.ceil(np.max(np.abs(delta)) / max_joint_step)))
            for substep in range(1, subdivisions + 1):
                data.qpos[right.qpos_index] = current + delta * substep / subdivisions
                env.sim.forward()
                checked += 1
                for shelf in shelves:
                    for robot_geom in robot_geoms:
                        distance = float(mujoco.mj_geomDistance(
                            native_model, native_data, shelf, robot_geom,
                            float(clearance_m), None,
                        ))
                        if not np.isfinite(distance):
                            return {"pass": False, "required_m": clearance_m,
                                    "checked_steps": checked,
                                    "reason": "nonfinite_shelf_distance"}
                        min_distance = min(min_distance, distance)
                        if distance < clearance_m - 1e-9:
                            return {
                                "pass": False, "required_m": clearance_m,
                                "distance_m": distance, "checked_steps": checked,
                                "waypoint_index": waypoint_index, "substep_index": substep,
                                "geom1": model.geom_id2name(shelf),
                                "geom2": model.geom_id2name(robot_geom),
                            }
            current = target
    return {"pass": True, "required_m": clearance_m,
            "min_distance_at_least_m": min_distance, "checked_steps": checked}


class WR08ManipulationController(ManipulationController):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        previous = self.stage_executor
        self.stage_executor = WorldAnchoredPickExecutor(
            self.env, self.obs, driver=self.driver, planner=previous.planner,
            runtime_config=self.runtime_config, plan_template=previous.plan_template,
            model_candidates=previous.model_candidates,
        )

    def pick(self, object_name: str, **kwargs: Any) -> dict[str, Any]:
        with self.stage_executor.pick_frame(self.obs):
            return super().pick(object_name, **kwargs)


__all__ = ["WR08ManipulationController"]
