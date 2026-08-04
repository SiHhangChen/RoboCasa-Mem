"""Build TM01's task-local navigation and manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner
from bench.tasks.tm01.execution.manipulation import build_manipulation_assembly
from bench.tasks.tm01.execution.navigation import build_navigation_assembly


def build_operation_handlers(descriptor: Any, env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    navigation = build_navigation_assembly(
        env, observation, config=descriptor.navigation, seed=descriptor.scene.scene_seed
    )
    manipulation = build_manipulation_assembly(
        env,
        observation,
        settings=dict(descriptor.config.manipulation),
        planner_script=descriptor.resources["planner_script"],
        interaction_poses_file=descriptor.resources["stove_knob_interaction_poses"],
        execute=descriptor.execute,
        missing_resources=tuple(descriptor.release_facts.get("missing_physical_calibration", ())),
    )
    return {"runner": StageRunner(navigation=navigation.controller, manipulation=manipulation.controller)}


__all__ = ["build_operation_handlers"]
