"""Build TM04's navigation and hybrid manipulation runtime."""

from typing import Any

from bench.execution.runner import StageRunner

from ..execution.manipulation import build_manipulation_assembly
from ..execution.navigation import build_navigation_assembly


def build_operation_handlers(
    descriptor: Any, env: Any, observation: dict[str, Any]
) -> dict[str, Any]:
    navigation = build_navigation_assembly(
        env,
        observation,
        config=descriptor.navigation,
        seed=descriptor.scene.scene_seed,
    )
    manipulation = build_manipulation_assembly(
        env,
        observation,
        settings=descriptor.config.manipulation,
        planner_script=descriptor.resources["planner_script"],
        execute=descriptor.execute,
        missing_resources=tuple(
            descriptor.release_facts.get("missing_physical_calibration", ())
        ),
        resources={
            name: descriptor.resources[name]
            for name in (
                "cabinet_door_pose",
                "can_grasp_pose",
                "can_retrieval_prepose",
                "can_place_pose",
            )
        },
    )
    return {
        "runner": StageRunner(
            navigation=navigation.controller,
            manipulation=manipulation.controller,
        )
    }


__all__ = ["build_operation_handlers"]
