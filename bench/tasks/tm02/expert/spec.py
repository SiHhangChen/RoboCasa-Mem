"""Canonical TM02 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "TM02":
        raise ValueError(f"TM02 cannot build a plan for {episode.task_id!r}")
    route = getattr(episode.facts["route"], "value", str(episode.facts["route"]))
    actions = inputs.action_config

    def return_arm(target: str, name: str):
        return operate("return_arm", target, name=name, metadata={**dict(actions.return_arm_metadata), "stage": name})

    a1 = [
        navigate("water_button_station", purpose="cold_on", name="navigate_sink_A1"),
        operate("operate_water_button", "sink", name="press_blue_water_button_on", metadata={**dict(actions.water_button_metadata), "mode": "on", "temperature": "cold"}),
        return_arm("sink", "return_arm_after_A1"),
    ]
    b1 = [
        navigate("cabinet_door_station", name="navigate_cabinet_door_B1"),
        operate("open_tm02_cabinet", "cabinet", name="open_cabinet_door_B1", metadata=actions.cabinet_open_metadata),
        return_arm("cabinet", "return_arm_after_cabinet_open"),
        navigate("can_retrieval_station", name="navigate_can_retrieval_B1"),
        operate("pick_can_from_cabinet", "can", name="pick_can_from_cabinet_B1", metadata=actions.can_pick_metadata),
        navigate("can_place_station", holding="can", name="navigate_can_place_region_B1"),
        operate("place_can_in_marked_region", "can", name="place_and_release_can_B1", metadata=actions.can_place_metadata),
        return_arm("can", "return_arm_after_B1"),
    ]
    a2 = [
        navigate("water_button_station", purpose="cold_off", name="navigate_sink_A2"),
        operate("operate_water_button", "sink", name="press_blue_water_button_off", metadata={**dict(actions.water_button_metadata), "mode": "off", "temperature": "cold"}),
        return_arm("sink", "return_arm_after_A2"),
    ]
    b2 = [
        navigate("cabinet_door_station", name="navigate_cabinet_door_B2"),
        operate("close_tm02_cabinet", "cabinet", name="close_cabinet_door_B2", metadata=actions.cabinet_close_metadata),
        return_arm("cabinet", "return_arm_after_B2"),
    ]
    tail = [*a2, *b2] if route == "L1" else [*b2, *a2]
    stages = [
        *a1, *b1, *tail,
        navigate("completion_button_station", name="navigate_red_completion_button"),
        operate("press_completion_button", "completion_button", name="press_red_completion_button", metadata=actions.completion_button_metadata),
    ]
    return finish_plan(episode, stages, metadata={"route": route})


__all__ = ["build_expert_plan"]
