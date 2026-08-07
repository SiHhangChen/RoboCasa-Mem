"""Canonical TM04 expert plan expanded from the twelve-route event graph."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "TM04":
        raise ValueError(f"TM04 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    prefix = str(facts["prefix"])
    tail = str(facts["tail"])
    sugars = tuple(str(name) for name in facts["selected_sugar_names"])
    if len(sugars) != 2 or len(set(sugars)) != 2:
        raise ValueError("TM04 expert plan requires two distinct selected sugars")
    actions = inputs.action_config

    def return_arm(
        target: str,
        event: str,
        *,
        gripper_action: float | None = None,
    ):
        metadata = {**dict(actions.return_arm_metadata), "stage": event}
        if gripper_action is not None:
            metadata["gripper_action"] = float(gripper_action)
        return operate(
            "return_arm", target, name=f"return_arm_after_{event}",
            metadata=metadata,
        )

    def stages_for(event: str):
        if event == "A1":
            return [
                navigate("water_button_station", purpose="cold_on", name="navigate_sink_A1"),
                operate("operate_water_button", "sink", name="press_blue_water_button_on_A1", metadata={**dict(actions.water_button_metadata), "mode": "on"}),
                return_arm("sink", event),
            ]
        if event == "A2":
            return [
                navigate("water_button_station", purpose="cold_off", name="navigate_sink_A2"),
                operate("operate_water_button", "sink", name="press_blue_water_button_off_A2", metadata={**dict(actions.water_button_metadata), "mode": "off"}),
                return_arm("sink", event),
            ]
        if event == "B1":
            return [
                navigate("cabinet_door_station", name="navigate_cabinet_B1"),
                operate("open_tm04_cabinet", "cabinet", name="open_cabinet_B1", metadata=actions.cabinet_open_metadata),
                return_arm("cabinet", "cabinet_open_B1"),
                navigate("can_retrieval_station", name="navigate_can_B1"),
                operate("pick_can_from_cabinet", "can", name="pick_can_B1", metadata=actions.can_pick_metadata),
                navigate("can_place_station", holding="can", name="navigate_can_place_B1"),
                operate("place_can_in_marked_region", "can", name="place_can_B1", metadata=actions.can_place_metadata),
                return_arm("can", event),
            ]
        if event == "B2":
            return [
                navigate("cabinet_door_station", name="navigate_cabinet_B2"),
                operate("close_tm04_cabinet", "cabinet", name="close_cabinet_B2", metadata=actions.cabinet_close_metadata),
                return_arm("cabinet", event),
            ]
        sugar_index = 0 if event == "C1" else 1
        sugar = sugars[sugar_index]
        return [
            navigate("sugar_station", purpose="pick", name=f"navigate_{sugar}_{event}"),
            operate("pick_sugar", sugar, name=f"pick_{sugar}_{event}", metadata=actions.sugar_pick_metadata),
            return_arm(sugar, f"{event}_pick", gripper_action=1.0),
            navigate("saucepan_station", purpose="place", holding=sugar, name=f"navigate_{sugar}_to_saucepan_{event}"),
            operate("place_sugar_in_hot_chocolate", sugar, name=f"place_{sugar}_{event}", metadata={**dict(actions.sugar_delivery_metadata), "sugar_step": sugar_index + 1}),
            return_arm(sugar, event, gripper_action=-1.0),
        ]

    prefix_events = ["A1", *(f"{letter}1" for letter in prefix)]
    tail_events = [f"{letter}2" for letter in tail]
    route = [*prefix_events, *tail_events]
    stages = [stage for event in route for stage in stages_for(event)]
    stages.extend([
        navigate("completion_button_station", name="navigate_red_completion_button"),
        operate("press_completion_button", "completion_button", name="press_red_completion_button", metadata=actions.completion_button_metadata),
    ])
    return finish_plan(
        episode,
        stages,
        metadata={
            "prefix": prefix,
            "tail": tail,
            "route": route,
            "target_sugar_count": 2,
            "total_sugar_count": 3,
            "selected_sugar_names": list(sugars),
        },
    )


__all__ = ["build_expert_plan"]
