"""Canonical TM03 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "TM03":
        raise ValueError(f"TM03 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    prefix = getattr(facts["prefix"], "value", str(facts["prefix"]))
    tail = getattr(facts["tail"], "value", str(facts["tail"]))
    drawers = tuple(str(value) for value in facts["drawer_order"])
    actions = inputs.action_config

    def appliance(event: str):
        if event == "A1":
            return [navigate("stove_station", name="navigate_stove_A1"), operate("turn_native_stove_knob_on", "front_right", name="turn_native_stove_knob_on", metadata=actions.knob_on_metadata), operate("return_arm", "stove", name="return_arm_after_A1")]
        if event == "A2":
            return [navigate("stove_station", name="navigate_stove_A2"), operate("turn_native_stove_knob_off", "front_right", name="turn_native_stove_knob_off", metadata=actions.knob_off_metadata), operate("return_arm", "stove", name="return_arm_after_A2")]
        if event in {"B1", "B2"}:
            index = 0 if event == "B1" else 1
            drawer = drawers[index]
            return [
                navigate(f"{drawer}_station", name=f"navigate_{drawer}_{event}"),
                operate("open_drawer_physically", drawer, name=f"open_{drawer}_{event}", metadata=actions.drawer_open_metadata),
                operate("inspect_drawer", drawer, name=f"inspect_{drawer}_{event}", metadata=actions.inspect_metadata),
                operate("close_drawer_physically", drawer, name=f"close_{drawer}_{event}", metadata=actions.drawer_close_metadata),
                operate("retreat_and_return_arm_after_drawer", drawer, name=f"return_arm_after_{event}"),
            ]
        turn_on = event == "C1"
        operation = "turn_native_toaster_oven_time_knob_on" if turn_on else "turn_native_toaster_oven_time_knob_off"
        return [
            navigate("toaster_oven_station", name=f"navigate_toaster_oven_{event}"),
            operate(operation, "time", name=f"{operation}_{event}", metadata=(actions.toaster_oven_start_metadata if turn_on else actions.toaster_oven_stop_metadata)),
            operate("return_arm", "toaster_oven", name=f"return_arm_after_{event}"),
        ]

    prefix_events = ["A1", "C1", "B1"] if prefix == "CB" else ["A1", "B1", "C1"]
    tail_events = [f"{letter}2" for letter in tail]
    route = [*prefix_events, *tail_events]
    stages = [stage for event in route for stage in appliance(event)]
    drawer_side = route[-1] == "B2"
    side = "drawer_side" if drawer_side else "stove_side"
    stages.extend([
        navigate(f"{side}_completion_station", name=f"navigate_{side}_completion_button"),
        operate("press_completion_button", f"completion_button_{side}", name=f"press_{side}_completion_button", metadata=actions.completion_button_metadata),
    ])
    return finish_plan(episode, stages, metadata={"prefix": prefix, "tail": tail, "drawer_order": list(drawers)})


__all__ = ["build_expert_plan"]
