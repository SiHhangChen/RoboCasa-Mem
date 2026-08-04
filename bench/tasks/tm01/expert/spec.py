"""Canonical TM01 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs):
    episode = inputs.episode
    if episode.task_id != "TM01":
        raise ValueError(f"TM01 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    drawers = tuple(str(value) for value in facts["drawer_order"])
    route = getattr(facts["route"], "value", str(facts["route"]))
    actions = inputs.action_config

    def drawer_unit(drawer: str, suffix: str):
        return [
            navigate(f"{drawer}_station", name=f"navigate_{drawer}_{suffix}"),
            operate("open_drawer_physically", drawer, name=f"open_{drawer}_{suffix}", metadata=actions.drawer_open_metadata),
            operate("inspect_drawer", drawer, name=f"inspect_{drawer}_{suffix}", metadata=actions.inspect_metadata),
            operate("close_drawer_physically", drawer, name=f"close_{drawer}_{suffix}", metadata=actions.drawer_close_metadata),
            operate("retreat_and_return_arm_after_drawer", drawer, name=f"return_arm_after_{suffix}"),
        ]

    a1 = [
        navigate("stove_station", name="navigate_stove_A1"),
        operate("turn_native_stove_knob_on", "front_right", name="turn_native_stove_knob_on", metadata=actions.knob_on_metadata),
        operate("return_arm", "stove", name="return_arm_after_A1"),
    ]
    a2 = [
        navigate("stove_station", name="navigate_stove_A2"),
        operate("turn_native_stove_knob_off", "front_right", name="turn_native_stove_knob_off", metadata=actions.knob_off_metadata),
        operate("return_arm", "stove", name="return_arm_after_A2"),
    ]
    b1 = drawer_unit(drawers[0], "B1")
    b2 = drawer_unit(drawers[1], "B2")
    tail = [*a2, *b2] if route == "L1" else [*b2, *a2]
    drawer_side = route == "L1"
    side = "drawer_side" if drawer_side else "stove_side"
    stages = [
        *a1,
        *b1,
        *tail,
        navigate(f"{side}_completion_station", name=f"navigate_{side}_completion_button"),
        operate(
            "press_completion_button",
            f"completion_button_{side}",
            name=f"press_{side}_completion_button",
            metadata=actions.completion_button_metadata,
        ),
    ]
    return finish_plan(episode, stages, metadata={"route": route, "drawer_order": list(drawers)})


__all__ = ["build_expert_plan"]
