"""Task-local WR-02 expert plan built from immutable episode facts."""

from __future__ import annotations

from bench.execution.contracts import ExpertPlan
from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs) -> ExpertPlan:
    episode = inputs.episode
    if episode.task_id != "WR-02":
        raise ValueError(f"WR-02 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    home = dict(facts["home_assignment"])
    use = dict(facts["use_assignment"])
    sampled = dict(facts.get("return_sampled_offsets_xy", {}))
    stages: list[object] = [navigate("shelf_lower", purpose="shelf_observe", name="observe_condiment_shelf",
                                      metadata={"observe": "shelf_lower_upper_binding"})]

    def ret(name: str, stage: str, gripper: float) -> object:
        return operate("return_arm", name, name=stage,
                       metadata={**dict(actions.return_arm_holding if gripper > 0 else actions.return_arm_empty),
                                 "stage": stage, "gripper_action": gripper})

    for condiment in ("ketchup", "mustard"):
        level, sink = str(home[condiment]), str(use[condiment])
        place = {
            **dict(actions.sink_place_by_region[sink]),
            "target_name": f"{condiment}_{sink}",
        }
        stages.extend([
            navigate(level, purpose="shelf_pick", name=f"navigate_{condiment}_home"),
            operate("pick", condiment, name=f"pick_{condiment}", metadata=actions.shelf_pick_by_condiment[condiment]),
            ret(condiment, f"return_after_{condiment}_shelf_pick", 1.0),
        ])
        if level == "shelf_upper":
            stages.append(operate("set_torso_height", condiment, name=f"lower_torso_after_{condiment}_upper_shelf_pick",
                                  metadata=actions.lower_torso_after_upper_pick))
        stages.extend([
            navigate("sink_place_station", holding=condiment, name=f"navigate_{condiment}_sink_place"),
            operate("place_at_accessible", condiment, name=f"place_{condiment}_at_sink_side",
                    metadata={**place, "target_pos": f"{condiment}:{sink}"}),
            ret(condiment, f"return_after_{condiment}_sink_place", -1.0),
        ])
    for condiment in ("ketchup", "mustard"):
        level = str(home[condiment])
        offset = sampled.get(condiment)
        place = {
            **dict(actions.home_place_by_condiment[condiment]),
            "target_name": f"{condiment}_return_shelf_region",
            "return_sampled_xy_offset": offset,
        }
        stages.extend([
            navigate("sink_place_station", name=f"navigate_{condiment}_sink_pick"),
            operate("pick", condiment, name=f"pick_{condiment}_from_sink", metadata=actions.sink_pick_by_condiment[condiment]),
            ret(condiment, f"stow_{condiment}_for_return_transport", 1.0),
            navigate(level, purpose="home_return", holding=condiment, name=f"navigate_{condiment}_home_return",
                     metadata=actions.return_navigation),
            operate("place_at_accessible", condiment, name=f"return_{condiment}_home",
                    metadata={**place, "target_pos": f"{condiment}:{level}"}),
            ret(condiment, f"return_after_{condiment}_home_place", -1.0),
        ])
    return finish_plan(episode, stages, metadata={"home_assignment": home, "use_assignment": use, "return_sampled_offsets_xy": sampled})


__all__ = ["build_expert_plan"]
