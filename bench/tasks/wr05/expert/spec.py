"""Task-local WR-05 expert plan built from immutable episode facts."""

from __future__ import annotations

from bench.execution.contracts import ExpertPlan
from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs) -> ExpertPlan:
    episode = inputs.episode
    if episode.task_id != "WR-05":
        raise ValueError(f"WR-05 cannot build a plan for {episode.task_id!r}")
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
        level, region = str(home[condiment]), str(use[condiment])
        plate_station = _plate_station_for_condiment_region(condiment, region)
        place = {
            **dict(actions.plate_place_by_region[region]),
            "target_name": f"{condiment}_{region}",
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
            navigate(plate_station, holding=condiment, name=f"navigate_{condiment}_plate_place",
                     metadata={"use_manual_path": False, "xy_tolerance": 0.015, "yaw_tolerance": 0.02}),
            operate("place_at_accessible", condiment, name=f"place_{condiment}_at_plate_side",
                    metadata={**place, "target_pos": f"{condiment}:{region}"}),
            ret(condiment, f"return_after_{condiment}_plate_place", -1.0),
        ])
    for condiment in ("ketchup", "mustard"):
        level = str(home[condiment])
        region = str(use[condiment])
        offset = sampled.get(condiment)
        place = {
            **dict(actions.home_place_by_condiment[condiment]),
            "target_name": f"{condiment}_return_shelf_region",
            "return_sampled_xy_offset": offset,
        }
        stages.extend([
            navigate(_plate_pick_station_for_region(region), name=f"navigate_{condiment}_plate_pick"),
            operate("pick", condiment, name=f"pick_{condiment}_from_plate", metadata=actions.plate_pick_by_condiment[condiment]),
            ret(condiment, f"stow_{condiment}_for_return_transport", 1.0),
            navigate(level, purpose="home_return", holding=condiment, name=f"navigate_{condiment}_home_return",
                     metadata=actions.return_navigation),
            operate("place_at_accessible", condiment, name=f"return_{condiment}_home",
                    metadata={**place, "target_pos": f"{condiment}:{level}"}),
            ret(condiment, f"return_after_{condiment}_home_place", -1.0),
        ])
    return finish_plan(episode, stages, metadata={"home_assignment": home, "use_assignment": use, "return_sampled_offsets_xy": sampled})


def _plate_station_for_condiment_region(condiment: str, use_region: str) -> str:
    condiment = str(condiment)
    use_region = str(use_region)
    if condiment not in {"ketchup", "mustard"}:
        raise KeyError(f"unknown WR-05 condiment {condiment!r}")
    # The left slot has one calibrated base pose. Both bottles must use it so
    # mustard-left follows exactly the same station geometry as ketchup-left.
    if use_region == "plate_left":
        return "plate_left_station"
    if use_region == "plate_right":
        # Use the same base geometry as the left slot, translated by the
        # plate-slot delta. The legacy right station leaves mustard near the
        # reach boundary, where the guarded descent stops with the bottle
        # tilted and it falls after release.
        return "ketchup_plate_right_station"
    raise KeyError(
        f"unknown WR-05 condiment/plate assignment "
        f"{condiment!r}/{use_region!r}"
    )


def _plate_pick_station_for_region(use_region: str) -> str:
    station_by_region = {
        "plate_left": "plate_left_pick_station",
        "plate_right": "plate_right_pick_station",
    }
    try:
        return station_by_region[str(use_region)]
    except KeyError as exc:
        raise KeyError(f"unknown WR-05 plate region {use_region!r}") from exc


__all__ = ["build_expert_plan"]
