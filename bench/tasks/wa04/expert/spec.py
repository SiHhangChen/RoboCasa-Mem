"""Task-local WA-04 expert plan built from resolved ExpertPlanInputs."""

from __future__ import annotations

from bench.execution.contracts import ExpertPlan
from bench.execution.plan_builder import finish_plan, navigate, operate
from bench.tasks.contracts import ExpertPlanInputs


def build_expert_plan(inputs: ExpertPlanInputs) -> ExpertPlan:
    episode = inputs.episode
    if episode.task_id != "WA-04":
        raise ValueError(f"WA-04 cannot build a plan for {episode.task_id!r}")
    facts = dict(episode.facts)
    actions = inputs.action_config
    temperature_by_cup = {
        str(cup): str(temperature)
        for cup, temperature in dict(facts["temperature_by_cup"]).items()
    }
    order = tuple(str(v) for v in facts["fill_cup_order"])
    slot_by_cup = dict(facts["service_slot_by_cup"])
    stages: list[object] = []

    def ret(target: str, name: str, gripper: float = -1.0) -> object:
        return operate("return_arm", target, name=name,
                       metadata={**dict(actions.return_arm_metadata), "stage": name,
                                 "gripper_action": gripper})

    for ordinal, cup in enumerate(order, start=1):
        temperature = temperature_by_cup[cup]
        label = f"cup_{ordinal}"
        stages.extend([
            navigate("water_button_station", purpose=f"{temperature}_on",
                     name=f"navigate_{label}_{temperature}_button_on"),
            operate("operate_water_button", "sink", name=f"press_{label}_{temperature}_button_on",
                    metadata={**dict(actions.faucet_metadata), "mode": "on", "temperature": temperature}),
            ret("sink", f"return_after_{label}_{temperature}_button_on"),
            navigate(cup, purpose="pick", name=f"navigate_pick_{label}_{cup}",
                     metadata={
                         "xy_tolerance": float(actions.cup_pick_navigation_xy_tolerance),
                         "yaw_tolerance": float(actions.cup_pick_navigation_yaw_tolerance),
                     }),
            operate("pick_upright_cup", cup, name=f"pick_{label}_{cup}", metadata=actions.cup_pick_metadata),
            ret(cup, f"return_to_regular_after_pick_{label}_{cup}", 1.0),
            navigate("sink_basin", purpose="fill", holding=cup,
                     name=f"navigate_fill_{label}_{temperature}_{cup}",
                     metadata={"xy_tolerance": float(actions.held_fill_navigation_xy_tolerance)}),
            operate("fill_held_under_faucet", cup, name=f"physically_fill_{label}_{temperature}_{cup}",
                    metadata=actions.held_fill_metadata_by_temperature[temperature]),
            operate("return_arm_holding_cup", cup,
                    name=f"return_after_fill_{label}_{temperature}_{cup}",
                    metadata={**dict(actions.return_arm_holding_cup_metadata), "stage": f"return_after_fill_{label}_{temperature}_{cup}", "gripper_action": 1.0}),
            operate("lift_held_cup_after_sink_exit", cup,
                    name=f"lift_after_sink_exit_{label}_{temperature}_{cup}",
                    metadata={**dict(actions.sink_exit_lift_metadata), "stage": f"lift_after_sink_exit_{label}_{temperature}_{cup}"}),
            navigate("service_area", purpose=str(slot_by_cup[cup]), holding=cup,
                     name=f"navigate_service_{label}_{cup}_{slot_by_cup[cup]}"),
            operate("place_on_service_slot", cup, name=f"place_{label}_{cup}_{slot_by_cup[cup]}",
                    metadata=actions.service_place_metadata_by_slot[slot_by_cup[cup]]),
            operate("record_verified_staging", cup, name=f"verify_and_record_{label}_{cup}_{slot_by_cup[cup]}",
                    metadata=actions.service_place_metadata_by_slot[slot_by_cup[cup]]),
            ret(cup, f"return_after_service_{label}_{cup}"),
        ])
    last = temperature_by_cup[order[-1]]
    requested = str(facts["requested_cup_id"])
    stages.extend([
        navigate("water_button_station", purpose=f"{last}_off", name=f"navigate_final_{last}_button_off"),
        operate("operate_water_button", "sink", name=f"press_final_{last}_button_off",
                metadata={**dict(actions.faucet_metadata), "mode": "off", "temperature": last}),
        ret("sink", f"return_after_final_{last}_button_off"),
        operate("verify_cups_staged", "cup_triplet", name="verify_all_three_cups_staged_before_final_pick"),
        navigate("service_area", purpose=str(slot_by_cup[requested]), name=f"navigate_final_pick_{requested}"),
        operate("pick_upright_cup", requested, name=f"physical_final_pick_{requested}", metadata=actions.final_pick_metadata),
        operate("record_final_take", requested, name=f"record_final_take_{requested}"),
    ])
    return finish_plan(
        episode,
        stages,
        metadata={
            "temperature_by_cup": temperature_by_cup,
            "fill_cup_order": list(order),
            "requested_cup_id": requested,
            "service_slot_by_cup": slot_by_cup,
        },
    )


__all__ = ["build_expert_plan"]
