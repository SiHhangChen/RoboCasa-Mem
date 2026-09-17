from scripts.annotate.planner_to_stage_boundaries_tm02 import expected_stage_contract


def test_cabinet_post_open_stage_accepts_both_recorded_primitives() -> None:
    kind, primitives = expected_stage_contract("return_arm_after_cabinet_open")

    assert kind == "operate"
    assert primitives == {
        "return_arm_to_regular_pose",
        "cabinet_post_open_inward_and_return",
    }
