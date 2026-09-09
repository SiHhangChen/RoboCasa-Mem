from __future__ import annotations

from types import SimpleNamespace

from bench.evaluation.stage_integration import get_stage_evaluation_integration
from bench.tasks.ts02.stage_evaluation import (
    BEFORE_TARGET_DRAWER,
    TARGET_DRAWER_RETURNED,
    build_expert_stage_boundaries,
)


def test_ts02_stage_integration_uses_independent_memory_handoffs() -> None:
    integration = get_stage_evaluation_integration("TS02")

    assert integration.target_stage_ids == (
        "one_drawer_checked",
        "target_found",
        "search_stopped_after_target",
        "button_pressed",
        "correct_press_count",
    )
    assert integration.prefix_stage_by_target["target_found"] == BEFORE_TARGET_DRAWER
    assert (
        integration.prefix_stage_by_target["search_stopped_after_target"]
        == BEFORE_TARGET_DRAWER
    )
    assert integration.prefix_stage_by_target["button_pressed"] == TARGET_DRAWER_RETURNED
    assert (
        integration.prefix_stage_by_target["correct_press_count"]
        == TARGET_DRAWER_RETURNED
    )


def test_ts02_first_target_starts_from_initial_state() -> None:
    plan = SimpleNamespace(
        metadata={
            "fixture_order": ["cabinet_slot_1"],
            "target_fixture": "cabinet_slot_1",
        }
    )
    boundaries = build_expert_stage_boundaries(
        None,
        plan,
        {"return_arm_after_cabinet_slot_1": 420},
    )

    assert boundaries == {
        BEFORE_TARGET_DRAWER: 0,
        TARGET_DRAWER_RETURNED: 420,
    }


def test_ts02_later_target_handoff_follows_previous_drawer_return_arm() -> None:
    plan = SimpleNamespace(
        metadata={
            "fixture_order": [
                "cabinet_slot_2",
                "cabinet_slot_0",
                "cabinet_slot_1",
            ],
            "target_fixture": "cabinet_slot_1",
        }
    )
    boundaries = build_expert_stage_boundaries(
        None,
        plan,
        {
            "return_arm_after_cabinet_slot_2": 400,
            "return_arm_after_cabinet_slot_0": 780,
            "return_arm_after_cabinet_slot_1": 1160,
        },
    )

    assert boundaries == {
        BEFORE_TARGET_DRAWER: 780,
        TARGET_DRAWER_RETURNED: 1160,
    }
