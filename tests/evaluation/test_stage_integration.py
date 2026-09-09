from __future__ import annotations

import numpy as np
import pytest

from bench.evaluation.expert_prefix import ExpertPrefixArtifact
from bench.evaluation.stage_integration import get_stage_evaluation_integration


def test_builtin_stage_integrations_cover_ts02_wa01_and_wa05() -> None:
    ts02 = get_stage_evaluation_integration("TS02")
    wa01 = get_stage_evaluation_integration("WA-01")
    wa05 = get_stage_evaluation_integration("WA-05")

    assert ts02.target_stage_ids[0] == "one_drawer_checked"
    assert wa01.target_stage_ids[0] == "sugar_added"
    assert wa05.target_stage_ids == ("cup_filled", "cup_staged", "success")


def test_wa01_selection_and_success_replay_the_same_ready_prefix() -> None:
    integration = get_stage_evaluation_integration("WA-01")
    artifact = ExpertPrefixArtifact(
        task_id="WA-01",
        seed=1050,
        episode_identity="episode",
        expert_plan_version="expert",
        stage_boundary_steps={
            "sugar": 2,
            "one_ready": 4,
            "ready": 6,
            "selection": 8,
            "correct": 8,
            "success": 8,
        },
        actions=np.arange(9 * 13, dtype=np.float32).reshape(9, 13),
        success=True,
        stage_order=("sugar", "one_ready", "ready", "selection", "correct", "success"),
    )

    selection = integration.prefix_actions(artifact, "selection")
    success = integration.prefix_actions(artifact, "success")
    np.testing.assert_array_equal(selection, artifact.actions[:6])
    np.testing.assert_array_equal(success, artifact.actions[:6])


def test_stage_integration_rejects_mismatched_suite_order() -> None:
    integration = get_stage_evaluation_integration("WA-01")
    with pytest.raises(ValueError, match="must declare"):
        integration.validate_stage_suite(("sugar_added", "success"))


def test_stage_integration_selects_tm02_diagnostic_profile_by_stage_ids() -> None:
    stage_ids = (
        "faucet_on",
        "pot_filled",
        "cabinet_opened",
        "can_removed_from_cabinet",
        "can_released_in_place_region",
    )

    integration = get_stage_evaluation_integration("TM02", stage_ids=stage_ids)

    assert integration.target_stage_ids == stage_ids
