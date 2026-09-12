from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.evaluation.stage_integration import get_stage_evaluation_integration
from bench.evaluation.stage_suite import load_stage_eval_suite
from bench.tasks.wr03.stage_evaluation import build_stage_specs, stage_completed
from scripts.eval.build_wr03_dataset_prefixes import BOUNDARY_STAGE, handoff_step_for_subtask


def test_wr03_open_cabinet_replays_the_closed_cabinet_prefix() -> None:
    integration = get_stage_evaluation_integration("WR-03")

    assert integration.target_stage_ids == (
        "close_second_door",
        "open_remembered_cabinet",
    )
    assert integration.prefix_stage_by_target == {
        "close_second_door": None,
        "open_remembered_cabinet": "close_second_door",
    }
    assert tuple(
        spec.id
        for spec in build_stage_specs(target_stage="open_remembered_cabinet")
    ) == ("open_remembered_cabinet",)


def test_wr03_stage_suite_and_dataset_prefix_share_the_closing_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    suite = load_stage_eval_suite(
        root / "configs/eval/stages/wr03_pi05_59999_open_cabinet_id5_v1.yaml"
    )
    integration = get_stage_evaluation_integration(suite.task_id, stage_ids=suite.stage_ids)
    integration.validate_stage_suite(suite.stage_ids)

    assert suite.targets[0].source == "main_rollout"
    assert suite.targets[1].source == "expert_prefix"
    assert suite.expert_prefix_root.name == "wr03_dataset_id5_v2"
    assert integration.prefix_stage_by_target["open_remembered_cabinet"] == BOUNDARY_STAGE
    assert BOUNDARY_STAGE == "close_second_door"
    assert integration.build_stage_specs(suite.budgets, "open_remembered_cabinet")[0].id == (
        "open_remembered_cabinet"
    )


def test_wr03_neutral_reset_is_not_an_evaluation_target() -> None:
    with pytest.raises(KeyError, match="unknown WR-03 target stage"):
        build_stage_specs(target_stage="canonical_reset")


def test_wr03_stage_completion_uses_authoritative_evaluator_state() -> None:
    state = SimpleNamespace(
        stages={"open_remembered_cabinet": SimpleNamespace(qualified=True)}
    )

    assert stage_completed(state, "open_remembered_cabinet")


def test_wr03_dataset_handoff_starts_before_remembered_door_subtask() -> None:
    subtasks = np.asarray([0, 1, 2, 2, 4, 4, 5], dtype=np.int64)

    assert handoff_step_for_subtask(subtasks, 4) == 4
