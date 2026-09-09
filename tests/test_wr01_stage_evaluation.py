from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bench.evaluation.stage_integration import get_stage_evaluation_integration
from bench.tasks.wr01.stage_evaluation import build_stage_specs, stage_completed
from scripts.eval.build_wr01_dataset_prefixes import handoff_step_for_subtask


def test_wr01_open_cabinet_replays_the_closed_cabinet_prefix() -> None:
    integration = get_stage_evaluation_integration("WR-01")

    assert integration.target_stage_ids == (
        "close_lower_cabinet",
        "open_cabinet",
    )
    assert integration.prefix_stage_by_target == {
        "close_lower_cabinet": None,
        "open_cabinet": "close_lower_cabinet",
    }
    assert tuple(spec.id for spec in build_stage_specs(target_stage="open_cabinet")) == (
        "open_cabinet",
    )


def test_wr01_stage_completion_uses_authoritative_evaluator_state() -> None:
    state = SimpleNamespace(
        stages={"open_cabinet": SimpleNamespace(qualified=True)}
    )

    assert stage_completed(state, "open_cabinet")


def test_wr01_dataset_handoff_starts_before_retrieval_door_subtask() -> None:
    subtasks = np.asarray([0, 4, 5, 5, 7, 7, 8], dtype=np.int64)

    assert handoff_step_for_subtask(subtasks, 7) == 4
