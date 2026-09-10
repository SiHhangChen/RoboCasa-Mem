from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bench.evaluation.stage_integration import get_stage_evaluation_integration
from bench.tasks.wr03.stage_evaluation import build_stage_specs, stage_completed
from scripts.eval.build_wr03_dataset_prefixes import handoff_step_for_subtask


def test_wr03_open_cabinet_replays_the_canonical_reset_prefix() -> None:
    integration = get_stage_evaluation_integration("WR-03")

    assert integration.target_stage_ids == (
        "canonical_reset",
        "open_remembered_cabinet",
    )
    assert integration.prefix_stage_by_target == {
        "canonical_reset": None,
        "open_remembered_cabinet": "canonical_reset",
    }
    assert tuple(
        spec.id
        for spec in build_stage_specs(target_stage="open_remembered_cabinet")
    ) == ("open_remembered_cabinet",)


def test_wr03_stage_completion_uses_authoritative_evaluator_state() -> None:
    state = SimpleNamespace(
        stages={"open_remembered_cabinet": SimpleNamespace(qualified=True)}
    )

    assert stage_completed(state, "open_remembered_cabinet")


def test_wr03_dataset_handoff_starts_before_remembered_door_subtask() -> None:
    subtasks = np.asarray([0, 1, 2, 2, 4, 4, 5], dtype=np.int64)

    assert handoff_step_for_subtask(subtasks, 4) == 4
