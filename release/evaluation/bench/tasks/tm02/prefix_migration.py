"""Explicit, action-preserving migration of TM02's four-stage expert prefixes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from bench.evaluation.expert_prefix import ExpertPrefixArtifact
from bench.evaluation.report_profiles import TM02_REPORT_PROFILE_V1
from bench.tasks.tm02.evaluation import (
    REPORT_STAGE_PROFILE_ID,
    STAGE_ORDER,
    TM02EvaluationSnapshot,
    TM02Evaluator,
)


def upgrade_v1_prefix(artifact: ExpertPrefixArtifact) -> ExpertPrefixArtifact:
    """Derive S3 only from complete, validated historical semantic evidence.

    This does not replay or modify actions. The original action hash and the
    derivation method are retained as provenance; physical replay is a separate
    validation step. Incomplete or differently indexed evidence is rejected.
    """

    if artifact.task_id != "TM02" or artifact.stage_order != TM02_REPORT_PROFILE_V1.stage_ids:
        raise ValueError("migration requires a TM02 four-stage v1 prefix")
    if not artifact.success or artifact.failure_reason is not None:
        raise ValueError("cannot migrate an unsuccessful expert prefix")
    details = artifact.diagnostics
    required_facts = (
        "success", "domain_success", "faucet_on_completed",
        "pot_filled_and_can_retrieved", "fill_pot_thread_completed",
        "retrieve_can_thread_completed", "completion_button_pressed",
    )
    if any(details.get(name) is not True for name in required_facts):
        raise ValueError("migration requires complete successful task diagnostics")
    if details.get("absorbing_failure") is not False or details.get("failure_code") is not None:
        raise ValueError("cannot migrate absorbing-failure diagnostics")

    evaluation = details.get("evaluation")
    recorded = evaluation.get("semantic_event_steps") if isinstance(evaluation, Mapping) else None
    if not isinstance(recorded, Mapping):
        raise ValueError("prefix has no recorded semantic_event_steps; replay is required")
    events = ("A1", "B1", "A2", "B2", "completion_button")
    steps: dict[str, int] = {}
    for event in events:
        value = recorded.get(event)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 < value <= len(artifact.actions)
        ):
            raise ValueError(f"missing or invalid semantic step for {event}; replay is required")
        steps[event] = value
    if "semantic_event_steps" in details:
        duplicate = details["semantic_event_steps"]
        if not isinstance(duplicate, Mapping) or dict(duplicate) != dict(recorded):
            raise ValueError("conflicting semantic event records")
    first = min(steps["A2"], steps["B2"])
    last = max(steps["A2"], steps["B2"])
    if not steps["A1"] < steps["B1"] < first < last < steps["completion_button"]:
        raise ValueError("semantic events do not follow a legal TM02 route")
    expected = {
        "faucet_on": steps["A1"],
        "pot_filled_and_can_retrieved": steps["B1"],
        "both_threads_completed": last,
        "success": steps["completion_button"],
    }
    if dict(artifact.stage_boundary_steps) != expected:
        raise ValueError("semantic steps do not match the recorded action boundaries")

    evaluator = TM02Evaluator()
    evaluator.reset(TM02EvaluationSnapshot(step_id=0))
    result = evaluator.observe(TM02EvaluationSnapshot(
        step_id=len(artifact.actions), faucet_on_completed=True,
        pot_filled_and_can_retrieved=True, fill_pot_thread_completed=True,
        retrieve_can_thread_completed=True, completion_button_pressed=True,
        domain_success=True, event_steps=steps, details=details,
    ))
    boundaries: dict[str, int] = {}
    for name in STAGE_ORDER:
        step = result.stages[name].qualified_step
        if step is None:
            raise ValueError(f"historical evidence did not qualify stage {name!r}")
        boundaries[name] = step
    return replace(
        artifact,
        stage_order=STAGE_ORDER,
        stage_boundary_steps=boundaries,
        diagnostics=result.to_diagnostics(),
        provenance={
            **artifact.provenance,
            "stage_profile_migration": {
                "from_profile_id": TM02_REPORT_PROFILE_V1.profile_id,
                "to_profile_id": REPORT_STAGE_PROFILE_ID,
                "boundary_source": "recorded_semantic_event_steps",
                "source_action_sha256": artifact.manifest()["action_sha256"],
                "physical_replay_performed": False,
            },
        },
    )


__all__ = ["upgrade_v1_prefix"]
