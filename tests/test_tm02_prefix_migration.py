from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench.evaluation.expert_prefix import (
    ExpertPrefixArtifact,
    load_expert_prefix,
    write_expert_prefix,
)
from bench.evaluation.stage_suite import load_stage_eval_suite
from bench.tasks.tm02.evaluation import STAGE_ORDER
from bench.tasks.tm02.prefix_migration import upgrade_v1_prefix
from bench.tasks.tm02.stage_evaluation import DEFAULT_STAGE_BUDGETS, LEGACY_STAGE_ORDER
import scripts.eval.upgrade_tm02_stage_prefixes as migration_cli


ROOT = Path(__file__).resolve().parents[1]


def _artifact(a2_first: bool = True, seed: int = 1) -> ExpertPrefixArtifact:
    events = {"A1": 2, "B1": 5, "A2": 8 if a2_first else 16,
              "B2": 16 if a2_first else 8, "completion_button": 20}
    return ExpertPrefixArtifact(
        task_id="TM02", seed=seed, episode_identity="episode", expert_plan_version="expert/v2",
        actions=np.zeros((22, 13), dtype=np.float32), success=True,
        stage_order=LEGACY_STAGE_ORDER,
        stage_boundary_steps=dict(zip(LEGACY_STAGE_ORDER, (2, 5, 16, 20))),
        diagnostics={
            "success": True, "domain_success": True, "absorbing_failure": False,
            "faucet_on_completed": True, "pot_filled_and_can_retrieved": True,
            "fill_pot_thread_completed": True, "retrieve_can_thread_completed": True,
            "completion_button_pressed": True, "failure_code": None,
            "evaluation": {"semantic_event_steps": events},
            "semantic_event_steps": dict(events),
            "physical": {"can_in_place_region": True},
        },
        provenance={"git_revision": "original", "stage_suite_version": "v1"},
    )


@pytest.mark.parametrize("a2_first", (True, False))
def test_upgrade_keeps_actions_and_existing_boundaries(a2_first: bool) -> None:
    old = _artifact(a2_first)
    original = deepcopy(old.manifest())
    new = upgrade_v1_prefix(old)
    assert old.manifest() == original
    np.testing.assert_array_equal(old.actions, new.actions)
    assert new.manifest()["action_sha256"] == original["action_sha256"]
    assert new.stage_order == STAGE_ORDER
    assert new.stage_boundary_steps["one_thread_completed"] == 8
    for name, step in old.stage_boundary_steps.items():
        assert new.stage_boundary_steps[name] == step
    assert new.diagnostics["stage_qualified_steps"]["one_thread_completed"] == 8
    assert new.diagnostics["thread_completion"]["completed_count"] == 2
    assert new.diagnostics["physical"] == old.diagnostics["physical"]
    assert new.provenance["git_revision"] == "original"
    assert new.provenance["stage_profile_migration"]["physical_replay_performed"] is False


@pytest.mark.parametrize("problem", (
    "failure", "wrong_task", "missing_event", "simultaneous", "wrong_boundary",
    "conflicting_events", "missing_fact", "absorbing_failure", "boolean_step",
))
def test_upgrade_rejects_unreliable_or_incompatible_evidence(problem: str) -> None:
    old = _artifact()
    details = deepcopy(old.diagnostics)
    if problem == "failure":
        old = replace(old, success=False)
    elif problem == "wrong_task":
        old = replace(old, task_id="TM01")
    elif problem == "wrong_boundary":
        old = replace(old, stage_boundary_steps={
            **old.stage_boundary_steps, "both_threads_completed": 17,
        })
    elif problem == "missing_event":
        del details["evaluation"]["semantic_event_steps"]["B2"]
    elif problem == "simultaneous":
        details["evaluation"]["semantic_event_steps"]["B2"] = 8
        details["semantic_event_steps"]["B2"] = 8
    elif problem == "conflicting_events":
        details["semantic_event_steps"]["A2"] = 9
    elif problem == "missing_fact":
        del details["fill_pot_thread_completed"]
    elif problem == "absorbing_failure":
        details["absorbing_failure"] = True
    elif problem == "boolean_step":
        details["evaluation"]["semantic_event_steps"]["A2"] = True
    with pytest.raises(ValueError):
        upgrade_v1_prefix(replace(old, diagnostics=details))


def _cli_suite(tmp_path: Path, monkeypatch, seeds=(1,)):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    suite = SimpleNamespace(
        task_id="TM02", stage_ids=STAGE_ORDER,
        base_suite=SimpleNamespace(seeds=seeds), expert_prefix_root=destination,
        stage_suite_version="v2", content_hash="v2-hash",
    )
    monkeypatch.setattr(migration_cli, "load_stage_eval_suite", lambda _: suite)
    args = ["--source-root", str(source), "--stage-suite", "suite.yaml"]
    return source, destination, args


def test_migration_cli_writes_a_separate_validated_artifact(tmp_path, monkeypatch) -> None:
    source, destination, args = _cli_suite(tmp_path, monkeypatch)
    write_expert_prefix(source, _artifact())
    original = (source / "seed_000001/manifest.json").read_bytes()
    assert migration_cli.main(args) == 0
    assert (source / "seed_000001/manifest.json").read_bytes() == original
    upgraded = load_expert_prefix(destination, 1)
    assert upgraded.stage_order == STAGE_ORDER
    assert upgraded.provenance["source_stage_suite_version"] == "v1"
    assert upgraded.provenance["stage_suite_version"] == "v2"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        migration_cli.main(args)
    with pytest.raises(ValueError, match="must differ"):
        migration_cli.main([*args, "--output-root", str(source)])


def test_migration_preflights_all_sources_before_writing(tmp_path, monkeypatch) -> None:
    source, destination, args = _cli_suite(tmp_path, monkeypatch, seeds=(1, 2))
    write_expert_prefix(source, _artifact(seed=1))
    write_expert_prefix(source, replace(_artifact(seed=2), success=False))
    with pytest.raises(ValueError, match="unsuccessful"):
        migration_cli.main(args)
    assert not destination.exists()


def test_tm02_v2_configs_use_five_stages_and_separate_prefixes() -> None:
    for kind in ("pilot", "formal"):
        suite = load_stage_eval_suite(ROOT / f"configs/eval/stages/tm02_gr00t_r40_{kind}_v2.yaml")
        assert suite.stage_ids == STAGE_ORDER
        assert suite.budgets == DEFAULT_STAGE_BUDGETS
        assert suite.base_suite.seeds == tuple(range(1050, 1100))
        assert suite.expert_prefix_root == ROOT / "outputs/stage_prefixes/tm02_gr00t_r40_v2"


def test_tm02_launcher_defaults_isolate_v2_results_from_old_runs() -> None:
    launcher = (ROOT / "scripts/eval/gr00t/eval_gr00t_tm02.sh").read_text()
    assert 'RUN_ID="${RUN_ID:-tm02-gr00t-${CKPT_STEP}-50seeds-r40-report-v2}"' in launcher
    assert 'STAGE_RUN_ID="${STAGE_RUN_ID:-tm02-gr00t-${CKPT_STEP}-stage-r40-formal-v2}"' in launcher
    assert 'STAGE_SUITE="${STAGE_SUITE:-configs/eval/stages/tm02_gr00t_r40_formal_v2.yaml}"' in launcher
