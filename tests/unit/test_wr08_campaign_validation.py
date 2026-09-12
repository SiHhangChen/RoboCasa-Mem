import copy

import h5py
import numpy as np
import pytest

from bench.evaluation.summary import aggregate_episode_results
from scripts.analysis.validate_wr08_campaign import PUBLIC_STAGES, replay_evidence_errors


@pytest.fixture
def evidence(tmp_path):
    actions = np.zeros((3, 13), dtype=np.float32)
    with h5py.File(tmp_path / "trajectory.h5", "w") as f:
        f.create_dataset("data/demo_1/actions", data=actions)
    native = {"episode_dir": str(tmp_path), "steps": 3, "episode_identity": "test"}
    result = {
        "task_id": "WR-08", "seed": 0, "status": "success", "success": True,
        "env_steps": 3, "policy_actions_consumed": 3,
        "episode_metadata": {"replay": {"episode_identity_matches": True,
                                           "expected_episode_identity": "test",
                                           "current_episode_identity": "test"}},
        "report_stages": {"profile_id": "WR-08/report-v3", "order": list(PUBLIC_STAGES),
                          "stages": {k: {"passed": True, "applicable": True} for k in PUBLIC_STAGES}},
    }
    return result, native, actions


def test_exact_native_replay_is_accepted(evidence):
    assert replay_evidence_errors(*evidence) == []


def _successful_prefix(evidence):
    result, native, actions = copy.deepcopy(evidence)
    result.update(env_steps=2, policy_actions_consumed=2, success_step=2)
    result["diagnostics"] = {
        "success": True, "terminal": True, "stop_execution": True, "failure_code": None,
        "evaluation": {"mode": "policy_physical", "terminal_step": 2},
    }
    return result, native, actions[:2]


def test_verified_task_stop_accepts_the_exact_recorded_action_prefix(evidence):
    assert replay_evidence_errors(*_successful_prefix(evidence)) == []


@pytest.mark.parametrize("fault", ("not_stopped", "wrong_step", "wrong_actions", "expert_mode", "failure", "old_profile"))
def test_short_replay_requires_matching_actions_and_verified_policy_termination(evidence, fault):
    result, native, actions = _successful_prefix(evidence)
    if fault == "not_stopped":
        result["diagnostics"]["stop_execution"] = False
    elif fault == "wrong_step":
        result["diagnostics"]["evaluation"]["terminal_step"] = 1
    elif fault == "wrong_actions":
        actions[0, 0] = 0.5
    elif fault == "expert_mode":
        result["diagnostics"]["evaluation"]["mode"] = "expert_recording"
    elif fault == "failure":
        result["diagnostics"]["failure_code"] = "wrong_target_placement"
    else:
        result["report_stages"]["profile_id"] = "WR-08/report-v2"
    assert replay_evidence_errors(result, native, actions)


@pytest.mark.parametrize("case", ["identity", "short_actions", "changed_actions", "unverified_identity",
                                  "missing_stage", "inapplicable_stage", "internal_profile", "failure"])
def test_incomplete_or_mismatched_replay_cannot_qualify(evidence, case):
    result, native, actions = copy.deepcopy(evidence)
    if case == "identity":
        result["episode_metadata"]["replay"]["current_episode_identity"] = "different"
    elif case == "short_actions":
        actions = actions[:2]
        result["env_steps"] = 2
    elif case == "changed_actions":
        actions[1, 0] = 0.5
    elif case == "unverified_identity":
        result["episode_metadata"]["replay"]["episode_identity_matches"] = None
    elif case == "missing_stage":
        del result["report_stages"]["stages"][PUBLIC_STAGES[-1]]
    elif case == "inapplicable_stage":
        result["report_stages"]["stages"][PUBLIC_STAGES[0]]["applicable"] = False
    elif case == "internal_profile":
        result["report_stages"]["order"] = list(range(17))
    else:
        result["status"] = "failure"
        result["success"] = False
    assert replay_evidence_errors(result, native, actions)


def test_success_rate_does_not_replace_seed_coverage():
    summary = aggregate_episode_results([{"seed": 0, "status": "success"}], expected_seeds=[0, 1])
    assert summary["success_rate"] == 1.0
    assert summary["pending"] == 1
    assert not summary["publishable"]
