"""Contract tests for TS03's independent ordered evaluator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from bench.tasks.ts03.domain.contracts import PROFILE_IDS, profile_for_id
from bench.tasks.ts03.evaluation import (
    REPORT_STAGE_ORDER,
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    TS03EvaluationSnapshot,
    TS03Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    profile_id: str,
    count: int = 0,
    objects_released: bool = True,
    button_pressed: bool = False,
) -> TS03EvaluationSnapshot:
    profile = profile_for_id(profile_id)
    return TS03EvaluationSnapshot(
        step_id=step,
        profile_id=profile.profile_id,
        target_sugar_count=profile.target_sugar_count,
        total_sugar_count=profile.total_sugar_count,
        confirmed_sugar_count=count,
        objects_released=objects_released,
        completion_button_pressed=button_pressed,
    )


def _evaluator(profile_id: str) -> TS03Evaluator:
    profile = profile_for_id(profile_id)
    evaluator = TS03Evaluator(
        profile_id=profile.profile_id,
        target_sugar_count=profile.target_sugar_count,
        total_sugar_count=profile.total_sugar_count,
    )
    evaluator.reset(_snapshot(0, profile_id=profile_id))
    return evaluator


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_each_profile_uses_its_seed_resolved_target(profile_id: str) -> None:
    profile = profile_for_id(profile_id)
    evaluator = _evaluator(profile_id)

    result = evaluator.result()
    for count in range(1, profile.target_sugar_count + 1):
        result = evaluator.observe(
            _snapshot(count, profile_id=profile_id, count=count)
        )

    assert result.stages["sugar_1"].qualified_step == 1
    assert result.stages["sugar_2"].applicable is (profile.target_sugar_count >= 2)
    assert result.stages["sugar_2"].qualified_step == (
        2 if profile.target_sugar_count >= 2 else None
    )
    assert result.stages["sugar_3"].applicable is (profile.target_sugar_count == 3)
    assert result.stages["sugar_3"].qualified_step == (
        3 if profile.target_sugar_count == 3 else None
    )

    result = evaluator.observe(
        _snapshot(
            profile.target_sugar_count + 1,
            profile_id=profile_id,
            count=profile.target_sugar_count,
            button_pressed=True,
        )
    )
    diagnostics = result.to_diagnostics()

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert tuple(diagnostics["stages"]) == STAGE_ORDER
    assert diagnostics["profile_id"] == profile.profile_id
    assert diagnostics["hierarchy_tier"] == profile.hierarchy_tier
    assert diagnostics["target_sugar_count"] == profile.target_sugar_count
    assert diagnostics["total_sugar_count"] == profile.total_sugar_count


@pytest.mark.parametrize("profile_id", ("k1_n1", "k1_n2", "k1_n3"))
def test_k1_profiles_pad_unused_sugar_stages_after_sugar_1(profile_id: str) -> None:
    evaluator = _evaluator(profile_id)

    initial = evaluator.result().to_diagnostics()["stages"]
    assert not initial["sugar_2"]["effective_success"]
    assert not initial["sugar_3"]["effective_success"]

    result = evaluator.observe(_snapshot(1, profile_id=profile_id, count=1))
    stages = result.to_diagnostics()["stages"]

    assert stages["sugar_1"]["applicable"]
    assert stages["sugar_1"]["qualified"]
    assert stages["sugar_1"]["completion_mode"] == "physical"
    for name in ("sugar_2", "sugar_3"):
        assert not stages[name]["applicable"]
        assert not stages[name]["qualified"]
        assert stages[name]["qualified_step"] is None
        assert stages[name]["effective_success"]
        assert stages[name]["completion_mode"] == "not_required"
        assert stages[name]["derived_from"] == "sugar_1"


@pytest.mark.parametrize("profile_id", ("k2_n2", "k2_n3"))
def test_k2_profiles_require_both_physical_sugar_stages(profile_id: str) -> None:
    evaluator = _evaluator(profile_id)

    first = evaluator.observe(_snapshot(1, profile_id=profile_id, count=1))
    assert first.stages["sugar_1"].qualified
    assert not first.stages["sugar_2"].qualified

    second = evaluator.observe(_snapshot(2, profile_id=profile_id, count=2))
    stages = second.to_diagnostics()["stages"]
    assert stages["sugar_2"]["applicable"]
    assert stages["sugar_2"]["qualified"]
    assert stages["sugar_2"]["completion_mode"] == "physical"
    assert stages["sugar_2"]["derived_from"] is None
    assert not stages["sugar_3"]["applicable"]
    assert stages["sugar_3"]["effective_success"]
    assert stages["sugar_3"]["completion_mode"] == "not_required"
    assert stages["sugar_3"]["derived_from"] == "sugar_2"


def test_k3_n3_requires_three_physical_sugar_stages() -> None:
    evaluator = _evaluator("k3_n3")

    first = evaluator.observe(_snapshot(1, profile_id="k3_n3", count=1))
    assert first.stages["sugar_1"].qualified
    assert not first.stages["sugar_2"].qualified

    second = evaluator.observe(_snapshot(2, profile_id="k3_n3", count=2))
    assert second.stages["sugar_2"].qualified
    assert not second.stages["sugar_3"].qualified

    third = evaluator.observe(_snapshot(3, profile_id="k3_n3", count=3))
    stages = third.to_diagnostics()["stages"]
    assert stages["sugar_3"]["applicable"]
    assert stages["sugar_3"]["qualified"]
    assert stages["sugar_3"]["completion_mode"] == "physical"
    assert stages["sugar_3"]["derived_from"] is None


@pytest.mark.parametrize("profile_id", ("k1_n2", "k1_n3", "k2_n3"))
def test_profile_specific_over_add_is_terminal(profile_id: str) -> None:
    profile = profile_for_id(profile_id)
    evaluator = _evaluator(profile_id)

    for count in range(1, profile.target_sugar_count + 1):
        evaluator.observe(_snapshot(count, profile_id=profile_id, count=count))
    result = evaluator.observe(
        _snapshot(
            profile.target_sugar_count + 1,
            profile_id=profile_id,
            count=profile.target_sugar_count + 1,
            button_pressed=True,
        )
    )

    assert result.terminal
    assert not result.success
    assert result.failure_code == "over_add"
    assert not result.stages["button"].qualified
    assert not result.stages["success"].qualified
    assert result.details["over_add"]["profile_id"] == profile_id


@pytest.mark.parametrize("profile_id", ("k1_n2", "k1_n3"))
def test_k1_over_add_revokes_unused_sugar_summary_padding(profile_id: str) -> None:
    evaluator = _evaluator(profile_id)

    evaluator.observe(_snapshot(1, profile_id=profile_id, count=1))
    before = evaluator.result().to_diagnostics()["stages"]
    assert before["sugar_2"]["effective_success"]

    result = evaluator.observe(_snapshot(2, profile_id=profile_id, count=2))
    after = result.to_diagnostics()["stages"]
    assert after["sugar_1"]["qualified"]
    assert after["sugar_1"]["effective_success"]
    assert not after["sugar_2"]["qualified"]
    assert not after["sugar_2"]["effective_success"]
    assert after["sugar_2"]["completion_mode"] == "blocked_by_over_add"


@pytest.mark.parametrize("profile_id", ("k2_n2", "k2_n3"))
def test_count_jump_cannot_bypass_sugar_1(profile_id: str) -> None:
    evaluator = _evaluator(profile_id)

    jumped = evaluator.observe(_snapshot(1, profile_id=profile_id, count=2))
    assert jumped.stages["sugar_1"].qualified
    assert jumped.stages["sugar_2"].observed
    assert not jumped.stages["sugar_2"].qualified
    assert jumped.stages["sugar_2"].reason == (
        "prerequisite_not_qualified:sugar_1"
    )

    held = evaluator.observe(_snapshot(2, profile_id=profile_id, count=2))
    assert not held.stages["sugar_2"].qualified

    evaluator.observe(_snapshot(3, profile_id=profile_id, count=1))
    recovered = evaluator.observe(_snapshot(4, profile_id=profile_id, count=2))
    assert recovered.stages["sugar_2"].qualified_step == 4


def test_button_before_profile_goal_requires_release_and_repress() -> None:
    evaluator = _evaluator("k2_n3")

    early = evaluator.observe(
        _snapshot(1, profile_id="k2_n3", button_pressed=True)
    )
    assert not early.stages["button"].qualified
    assert early.stages["button"].reason == (
        "prerequisite_not_qualified:sugar_2"
    )

    evaluator.observe(
        _snapshot(2, profile_id="k2_n3", count=1, button_pressed=True)
    )
    evaluator.observe(
        _snapshot(3, profile_id="k2_n3", count=2, button_pressed=True)
    )
    assert not evaluator.result().stages["button"].qualified

    evaluator.observe(
        _snapshot(4, profile_id="k2_n3", count=2, button_pressed=False)
    )
    retried = evaluator.observe(
        _snapshot(5, profile_id="k2_n3", count=2, button_pressed=True)
    )
    assert retried.success


def test_button_uses_current_count_and_release_state() -> None:
    evaluator = _evaluator("k1_n3")

    evaluator.observe(
        _snapshot(
            1,
            profile_id="k1_n3",
            count=1,
            objects_released=False,
        )
    )
    rejected = evaluator.observe(
        _snapshot(
            2,
            profile_id="k1_n3",
            count=1,
            objects_released=False,
            button_pressed=True,
        )
    )
    assert rejected.stages["sugar_1"].qualified
    assert not rejected.stages["button"].qualified
    assert rejected.stages["button"].reason == "sugar_goal_not_ready"

    evaluator.observe(
        _snapshot(3, profile_id="k1_n3", count=1, button_pressed=False)
    )
    result = evaluator.observe(
        _snapshot(4, profile_id="k1_n3", count=1, button_pressed=True)
    )
    assert result.success


def test_profile_configuration_is_frozen_during_an_episode() -> None:
    evaluator = _evaluator("k1_n2")

    with pytest.raises(ValueError, match="configuration changed"):
        evaluator.observe(_snapshot(1, profile_id="k1_n3", count=1))


def test_profile_id_must_match_k_and_n() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        TS03EvaluationSnapshot(
            step_id=0,
            profile_id="k1_n2",
            target_sugar_count=2,
            total_sugar_count=3,
            confirmed_sugar_count=0,
            objects_released=True,
            completion_button_pressed=False,
        )


def test_confirmed_count_cannot_exceed_profile_total() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _snapshot(0, profile_id="k2_n2", count=3)


def test_runtime_callbacks_use_the_ts03_task_local_evaluator() -> None:
    evaluator = _evaluator("k1_n2")

    class FakeEnv:
        snapshot = _snapshot(1, profile_id="k1_n2", count=1)

        def evaluation_snapshot(self) -> TS03EvaluationSnapshot:
            return self.snapshot

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["sugar_1"]
    assert diagnostics["effective_stage_checks"]["sugar_2"]
    assert not success_for_runtime(runtime, {})
    assert report_stages_for_runtime(runtime)["order"] == list(REPORT_STAGE_ORDER)
    assert diagnostics_for_runtime(runtime) == diagnostics


@pytest.mark.parametrize("profile_id", PROFILE_IDS)
def test_public_report_uses_profile_applicability_and_hides_button(
    profile_id: str,
) -> None:
    profile = profile_for_id(profile_id)
    evaluator = _evaluator(profile_id)
    result = evaluator.result()
    for count in range(1, profile.target_sugar_count + 1):
        result = evaluator.observe(
            _snapshot(count, profile_id=profile_id, count=count)
        )

    report = result.to_report_stages()

    assert report["task_id"] == "TS03"
    assert report["profile_id"] == "TS03/report-v2"
    assert report["order"] == ["sugar_1", "sugar_2", "sugar_3", "success"]
    assert "button" not in report["stages"]
    assert report["stages"]["sugar_1"] == {
        "applicable": True,
        "passed": True,
        "first_pass_step": 1,
    }
    assert report["stages"]["sugar_2"] == {
        "applicable": profile.target_sugar_count >= 2,
        "passed": True if profile.target_sugar_count >= 2 else None,
        "first_pass_step": 2 if profile.target_sugar_count >= 2 else None,
    }
    assert report["stages"]["sugar_3"] == {
        "applicable": profile.target_sugar_count == 3,
        "passed": True if profile.target_sugar_count == 3 else None,
        "first_pass_step": 3 if profile.target_sugar_count == 3 else None,
    }
    assert report["stages"]["success"] == {
        "applicable": True,
        "passed": False,
        "first_pass_step": None,
    }

    success_step = profile.target_sugar_count + 1
    completed = evaluator.observe(
        _snapshot(
            success_step,
            profile_id=profile_id,
            count=profile.target_sugar_count,
            button_pressed=True,
        )
    ).to_report_stages()
    assert completed["stages"]["success"] == {
        "applicable": True,
        "passed": True,
        "first_pass_step": success_step,
    }


def test_public_report_keeps_sugar_history_when_over_add_blocks_success() -> None:
    evaluator = _evaluator("k2_n3")
    evaluator.observe(_snapshot(1, profile_id="k2_n3", count=1))
    evaluator.observe(_snapshot(2, profile_id="k2_n3", count=2))

    report = evaluator.observe(
        _snapshot(3, profile_id="k2_n3", count=3)
    ).to_report_stages()

    assert report["stages"]["sugar_1"]["passed"] is True
    assert report["stages"]["sugar_2"]["passed"] is True
    assert report["stages"]["success"]["passed"] is False
