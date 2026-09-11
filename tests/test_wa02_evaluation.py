from __future__ import annotations

from types import SimpleNamespace

import pytest

from bench.evaluation.report_stages import validate_report_stages
from bench.tasks.wa02.evaluation import (
    REPORT_STAGE_ORDER,
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WA02EvaluationSnapshot,
    WA02Evaluator,
    diagnostics_for_runtime,
    report_stages_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    requested_temperature: str = "hot",
    temperature_order: tuple[str, str] | None = ("hot", "cold"),
    picked: tuple[str, ...] = (),
    pick_steps: dict[str, int] | None = None,
    filled: dict[str, str] | None = None,
    staged: dict[str, str] | None = None,
    physically_ready_by_cup: dict[str, bool] | None = None,
    cup_on_counter: dict[str, bool] | None = None,
    cup_upright: dict[str, bool] | None = None,
    selected: str | None = None,
) -> WA02EvaluationSnapshot:
    resolved_pick_steps = (
        dict(pick_steps)
        if pick_steps is not None
        else {cup: 1 + 3 * index for index, cup in enumerate(picked)}
    )
    return WA02EvaluationSnapshot(
        step_id=step,
        requested_temperature=requested_temperature,
        fill_temperature_order=temperature_order,
        pickup_step_by_cup=resolved_pick_steps,
        filled_temperature_by_cup=filled or {},
        staged_slot_by_cup=staged or {},
        physically_ready_by_cup=physically_ready_by_cup or {},
        cup_on_counter=cup_on_counter or {},
        cup_upright=cup_upright or {},
        final_taken_cup_id=selected,
        details={"physical_events": {"test_step": step}},
    )


def _evaluator(
    *,
    requested_temperature: str = "hot",
    temperature_order: tuple[str, str] = ("hot", "cold"),
) -> WA02Evaluator:
    evaluator = WA02Evaluator(
        fill_temperature_order=temperature_order,
        requested_temperature=requested_temperature,
    )
    evaluator.reset(
        _snapshot(
            0,
            requested_temperature=requested_temperature,
            temperature_order=temperature_order,
        )
    )
    return evaluator


def _successful_snapshots(
    *,
    first: str = "cup_A",
    first_temperature: str = "hot",
    requested_temperature: str = "hot",
    first_slot: str = "slot_1",
) -> list[WA02EvaluationSnapshot]:
    second = "cup_B" if first == "cup_A" else "cup_A"
    second_temperature = "cold" if first_temperature == "hot" else "hot"
    second_slot = "slot_2" if first_slot == "slot_1" else "slot_1"
    common = {
        "requested_temperature": requested_temperature,
        "picked": (first, second),
    }
    return [
        _snapshot(1, requested_temperature=requested_temperature, picked=(first,)),
        _snapshot(
            2,
            requested_temperature=requested_temperature,
            picked=(first,),
            filled={first: first_temperature},
        ),
        _snapshot(
            3,
            requested_temperature=requested_temperature,
            picked=(first,),
            filled={first: first_temperature},
            staged={first: first_slot},
        ),
        _snapshot(
            4,
            **common,
            filled={first: first_temperature},
            staged={first: first_slot},
        ),
        _snapshot(
            5,
            **common,
            filled={first: first_temperature, second: second_temperature},
            staged={first: first_slot},
        ),
        _snapshot(
            6,
            **common,
            filled={first: first_temperature, second: second_temperature},
            staged={first: first_slot, second: second_slot},
        ),
    ]


def test_wa02_evaluator_qualifies_actual_hot_cold_binding() -> None:
    evaluator = _evaluator(requested_temperature="cold")
    snapshots = _successful_snapshots(
        first="cup_A", requested_temperature="cold", first_slot="slot_2"
    )
    for snapshot in snapshots:
        evaluator.observe(snapshot)
    result = evaluator.observe(
        _snapshot(
            7,
            requested_temperature="cold",
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_2", "cup_B": "slot_1"},
            selected="cup_B",
        )
    )

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert result.details["evaluation"]["profile_id"] == "wa02-seven-stage-v3"
    assert {stage: result.stages[stage].qualified_step for stage in STAGE_ORDER} == {
        stage: index for index, stage in enumerate(STAGE_ORDER, start=1)
    }
    bindings = result.details["evaluation"]["stage_bindings"]
    assert bindings == {
        "binding_source": "physical_events",
        "first_cup_id": "cup_A",
        "first_temperature": "hot",
        "second_cup_id": "cup_B",
        "second_temperature": "cold",
        "requested_temperature": "cold",
        "requested_cup_id": "cup_B",
        "service_slot_by_cup": {"cup_A": "slot_2", "cup_B": "slot_1"},
    }
    report = result.to_report_stages()
    assert report["order"] == list(REPORT_STAGE_ORDER)
    assert {
        name: report["stages"][name]["first_pass_step"]
        for name in REPORT_STAGE_ORDER
    } == {
        "first_cup_filled": 2,
        "first_cup_ready": 3,
        "second_cup_filled": 5,
        "both_cups_ready": 6,
        "success": 7,
    }
    assert validate_report_stages(
        report,
        expected_task_id="WA-02",
        expected_profile_id="WA-02/report-v2",
        expected_success=True,
    ) == report


def test_wa02_evaluator_accepts_either_physical_cup_as_first() -> None:
    evaluator = _evaluator(requested_temperature="hot")
    for snapshot in _successful_snapshots(
        first="cup_B", requested_temperature="hot", first_slot="slot_1"
    ):
        evaluator.observe(snapshot)
    result = evaluator.observe(
        _snapshot(
            7,
            requested_temperature="hot",
            picked=("cup_B", "cup_A"),
            filled={"cup_B": "hot", "cup_A": "cold"},
            staged={"cup_B": "slot_1", "cup_A": "slot_2"},
            selected="cup_B",
        )
    )

    assert result.success
    bindings = result.details["evaluation"]["stage_bindings"]
    assert bindings["first_cup_id"] == "cup_B"
    assert bindings["second_cup_id"] == "cup_A"
    assert bindings["requested_cup_id"] == "cup_B"


def test_wa02_policy_evaluation_ready_uses_counter_contact_and_upright_only() -> None:
    """Policy readiness must not require a service slot or slot assignment."""

    evaluator = _evaluator()
    false_ready = {"cup_A": False, "cup_B": False}
    snapshots = [
        _snapshot(1, picked=("cup_A",), staged={}),
        _snapshot(
            2,
            picked=("cup_A",),
            filled={"cup_A": "hot"},
            staged={},
            physically_ready_by_cup=false_ready,
        ),
        _snapshot(
            3,
            picked=("cup_A",),
            filled={"cup_A": "hot"},
            staged={},
            cup_on_counter={"cup_A": True},
            cup_upright={"cup_A": True},
        ),
        _snapshot(
            4,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot"},
            staged={},
            physically_ready_by_cup={"cup_A": True, "cup_B": False},
        ),
        _snapshot(
            5,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={},
            physically_ready_by_cup={"cup_A": True, "cup_B": False},
        ),
        _snapshot(
            6,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            # Deliberately provide unreliable slot metadata.  The physical
            # readiness contract must ignore this metadata.
            staged={"cup_A": "slot_1", "cup_B": "not_reliably_assigned"},
            physically_ready_by_cup={"cup_A": True, "cup_B": True},
        ),
    ]
    for snapshot in snapshots:
        evaluator.observe(snapshot)

    result = evaluator.observe(
        _snapshot(
            7,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_1", "cup_B": "not_reliably_assigned"},
            physically_ready_by_cup={"cup_A": False, "cup_B": False},
            selected="cup_A",
        )
    )

    assert result.success
    assert result.stages["first_cup_ready"].qualified
    assert result.stages["second_cup_ready"].qualified


def test_wa02_evaluator_rejects_second_pick_before_first_cup_ready() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, picked=("cup_A",)))
    result = evaluator.observe(_snapshot(2, picked=("cup_A", "cup_B")))

    stage = result.stages["second_cup_pick"]
    assert stage.observed
    assert not stage.qualified
    assert stage.invalidated
    assert stage.reason == "prerequisite_not_qualified:first_cup_ready"
    assert result.failure_code == stage.reason


def test_wa02_evaluator_binds_first_temperature_from_actual_fill() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, picked=("cup_A",)))
    result = evaluator.observe(
        _snapshot(2, picked=("cup_A",), filled={"cup_A": "cold"})
    )

    assert result.stages["first_cup_fill"].observed
    assert result.stages["first_cup_fill"].qualified
    bindings = result.details["evaluation"]["stage_bindings"]
    assert bindings["first_temperature"] == "cold"
    assert bindings["second_temperature"] is None
    report = result.to_report_stages()
    assert report["stages"]["first_cup_filled"]["passed"]
    assert not report["stages"]["first_cup_ready"]["passed"]
    validate_report_stages(
        report,
        expected_task_id="WA-02",
        expected_profile_id="WA-02/report-v2",
        expected_success=False,
    )


@pytest.mark.parametrize(
    ("requested_temperature", "selected"),
    (("cold", "cup_A"), ("hot", "cup_B")),
)
def test_wa02_evaluator_accepts_cold_first_execution(
    requested_temperature: str,
    selected: str,
) -> None:
    evaluator = _evaluator(requested_temperature=requested_temperature)
    for snapshot in _successful_snapshots(
        first="cup_A",
        first_temperature="cold",
        requested_temperature=requested_temperature,
        first_slot="slot_2",
    ):
        evaluator.observe(snapshot)
    result = evaluator.observe(
        _snapshot(
            7,
            requested_temperature=requested_temperature,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "cold", "cup_B": "hot"},
            staged={"cup_A": "slot_2", "cup_B": "slot_1"},
            selected=selected,
        )
    )

    assert result.success
    assert all(result.stages[name].qualified for name in STAGE_ORDER)
    bindings = result.details["evaluation"]["stage_bindings"]
    assert bindings["binding_source"] == "physical_events"
    assert bindings["first_cup_id"] == "cup_A"
    assert bindings["first_temperature"] == "cold"
    assert bindings["second_cup_id"] == "cup_B"
    assert bindings["second_temperature"] == "hot"
    assert bindings["requested_temperature"] == requested_temperature
    assert bindings["requested_cup_id"] == selected
    assert bindings["service_slot_by_cup"] == {
        "cup_A": "slot_2",
        "cup_B": "slot_1",
    }
    report = result.to_report_stages()
    assert {
        name: report["stages"][name]["first_pass_step"]
        for name in REPORT_STAGE_ORDER
    } == {
        "first_cup_filled": 2,
        "first_cup_ready": 3,
        "second_cup_filled": 5,
        "both_cups_ready": 6,
        "success": 7,
    }
    assert validate_report_stages(
        report,
        expected_task_id="WA-02",
        expected_profile_id="WA-02/report-v2",
        expected_success=True,
    ) == report


def test_wa02_evaluator_rejects_same_temperature_for_both_cups() -> None:
    evaluator = _evaluator()
    states = [
        _snapshot(1, picked=("cup_A",)),
        _snapshot(2, picked=("cup_A",), filled={"cup_A": "cold"}),
        _snapshot(
            3,
            picked=("cup_A",),
            filled={"cup_A": "cold"},
            staged={"cup_A": "slot_1"},
        ),
        _snapshot(
            4,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "cold"},
            staged={"cup_A": "slot_1"},
        ),
    ]
    for snapshot in states:
        evaluator.observe(snapshot)
    result = evaluator.observe(
        _snapshot(
            5,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "cold", "cup_B": "cold"},
            staged={"cup_A": "slot_1"},
        )
    )

    stage = result.stages["second_cup_fill"]
    assert stage.observed
    assert not stage.qualified
    assert stage.reason == "duplicate_fill_temperature"


def test_wa02_evaluator_rejects_duplicate_service_slot() -> None:
    evaluator = _evaluator()
    states = [
        _snapshot(1, picked=("cup_A",)),
        _snapshot(2, picked=("cup_A",), filled={"cup_A": "hot"}),
        _snapshot(
            3,
            picked=("cup_A",),
            filled={"cup_A": "hot"},
            staged={"cup_A": "slot_1"},
        ),
        _snapshot(
            4,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot"},
            staged={"cup_A": "slot_1"},
        ),
        _snapshot(
            5,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_1"},
        ),
    ]
    for snapshot in states:
        evaluator.observe(snapshot)
    result = evaluator.observe(
        _snapshot(
            6,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_1", "cup_B": "slot_1"},
        )
    )

    assert result.stages["second_cup_ready"].observed
    assert result.stages["second_cup_ready"].reason == "invalid_service_staging"


def test_wa02_evaluator_rejects_selection_before_both_cups_are_ready() -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, picked=("cup_A",)))
    result = evaluator.observe(
        _snapshot(
            2,
            picked=("cup_A",),
            filled={"cup_A": "hot"},
            selected="cup_A",
        )
    )

    selection = result.stages["selection_success"]
    assert selection.observed
    assert not selection.qualified
    assert selection.reason == "prerequisite_not_qualified:second_cup_ready"


def test_wa02_evaluator_compares_selection_with_actual_fill() -> None:
    evaluator = _evaluator(requested_temperature="cold")
    for snapshot in _successful_snapshots(
        first="cup_B", requested_temperature="cold", first_slot="slot_2"
    ):
        evaluator.observe(snapshot)
    result = evaluator.observe(
        _snapshot(
            7,
            requested_temperature="cold",
            picked=("cup_B", "cup_A"),
            filled={"cup_B": "hot", "cup_A": "cold"},
            staged={"cup_B": "slot_2", "cup_A": "slot_1"},
            selected="cup_B",
        )
    )

    assert not result.success
    assert result.details["evaluation"]["stage_bindings"]["requested_cup_id"] == "cup_A"
    assert result.stages["selection_success"].reason == "wrong_cup_selected"


def test_wa02_evaluator_rejects_ambiguous_pickup_order() -> None:
    evaluator = _evaluator()
    result = evaluator.observe(
        _snapshot(1, pick_steps={"cup_A": 1, "cup_B": 1})
    )

    assert result.stages["first_cup_pick"].reason == "ambiguous_pickup_order"


def test_wa02_evaluator_is_idempotent_and_rejects_backward_steps() -> None:
    evaluator = _evaluator()
    first = evaluator.observe(_snapshot(1, picked=("cup_A",)))

    assert evaluator.observe(_snapshot(1, picked=("cup_A",))) is first
    with pytest.raises(ValueError, match="moved backwards"):
        evaluator.observe(_snapshot(0))


def test_wa02_evaluator_does_not_require_demonstration_order() -> None:
    evaluator = WA02Evaluator(requested_temperature="hot")
    evaluator.reset(_snapshot(0, temperature_order=None))

    result = evaluator.observe(
        _snapshot(1, temperature_order=("cold", "hot"), picked=("cup_B",))
    )

    assert result.stages["first_cup_pick"].qualified
    assert result.details["evaluation"]["demonstration_temperature_order"] is None


def test_wa02_runtime_callbacks_share_the_initialized_evaluator() -> None:
    evaluator = _evaluator()

    class FakeEnv:
        def evaluation_snapshot(self) -> WA02EvaluationSnapshot:
            return _snapshot(1, picked=("cup_A",))

    runtime = SimpleNamespace(
        env=FakeEnv(),
        metadata={TASK_EVALUATOR_METADATA_KEY: evaluator},
    )

    diagnostics = diagnostics_for_runtime(runtime)
    assert diagnostics["stage_checks"]["first_cup_pick"]
    assert not success_for_runtime(runtime, {})
    assert report_stages_for_runtime(runtime)["order"] == list(REPORT_STAGE_ORDER)
    assert diagnostics_for_runtime(runtime) == diagnostics


def test_wa02_runtime_initialization_attaches_dynamic_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.tasks.wa02 import runtime as runtime_module

    plan = SimpleNamespace(
        fill_temperature_order=("hot", "cold"),
        requested_temperature="cold",
    )
    descriptor = SimpleNamespace(
        episode=SimpleNamespace(split="evaluation"),
        episode_plan=plan,
        config=SimpleNamespace(
            initial_state=SimpleNamespace(objects={}, robot_base=object())
        ),
        navigation=object(),
        scene=SimpleNamespace(scene_seed=0),
    )

    class FakeEnv:
        def configure_episode(self, episode: object) -> None:
            assert episode is plan

        def configure_canonical_physics(self) -> None:
            self.canonical_physics_configured = True

        def evaluation_snapshot(self) -> WA02EvaluationSnapshot:
            return _snapshot(0, requested_temperature="cold")

    monkeypatch.setattr(
        runtime_module,
        "apply_sampled_initial_object_regions",
        lambda env, settings, enabled: ({"objects": True}, {"applied": True}),
    )
    monkeypatch.setattr(
        runtime_module,
        "apply_robot_base_initial_state",
        lambda env, obs, **kwargs: ({**obs, "base": True}, {"applied": True}),
    )
    env = FakeEnv()

    updated, _, metadata, _ = runtime_module._initialize(
        descriptor, env, {"initial": True}
    )

    evaluator = metadata[TASK_EVALUATOR_METADATA_KEY]
    assert isinstance(evaluator, WA02Evaluator)
    assert evaluator.requested_temperature == "cold"
    assert env._wa02_evaluator is evaluator
    assert env.require_service_region_for_staging is False
    assert updated == {"objects": True, "base": True}


def test_wa02_expert_recording_restores_region_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from bench.tasks.wa02.expert import operations

    env = SimpleNamespace(require_service_region_for_staging=False)
    descriptor = SimpleNamespace(
        episode=SimpleNamespace(split="evaluation"),
        navigation=object(),
        scene=SimpleNamespace(scene_seed=0),
        config=SimpleNamespace(manipulation={}),
        resources={"planner_script": "planner.py", "cup_grasp_candidates": "grasps.json"},
        execute=True,
        actions=SimpleNamespace(handler_settings={}),
    )
    monkeypatch.setattr(
        operations,
        "build_navigation_assembly",
        lambda *args, **kwargs: SimpleNamespace(controller=object()),
    )
    monkeypatch.setattr(
        operations,
        "build_manipulation_assembly",
        lambda *args, **kwargs: SimpleNamespace(controller=SimpleNamespace()),
    )
    monkeypatch.setattr(
        operations,
        "StageRunner",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    parts = operations.build_operation_handlers(descriptor, env, {})

    assert env.require_service_region_for_staging is True
    assert parts["runner"] is not None


def test_wa02_runtime_registers_public_report_stages() -> None:
    from bench.tasks.wa02 import runtime

    assert runtime.TASK_RUNTIME_API.report_stage_profile_id == "WA-02/report-v2"
    assert callable(runtime.TASK_RUNTIME_API.report_stages)
