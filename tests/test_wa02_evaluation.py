from __future__ import annotations

from types import SimpleNamespace

import pytest

from bench.tasks.wa02.evaluation import (
    STAGE_ORDER,
    TASK_EVALUATOR_METADATA_KEY,
    WA02EvaluationSnapshot,
    WA02Evaluator,
    diagnostics_for_runtime,
    success_for_runtime,
)


def _snapshot(
    step: int,
    *,
    cup_order: tuple[str, str] = ("cup_A", "cup_B"),
    temperature_order: tuple[str, str] = ("hot", "cold"),
    slots: dict[str, str] | None = None,
    requested: str = "cup_A",
    picked: tuple[str, ...] = (),
    filled: dict[str, str] | None = None,
    staged: dict[str, str] | None = None,
    selected: str | None = None,
) -> WA02EvaluationSnapshot:
    resolved_slots = slots or {"cup_A": "slot_1", "cup_B": "slot_2"}
    return WA02EvaluationSnapshot(
        step_id=step,
        fill_cup_order=cup_order,
        fill_temperature_order=temperature_order,
        service_slot_by_cup=resolved_slots,
        requested_cup_id=requested,
        pickup_events_by_cup={cup: cup in picked for cup in ("cup_A", "cup_B")},
        filled_temperature_by_cup=filled or {},
        staged_slot_by_cup=staged or {},
        final_taken_cup_id=selected,
        details={"physical_events": {"test_step": step}},
    )


def _evaluator(**configuration: object) -> WA02Evaluator:
    values = {
        "cup_order": ("cup_A", "cup_B"),
        "temperature_order": ("hot", "cold"),
        "slots": {"cup_A": "slot_1", "cup_B": "slot_2"},
        "requested": "cup_A",
    }
    values.update(configuration)
    evaluator = WA02Evaluator(
        fill_cup_order=values["cup_order"],
        fill_temperature_order=values["temperature_order"],
        service_slot_by_cup=values["slots"],
        requested_cup_id=values["requested"],
    )
    evaluator.reset(_snapshot(0, **values))
    return evaluator


def test_wa02_evaluator_qualifies_all_seven_stages_in_order() -> None:
    evaluator = _evaluator()

    evaluator.observe(_snapshot(1, picked=("cup_A",)))
    evaluator.observe(
        _snapshot(2, picked=("cup_A",), filled={"cup_A": "hot"})
    )
    evaluator.observe(
        _snapshot(
            3,
            picked=("cup_A",),
            filled={"cup_A": "hot"},
            staged={"cup_A": "slot_1"},
        )
    )
    evaluator.observe(
        _snapshot(
            4,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot"},
            staged={"cup_A": "slot_1"},
        )
    )
    evaluator.observe(
        _snapshot(
            5,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_1"},
        )
    )
    evaluator.observe(
        _snapshot(
            6,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_1", "cup_B": "slot_2"},
        )
    )
    result = evaluator.observe(
        _snapshot(
            7,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_1", "cup_B": "slot_2"},
            selected="cup_A",
        )
    )

    assert result.success
    assert result.terminal
    assert result.failure_code is None
    assert {stage: result.stages[stage].qualified_step for stage in STAGE_ORDER} == {
        stage: index for index, stage in enumerate(STAGE_ORDER, start=1)
    }
    diagnostics = result.to_diagnostics()
    assert diagnostics["evaluation"]["profile_id"] == "wa02-seven-stage-v1"
    assert all(diagnostics["stage_checks"].values())


def test_wa02_evaluator_binds_first_and_second_to_episode_order() -> None:
    configuration = {
        "cup_order": ("cup_B", "cup_A"),
        "temperature_order": ("cold", "hot"),
        "slots": {"cup_A": "slot_2", "cup_B": "slot_1"},
        "requested": "cup_B",
    }
    evaluator = _evaluator(**configuration)

    states = [
        {"picked": ("cup_B",)},
        {"picked": ("cup_B",), "filled": {"cup_B": "cold"}},
        {
            "picked": ("cup_B",),
            "filled": {"cup_B": "cold"},
            "staged": {"cup_B": "slot_1"},
        },
        {
            "picked": ("cup_B", "cup_A"),
            "filled": {"cup_B": "cold"},
            "staged": {"cup_B": "slot_1"},
        },
        {
            "picked": ("cup_B", "cup_A"),
            "filled": {"cup_B": "cold", "cup_A": "hot"},
            "staged": {"cup_B": "slot_1"},
        },
        {
            "picked": ("cup_B", "cup_A"),
            "filled": {"cup_B": "cold", "cup_A": "hot"},
            "staged": {"cup_B": "slot_1", "cup_A": "slot_2"},
        },
        {
            "picked": ("cup_B", "cup_A"),
            "filled": {"cup_B": "cold", "cup_A": "hot"},
            "staged": {"cup_B": "slot_1", "cup_A": "slot_2"},
            "selected": "cup_B",
        },
    ]
    result = None
    for step, state in enumerate(states, start=1):
        result = evaluator.observe(_snapshot(step, **configuration, **state))

    assert result is not None and result.success
    bindings = result.details["evaluation"]["stage_bindings"]
    assert bindings == {
        "first_cup_id": "cup_B",
        "first_temperature": "cold",
        "second_cup_id": "cup_A",
        "second_temperature": "hot",
    }


def test_wa02_evaluator_rejects_second_pick_before_first_cup_ready() -> None:
    evaluator = _evaluator()

    result = evaluator.observe(_snapshot(1, picked=("cup_B",)))

    stage = result.stages["second_cup_pick"]
    assert stage.observed
    assert not stage.qualified
    assert stage.invalidated
    assert stage.reason == "prerequisite_not_qualified:first_cup_ready"
    assert result.failure_code == stage.reason


@pytest.mark.parametrize(
    ("state", "stage", "reason"),
    [
        (
            {"picked": ("cup_A",), "filled": {"cup_A": "cold"}},
            "first_cup_fill",
            "wrong_fill_temperature",
        ),
        (
            {
                "picked": ("cup_A",),
                "filled": {"cup_A": "hot"},
                "staged": {"cup_A": "slot_2"},
            },
            "first_cup_ready",
            "wrong_service_slot",
        ),
    ],
)
def test_wa02_evaluator_observes_but_rejects_incorrect_events(
    state: dict[str, object], stage: str, reason: str
) -> None:
    evaluator = _evaluator()
    evaluator.observe(_snapshot(1, picked=("cup_A",)))
    if stage == "first_cup_ready":
        evaluator.observe(
            _snapshot(2, picked=("cup_A",), filled={"cup_A": "hot"})
        )
        step = 3
    else:
        step = 2

    result = evaluator.observe(_snapshot(step, **state))

    assert result.stages[stage].observed
    assert not result.stages[stage].qualified
    assert result.stages[stage].reason == reason


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


def test_wa02_evaluator_reports_wrong_final_selection() -> None:
    evaluator = _evaluator()
    snapshots = [
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
        _snapshot(
            6,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_1", "cup_B": "slot_2"},
        ),
    ]
    for snapshot in snapshots:
        evaluator.observe(snapshot)

    result = evaluator.observe(
        _snapshot(
            7,
            picked=("cup_A", "cup_B"),
            filled={"cup_A": "hot", "cup_B": "cold"},
            staged={"cup_A": "slot_1", "cup_B": "slot_2"},
            selected="cup_B",
        )
    )

    assert not result.success
    assert result.stages["selection_success"].reason == "wrong_cup_selected"
    assert result.failure_code == "wrong_cup_selected"


def test_wa02_evaluator_is_idempotent_and_rejects_backward_steps() -> None:
    evaluator = _evaluator()
    first = evaluator.observe(_snapshot(1, picked=("cup_A",)))

    assert evaluator.observe(_snapshot(1, picked=("cup_A",))) is first
    with pytest.raises(ValueError, match="moved backwards"):
        evaluator.observe(_snapshot(0))


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
    assert diagnostics_for_runtime(runtime) == diagnostics


def test_wa02_runtime_initialization_attaches_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bench.tasks.wa02 import runtime as runtime_module

    plan = SimpleNamespace(
        fill_cup_order=("cup_A", "cup_B"),
        fill_temperature_order=("hot", "cold"),
        service_slot_by_cup={"cup_A": "slot_1", "cup_B": "slot_2"},
        requested_cup_id="cup_A",
    )
    descriptor = SimpleNamespace(
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
            return _snapshot(0)

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
    assert env._wa02_evaluator is evaluator
    assert updated == {"objects": True, "base": True}
