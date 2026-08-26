from bench.tasks.wx01.config import load_task_config
from bench.tasks.wx01.domain.episode import resolve_task_definition
from bench.tasks.wx01.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
)


def _physical_state(**overrides: bool) -> PhysicalState:
    values = {
        "blocking_object_at_origin": True,
        "blocking_object_on_dining_table": True,
        "cleaning_tool_released": True,
        "blocking_object_released": True,
    }
    values.update(overrides)
    return PhysicalState(**values)


def test_blocking_object_release_is_not_required_for_success() -> None:
    report = evaluate_success(
        SemanticState(remembered_area_wiped=True),
        _physical_state(
            blocking_object_released=False,
        ),
    )

    assert report.success
    assert report.checks() == {
        "remembered_area_wiped": True,
        "cleaning_tool_released": True,
        "blocking_object_at_origin": True,
        "blocking_object_on_dining_table": True,
    }


def test_all_three_task_stages_are_required() -> None:
    cases = (
        (SemanticState(remembered_area_wiped=False), _physical_state()),
        (
            SemanticState(remembered_area_wiped=True),
            _physical_state(cleaning_tool_released=False),
        ),
        (
            SemanticState(remembered_area_wiped=True),
            _physical_state(blocking_object_at_origin=False),
        ),
        (
            SemanticState(remembered_area_wiped=True),
            _physical_state(blocking_object_on_dining_table=False),
        ),
    )

    assert all(not evaluate_success(state, physical).success for state, physical in cases)


def test_declared_success_conditions_match_runtime_checks() -> None:
    config = load_task_config()
    definition = resolve_task_definition(**config.task)
    report = evaluate_success(
        SemanticState(remembered_area_wiped=True),
        _physical_state(),
    )

    assert definition.success_conditions == tuple(report.checks())
