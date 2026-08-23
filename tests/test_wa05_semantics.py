from bench.tasks.wa05.domain.semantics import (
    SemanticState,
    evaluate_success,
    record_button_press,
    record_verified_fill,
    record_verified_staging,
)


def _ready_state(temperature: str = "hot") -> SemanticState:
    state = SemanticState()
    assert record_verified_fill(state, temperature)
    assert record_verified_staging(state, currently_ready=True)
    return state


def test_actual_fill_temperature_controls_success() -> None:
    state = _ready_state("cold")
    assert record_button_press(state, "cold")
    assert evaluate_success(state).success


def test_wrong_button_is_absorbing_failure() -> None:
    state = _ready_state("hot")
    assert not record_button_press(state, "cold")
    assert state.failure_code == "wrong_temperature_button"
    assert not evaluate_success(state).success


def test_early_and_multiple_answers_are_absorbing_failures() -> None:
    early = SemanticState()
    assert not record_button_press(early, "hot")
    assert early.failure_code == "button_pressed_before_fill"

    multiple = _ready_state("hot")
    assert record_button_press(multiple, "hot")
    assert not record_button_press(multiple, "cold")
    assert multiple.failure_code == "multiple_answer_buttons"


def test_mixed_temperature_fill_is_absorbing_failure() -> None:
    state = SemanticState()
    assert record_verified_fill(state, "hot")
    assert not record_verified_fill(state, "cold")
    assert state.failure_code == "mixed_temperature_fill"


def test_staging_requires_fill_and_current_physical_readiness() -> None:
    state = SemanticState()
    assert not record_verified_staging(state, currently_ready=True)
    assert record_verified_fill(state, "hot")
    assert not record_verified_staging(state, currently_ready=False)
    assert record_verified_staging(state, currently_ready=True)
