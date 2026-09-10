from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from bench.tasks.wa01.domain.semantics import SemanticState, evaluate_success
from bench.tasks.wa01.environment import WA01SortSugaredAndUnsugaredCoffee


def _bare_wa01_environment() -> WA01SortSugaredAndUnsugaredCoffee:
    env = object.__new__(WA01SortSugaredAndUnsugaredCoffee)
    env._semantic_state = SemanticState()
    env._selection_ready_positions_by_coffee = {}
    env._selection_armed = False
    env._physical_event_diagnostics = {
        "sugar_delivery": None,
        "cups_ready_latched": False,
        "cups_ready_released": False,
        "selection_armed": False,
        "selection": None,
    }
    env.cups_together = False
    env._hide_consumed_sugar_cube = lambda: None
    env.get_object_world_position = lambda coffee_id: np.asarray(
        [0.0, 0.0 if coffee_id == "coffee_A" else 0.3, 0.9],
        dtype=float,
    )
    return env


def test_wa01_physical_delivery_latches_once() -> None:
    env = _bare_wa01_environment()

    assert env._commit_sugar_delivery("coffee_A", source="physical_detection")
    assert env.sugar_consumed
    assert env.sugared_coffee_id == "coffee_A"
    assert not env._commit_sugar_delivery("coffee_B", source="physical_detection")
    assert env.sugared_coffee_id == "coffee_A"


def test_wa01_shared_physical_events_latch_ready_then_selection() -> None:
    env = _bare_wa01_environment()
    env.cups_together = True
    env._physically_delivered_sugar_coffee_id = lambda: "coffee_A"
    env._are_both_cups_released_for_selection = lambda: True
    selection_checks: list[bool] = []

    def selected() -> str:
        selection_checks.append(True)
        return "coffee_A"

    env._physically_selected_coffee_id = selected

    env._update_task_events_from_physics()
    assert env.sugar_consumed
    assert env.cups_together_before_selection
    assert env.selected_sugared_coffee_id is None
    assert selection_checks == []

    env._update_task_events_from_physics()
    assert env._selection_armed
    assert env._physical_event_diagnostics["cups_ready_released"]
    assert env._physical_event_diagnostics["selection_armed"]
    assert env.selected_sugared_coffee_id is None
    assert selection_checks == []

    env._update_task_events_from_physics()
    assert env.selected_sugared_coffee_id == "coffee_A"
    assert selection_checks == [True]
    assert evaluate_success(env._semantic_state).success


def test_wa01_placing_second_cup_cannot_count_as_selection() -> None:
    env = _bare_wa01_environment()
    env.cups_together = True
    env._physically_delivered_sugar_coffee_id = lambda: "coffee_A"
    env._are_both_cups_released_for_selection = lambda: False
    selection_checks: list[bool] = []

    def selected() -> str:
        selection_checks.append(True)
        return "coffee_A"

    env._physically_selected_coffee_id = selected

    env._update_task_events_from_physics()
    env._update_task_events_from_physics()

    assert env.cups_together_before_selection
    assert not env._selection_armed
    assert env.selected_sugared_coffee_id is None
    assert selection_checks == []


def test_wa01_selection_arms_only_while_both_cups_remain_placed() -> None:
    env = _bare_wa01_environment()
    env.cups_together = True
    env._physically_delivered_sugar_coffee_id = lambda: "coffee_A"
    env._are_both_cups_released_for_selection = lambda: True
    env._physically_selected_coffee_id = lambda: "coffee_A"

    env._update_task_events_from_physics()
    env.cups_together = False
    env._update_task_events_from_physics()

    assert not env._selection_armed
    assert env.selected_sugared_coffee_id is None


def test_wa01_physical_delivery_requires_release_and_unique_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.tasks.wa01.environment as environment_module

    env = _bare_wa01_environment()
    monkeypatch.setattr(environment_module.OU, "gripper_obj_far", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        environment_module.OU,
        "check_obj_in_receptacle",
        lambda *args, **kwargs: True,
    )
    assert env._physically_delivered_sugar_coffee_id() is None

    monkeypatch.setattr(environment_module.OU, "gripper_obj_far", lambda *args, **kwargs: True)
    assert env._physically_delivered_sugar_coffee_id() is None

    monkeypatch.setattr(
        environment_module.OU,
        "check_obj_in_receptacle",
        lambda _env, _sugar, coffee_id: coffee_id == "coffee_B",
    )
    assert env._physically_delivered_sugar_coffee_id() == "coffee_B"
