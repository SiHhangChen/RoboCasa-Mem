from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from bench.tasks.wa03.domain.semantics import SemanticState, evaluate_success
from bench.tasks.wa03.environment import WA03SortSugaredAndUnsugaredCoffee


def _bare_wa03_environment() -> WA03SortSugaredAndUnsugaredCoffee:
    env = object.__new__(WA03SortSugaredAndUnsugaredCoffee)
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
    env.all_cups_ready = False
    env._hide_consumed_sugar_cube = lambda: None
    env._set_target_coffee_marker_visibility = lambda _visible: None
    positions = {
        "coffee_A": [0.0, 0.00, 0.9],
        "coffee_B": [0.0, 0.15, 0.9],
        "coffee_C": [0.0, 0.30, 0.9],
    }
    env.get_object_world_position = lambda coffee_id: np.asarray(
        positions[coffee_id], dtype=float
    )
    return env


def test_wa03_shared_physical_events_latch_ready_then_selection() -> None:
    env = _bare_wa03_environment()
    env.all_cups_ready = True
    env._physically_delivered_sugar_coffee_id = lambda: "coffee_B"
    env._are_all_cups_released_for_selection = lambda: True
    selection_checks: list[bool] = []

    def selected() -> str:
        selection_checks.append(True)
        return "coffee_B"

    env._physically_selected_coffee_id = selected

    env._update_task_events_from_physics()
    assert env.sugar_consumed
    assert env.all_cups_ready_before_selection
    assert env.selected_sugared_coffee_id is None
    assert selection_checks == []

    env._update_task_events_from_physics()
    assert env._selection_armed
    assert env.selected_sugared_coffee_id is None
    assert selection_checks == []

    env._update_task_events_from_physics()
    assert env.selected_sugared_coffee_id == "coffee_B"
    assert selection_checks == [True]
    assert evaluate_success(env._semantic_state).success


def test_wa03_ready_state_does_not_depend_on_extra_pairwise_margin() -> None:
    env = _bare_wa03_environment()
    env.is_coffee_cup_correctly_placed_on_dining_counter = lambda _coffee_id: True
    env._are_all_dining_regions_occupied = lambda: True
    env._cup_pairwise_xy_distances = lambda: {"coffee_A:coffee_C": 0.085}
    env.minimum_cup_xy_separation = 0.10

    assert env._are_cups_correctly_arranged()


def test_wa03_physical_delivery_requires_release_and_unique_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.tasks.wa03.environment as environment_module

    env = _bare_wa03_environment()
    monkeypatch.setattr(
        environment_module.OU,
        "gripper_obj_far",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        environment_module.OU,
        "check_obj_in_receptacle",
        lambda *args, **kwargs: True,
    )
    assert env._physically_delivered_sugar_coffee_id() is None

    monkeypatch.setattr(
        environment_module.OU,
        "gripper_obj_far",
        lambda *args, **kwargs: True,
    )
    assert env._physically_delivered_sugar_coffee_id() is None

    monkeypatch.setattr(
        environment_module.OU,
        "check_obj_in_receptacle",
        lambda _env, _sugar, coffee_id: coffee_id == "coffee_B",
    )
    assert env._physically_delivered_sugar_coffee_id() == "coffee_B"
