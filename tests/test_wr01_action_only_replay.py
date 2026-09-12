from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from bench.tasks.wr01.config import load_task_config, resolve_action_config
from bench.tasks.wr01.domain.semantics import PhysicalState, SemanticState
from bench.tasks.wr01.physical_events import PhysicalSample, WR01PhysicalObserver


ROOT = Path(__file__).resolve().parents[1]
WR01_ROOT = ROOT / "bench" / "tasks" / "wr01"


def _function_node(path: Path, class_name: str, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def test_wr01_rejects_collision_masking_configuration() -> None:
    operations = deepcopy(dict(load_task_config().operations))
    operations["storage_pick"]["disable_collisions_during_transport"] = True

    with pytest.raises(ValueError, match="collision masking.*forbidden"):
        resolve_action_config(operations=operations)


def test_wr01_step_passes_the_recorded_action_through_unchanged() -> None:
    node = _function_node(
        WR01_ROOT / "environment.py",
        "WR01RetrieveRememberedCan",
        "step",
    )
    assert len(node.args.args) == 2
    action_name = node.args.args[1].arg
    super_step_calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "step"
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
    ]
    assert len(super_step_calls) == 1
    assert len(super_step_calls[0].args) == 1
    assert isinstance(super_step_calls[0].args[0], ast.Name)
    assert super_step_calls[0].args[0].id == action_name

    forbidden_writes = [
        target
        for statement in ast.walk(node)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        for target in (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if any(
            isinstance(part, ast.Attribute) and part.attr in {"data", "model", "sim"}
            for part in ast.walk(target)
        )
    ]
    assert forbidden_writes == []


def test_wr01_execution_has_no_object_lock_or_collision_mask_calls() -> None:
    forbidden_identifiers = {
        "install_held_item_follow_assist",
        "disable_object_collisions",
        "restore_object_collisions",
        "set_object_pose_from_eef",
        "sync_held_object_pose",
    }
    identifiers: set[str] = set()
    for path in sorted((WR01_ROOT / "execution").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        identifiers.update(
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        )
        identifiers.update(
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        )

    assert identifiers.isdisjoint(forbidden_identifiers)


def test_wr01_physical_evaluation_does_not_inspect_the_neutral_robot_pose() -> None:
    node = _function_node(
        WR01_ROOT / "environment.py",
        "WR01RetrieveRememberedCan",
        "_physical_event_sample",
    )
    namespace = {"PhysicalSample": PhysicalSample}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "environment.py", "exec"), namespace)
    sample = namespace["_physical_event_sample"]

    def unexpected_pose_check() -> bool:
        raise AssertionError("policy evaluation must not check the expert neutral pose")

    doors = {"lower": False, "upper": False}
    env = SimpleNamespace(
        timestep=0,
        is_item_in_level=lambda item, level: {
            "blue_can": "lower", "red_can": "upper"
        }[item] == level,
        is_door_closed=lambda level: doors[level],
        is_door_open=lambda level: not doors[level],
        is_canonical_reset=unexpected_pose_check,
    )
    state = SemanticState("blue_can", "lower")
    observer = WR01PhysicalObserver(state, dwell_steps=1)
    observer.reset(sample(env))
    for step in (1, 2):
        env.timestep = step
        observer.observe(sample(env))

    doors.update(lower=True, upper=True)
    env.timestep = 3
    observer.observe(sample(env))
    assert state.storage_hidden
    assert not state.canonical_reset_complete

    doors["lower"] = False
    env.timestep = 4
    observer.observe(sample(env))
    assert state.retrieval_level_opened == "lower"
    assert not state.wrong_retrieval_level_opened


@pytest.mark.parametrize("target_item", ("red_can", "blue_can"))
@pytest.mark.parametrize("target_level", ("lower", "upper"))
def test_wr01_distractor_check_uses_the_initial_cabinet(
    target_item: str, target_level: str,
) -> None:
    node = _function_node(
        WR01_ROOT / "environment.py",
        "WR01RetrieveRememberedCan",
        "physical_state",
    )
    namespace = {"PhysicalState": PhysicalState}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "environment.py", "exec"), namespace)
    physical_state = namespace["physical_state"]

    distractor = "blue_can" if target_item == "red_can" else "red_can"
    original_level = "upper" if target_level == "lower" else "lower"
    assignment = {target_item: target_level, distractor: original_level}
    current_levels = dict(assignment)
    checks = []

    def is_item_in_level(item: str, level: str) -> bool:
        checks.append((item, level))
        return current_levels[item] == level

    env = SimpleNamespace(
        distractor_item_id=distractor,
        level_by_item=assignment,
        is_item_in_level=is_item_in_level,
    )
    # Cabinet containment is sufficient; no exact pose or grasp query is needed.
    assert physical_state(env).distractor_still_stored

    current_levels[distractor] = target_level
    assert not physical_state(env).distractor_still_stored
    current_levels[distractor] = None
    assert not physical_state(env).distractor_still_stored
    current_levels[distractor] = original_level
    assert physical_state(env).distractor_still_stored
    assert checks == [(distractor, original_level)] * 4


@pytest.fixture
def table_placement_env() -> tuple[SimpleNamespace, dict[str, bool]]:
    node = _function_node(
        WR01_ROOT / "environment.py",
        "WR01RetrieveRememberedCan",
        "is_item_on_table",
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "environment.py", "exec"), namespace)

    contacts = {"counter": True, "gripper": False}
    counter, gripper = object(), object()
    items = {item_id: object() for item_id in ("red_can", "blue_can")}

    def check_contact(first: object, second: object) -> bool:
        if first in items.values() and second is counter:
            return contacts["counter"]
        if first is gripper and second in items.values():
            return contacts["gripper"]
        raise AssertionError("placement must inspect the task counter and gripper")

    def unexpected_pose_check(*args: object, **kwargs: object) -> None:
        raise AssertionError("policy placement must not constrain position or orientation")

    env = SimpleNamespace(
        objects=items,
        counter=counter,
        robots=[SimpleNamespace(gripper={"right": gripper})],
        check_contact=check_contact,
        table_target_position=unexpected_pose_check,
        get_object_world_position=unexpected_pose_check,
        get_object_world_quaternion_xyzw=unexpected_pose_check,
    )
    env.is_item_on_table = MethodType(namespace["is_item_on_table"], env)
    return env, contacts


@pytest.mark.parametrize("item_id", ("red_can", "blue_can"))
@pytest.mark.parametrize(
    ("counter_contact", "gripper_contact", "expected"),
    ((True, False, True), (True, True, False), (False, False, False), (False, True, False)),
)
def test_wr01_table_placement_requires_counter_contact_and_release_only(
    table_placement_env: tuple[SimpleNamespace, dict[str, bool]],
    item_id: str,
    counter_contact: bool,
    gripper_contact: bool,
    expected: bool,
) -> None:
    env, contacts = table_placement_env
    contacts.update(counter=counter_contact, gripper=gripper_contact)

    # No sampled target, object pose, or EEF position is available. Released
    # contact must suffice, including far from the expert target or on its side.
    assert env.is_item_on_table(item_id) is expected


@pytest.mark.parametrize("failed_query", ("counter", "gripper"))
def test_wr01_table_placement_fails_closed_when_contact_is_unavailable(
    table_placement_env: tuple[SimpleNamespace, dict[str, bool]],
    failed_query: str,
) -> None:
    env, _ = table_placement_env
    original_check = env.check_contact

    def check_contact(first: object, second: object) -> bool:
        query = "counter" if second is env.counter else "gripper"
        if query == failed_query:
            raise RuntimeError("contact data unavailable")
        return original_check(first, second)

    env.check_contact = check_contact
    assert not env.is_item_on_table("red_can")


def test_wr01_table_placement_rejects_unknown_objects(
    table_placement_env: tuple[SimpleNamespace, dict[str, bool]],
) -> None:
    env, _ = table_placement_env
    assert not env.is_item_on_table("unknown_can")
