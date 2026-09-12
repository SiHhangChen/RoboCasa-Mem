from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from bench.tasks.wr03.config import load_task_config, resolve_action_config
from bench.tasks.wr03.domain.contracts import CAN_ID
from bench.tasks.wr03.domain.semantics import PhysicalState, SemanticState
from bench.tasks.wr03.physical_events import PhysicalSample, WR03PhysicalObserver


ROOT = Path(__file__).resolve().parents[1]
WR03_ROOT = ROOT / "bench" / "tasks" / "wr03"


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


def test_wr03_rejects_collision_masking_configuration() -> None:
    operations = deepcopy(dict(load_task_config().operations))
    operations["storage_pick"]["disable_collisions_during_transport"] = True

    with pytest.raises(ValueError, match="collision masking.*forbidden"):
        resolve_action_config(operations=operations)


def test_wr03_step_passes_the_recorded_action_through_unchanged() -> None:
    node = _function_node(
        WR03_ROOT / "environment.py",
        "WR03StoreAndRetrieveRememberedCan",
        "step",
    )
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


def test_wr03_execution_has_no_object_lock_or_collision_mask_calls() -> None:
    forbidden_identifiers = {
        "install_held_item_follow_assist",
        "disable_object_collisions",
        "restore_object_collisions",
        "set_object_pose_from_eef",
        "sync_held_object_pose",
    }
    identifiers: set[str] = set()
    for path in sorted((WR03_ROOT / "execution").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        identifiers.update(
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        )
        identifiers.update(
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        )

    assert identifiers.isdisjoint(forbidden_identifiers)


def test_wr03_physical_evaluation_does_not_inspect_the_neutral_robot_pose() -> None:
    node = _function_node(
        WR03_ROOT / "environment.py",
        "WR03StoreAndRetrieveRememberedCan",
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
        physical_state=lambda: PhysicalState(True, False, doors["lower"], doors["upper"]),
        is_door_open=lambda level: not doors[level],
        is_canonical_reset=unexpected_pose_check,
    )
    state = SemanticState(stored_level="lower")
    observer = WR03PhysicalObserver(state, dwell_steps=1)
    observer.reset(sample(env))
    doors.update(lower=True, upper=True)
    env.timestep = 1
    observer.observe(sample(env))
    assert state.storage_hidden
    assert not state.canonical_reset_complete

    doors.update(lower=False, upper=False)
    env.timestep = 2
    observer.observe(sample(env))
    assert state.retrieval_level_opened == "lower"


@pytest.fixture
def table_placement_env() -> tuple[SimpleNamespace, dict[str, bool]]:
    node = _function_node(
        WR03_ROOT / "environment.py",
        "WR03StoreAndRetrieveRememberedCan",
        "is_can_on_table",
    )
    namespace = {"CAN_ID": CAN_ID}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "environment.py", "exec"), namespace)

    contacts = {"counter": True, "gripper": False}
    counter, gripper, can = object(), object(), object()

    def check_contact(first: object, second: object) -> bool:
        if first is can and second is counter:
            return contacts["counter"]
        if first is gripper and second is can:
            return contacts["gripper"]
        raise AssertionError("placement must inspect the task counter and gripper")

    def unexpected_pose_check(*args: object, **kwargs: object) -> None:
        raise AssertionError("policy placement must not constrain position or orientation")

    env = SimpleNamespace(
        objects={CAN_ID: can},
        counter=counter,
        robots=[SimpleNamespace(gripper={"right": gripper})],
        check_contact=check_contact,
        table_target_position=unexpected_pose_check,
        get_object_world_position=unexpected_pose_check,
        get_object_world_quaternion_xyzw=unexpected_pose_check,
    )
    env.is_can_on_table = MethodType(namespace["is_can_on_table"], env)
    return env, contacts


@pytest.mark.parametrize(
    ("counter_contact", "gripper_contact", "expected"),
    ((True, False, True), (True, True, False), (False, False, False), (False, True, False)),
)
def test_wr03_table_placement_requires_counter_contact_and_release_only(
    table_placement_env, counter_contact, gripper_contact, expected,
) -> None:
    env, contacts = table_placement_env
    contacts.update(counter=counter_contact, gripper=gripper_contact)

    # No sampled target, object pose, or EEF position is available. Released
    # contact suffices, including far from the expert target or on its side.
    assert env.is_can_on_table() is expected


@pytest.mark.parametrize("failed_query", ("counter", "gripper"))
def test_wr03_table_placement_rejects_unavailable_contact_data(
    table_placement_env, failed_query,
) -> None:
    env, _ = table_placement_env
    original_check = env.check_contact

    def check_contact(first: object, second: object) -> bool:
        query = "counter" if second is env.counter else "gripper"
        if query == failed_query:
            raise RuntimeError("contact data unavailable")
        return original_check(first, second)

    env.check_contact = check_contact
    assert not env.is_can_on_table()
