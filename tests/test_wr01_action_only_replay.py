from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from bench.tasks.wr01.config import load_task_config, resolve_action_config


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
