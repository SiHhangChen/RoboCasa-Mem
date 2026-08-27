from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from bench.tasks.tm02.config import load_task_config, resolve_action_config


ROOT = Path(__file__).resolve().parents[1]
TM02_ROOT = ROOT / "bench" / "tasks" / "tm02"


def _identifiers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    identifiers.update(
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    )
    return identifiers


def test_tm02_execution_has_no_object_lock_or_collision_mask_path() -> None:
    forbidden = {
        "install_held_item_follow_assist",
        "_wr01_hold_enabled",
        "_wr01_collision_state",
        "_neutralize_held_close_command",
        "_set_object_pose_matrix",
        "_disable_object_collisions",
        "_restore_object_collisions",
        "_zero_object_free_joint_velocity",
    }
    identifiers: set[str] = set()
    for path in sorted((TM02_ROOT / "execution").glob("*.py")):
        identifiers.update(_identifiers(path))
    assert identifiers.isdisjoint(forbidden)


def test_tm02_execution_does_not_replace_env_step() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((TM02_ROOT / "execution").glob("*.py"))
    )
    assert "env.step =" not in source


def test_tm02_rejects_collision_masking_configuration() -> None:
    operations = deepcopy(dict(load_task_config().operations))
    operations["can_pick"]["disable_collisions_during_transport"] = True
    config = load_task_config()
    object.__setattr__(config, "operations", operations)
    with pytest.raises(ValueError, match="collision.*forbidden"):
        resolve_action_config(config)


def test_tm02_task_config_has_physical_gripper_defaults() -> None:
    config = load_task_config()
    gripper = dict(config.manipulation["gripper"])
    assert float(gripper["force_limit"]) > 0.0
    assert tuple(float(value) for value in gripper["pad_friction"]) == (
        2.0,
        0.05,
        0.0001,
    )
    assert float(config.operations["can_pick"]["held_gripper_action"]) == 0.0


def test_tm02_leaves_place_station_before_turning_to_water_station() -> None:
    path = TM02_ROOT / "resources/navigation/manual_base_path_layout15_style34_seed0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = {int(item["index"]): item for item in payload["samples"]}
    return_route = payload["station_targets"]["water_button_station"]["routes"]["9"]

    assert return_route == [9, 10, 0]
    assert samples[9]["name"] == "can_place_station"
    assert samples[10]["name"] == "water_station_return_approach"
    assert samples[10]["xy"][0] > samples[9]["xy"][0]
    # The base turns around at the place station and then drives forward to
    # the sink. Reusing the place-station yaw here makes this segment a backup.
    assert abs(float(samples[10]["yaw"]) - float(samples[0]["yaw"])) < 0.02


def test_tm02_seed1_returns_to_cabinet_without_reversing_the_place_route() -> None:
    path = TM02_ROOT / "resources/navigation/manual_base_path_layout15_style34_seed0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    door_routes = payload["station_targets"]["cabinet_door_station"]["routes"]
    retrieval_routes = payload["station_targets"]["can_retrieval_station"]["routes"]

    expected = [9, 10, 0, 1, 2, 3, 4]
    assert door_routes["9"] == expected
    assert retrieval_routes["9"] == expected
    assert door_routes["9"] != [9, 8, 7, 6, 5, 4]

    samples = {int(item["index"]): item for item in payload["samples"]}
    for source_index, target_index in zip(expected, expected[1:]):
        source = samples[source_index]
        target = samples[target_index]
        dx = float(target["xy"][0]) - float(source["xy"][0])
        dy = float(target["xy"][1]) - float(source["xy"][1])
        if math.hypot(dx, dy) <= 0.12:
            continue
        forward_progress = (
            dx * math.cos(float(target["yaw"]))
            + dy * math.sin(float(target["yaw"]))
        )
        assert forward_progress > 0.0, (source_index, target_index)


def test_tm02_cabinet_to_sink_routes_skip_the_can_place_station() -> None:
    path = TM02_ROOT / "resources/navigation/manual_base_path_layout15_style34_seed0.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = {int(item["index"]): item for item in payload["samples"]}
    targets = payload["station_targets"]

    expected = [4, 5, 6, 11, 0]
    assert targets["water_button_station"]["routes"]["4"] == expected
    assert targets["completion_button_station"]["routes"]["4"] == expected
    assert 9 not in expected
    assert samples[11]["name"] == "sink_return_without_can_table_detour"

    for source_index, target_index in zip(expected, expected[1:]):
        source = samples[source_index]
        target = samples[target_index]
        dx = float(target["xy"][0]) - float(source["xy"][0])
        dy = float(target["xy"][1]) - float(source["xy"][1])
        forward_progress = (
            dx * math.cos(float(target["yaw"]))
            + dy * math.sin(float(target["yaw"]))
        )
        assert forward_progress > 0.0, (source_index, target_index)
