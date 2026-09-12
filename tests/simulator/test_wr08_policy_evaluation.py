"""Check both carton assets against real tray contacts and runtime hard stops."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.simulator
@pytest.mark.parametrize("scenario", ("contacts", "termination"))
def test_wr08_policy_physics_and_execution_stop(scenario):
    code = """
from dataclasses import replace
import sys
import numpy as np
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wr08.domain.contracts import ITEM_IDS, TARGET_SLOTS

episode = TaskCatalog.resolve(EpisodeRequest(task_id="WR-08", seed=0, split="evaluation"))
api = resolve_builtin_task_plugin("WR-08").load_runtime_api()
runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False)
env = runtime.env

def place_fixture(item, slot):
    # Set up a contact test fixture; this deliberately does not claim a pickup.
    plate = env.target_plate_name(slot)
    position = env._obj_pos(plate).copy()
    position[2] += float(env.objects[plate].top_offset[-1])
    position[2] -= float(env.objects[item].bottom_offset[-1])
    env.set_object_pose(item, position, 0.0)
    for _ in range(250):
        env.sim.step()
    env.sim.forward()
    return position

def seed_pick_history(item):
    # Pickup has separate contact/lift unit coverage. Inject only its history
    # here so placement and stopping still exercise genuine MuJoCo contacts.
    env.timestep += 1
    current = env.evaluation_snapshot()
    env._wr08_evaluator.observe(replace(
        current,
        item_picked_by_item={key: key == item for key in ITEM_IDS},
        item_placed_by_item={**current.item_placed_by_item, item: False},
        item_matched_by_item={**current.item_matched_by_item, item: False},
        released_by_item={**current.released_by_item, item: False},
    ))

try:
    assert not env.require_expert_protocol
    assert env._physical_observer is None
    assert runtime.initialized.runner is None
    assert not runtime.diagnostics().success
    if sys.argv[1] == "contacts":
        initial_qpos = {
            item: env.sim.data.get_joint_qpos(env.objects[item].joints[0]).copy()
            for item in ITEM_IDS
        }
        # A tiny expert radius must have no effect on contact-based evaluation.
        env.plate_threshold = 1e-9
        for item in ITEM_IDS:
            for slot in TARGET_SLOTS:
                position = place_fixture(item, slot)
                snapshot = env.evaluation_snapshot()
                assert snapshot.item_placed_by_item[item], snapshot.details
                assert snapshot.target_slot_attempted_by_item[item] == slot, snapshot.details
                assert snapshot.item_matched_by_item[item] == (slot == env.target_slot_for_item(item))
                evidence = snapshot.details["physical"]
                assert "reason" not in evidence["grasp_by_item"][item], evidence
                assert "reason" not in evidence["support_by_item"][item], evidence
                contacts = evidence["support_by_item"][item]["plate_support"]
                assert contacts == {
                    name: bool(env.check_contact(env.objects[item], env.objects[name]))
                    for name in contacts
                }
                for _ in range(3):
                    env.timestep += 1
                    assert not runtime.diagnostics().terminal, "placement bypassed pickup history"

                position[2] += 0.30
                env.set_object_pose(item, position, 0.0)
                airborne = env.evaluation_snapshot()
                assert not airborne.item_placed_by_item[item], airborne.details
                assert not airborne.item_matched_by_item[item]
            env.sim.data.set_joint_qpos(env.objects[item].joints[0], initial_qpos[item])
            env.sim.data.set_joint_qvel(env.objects[item].joints[0], np.zeros(6))
            env.sim.forward()
        assert not any(env._semantic_state.item_picked_by_item.values())
        assert not any(env._semantic_state.source_color_memory_latched_by_item.values())
    else:
        order = ITEM_IDS[::-1]
        for wrong_item in (None, *order):
            runtime.reset()
            assert not env.require_expert_protocol
            assert not runtime.diagnostics().stop_execution
            for item in ITEM_IDS:
                assert np.allclose(env._evaluation_initial_item_positions[item], env._obj_pos(item))
            action = np.zeros(13, dtype=np.float32)
            for index, item in enumerate(order):
                seed_pick_history(item)
                target = env.target_slot_for_item(item)
                if item == wrong_item:
                    occupied = {env.target_slot_for_item(previous) for previous in order[:index]}
                    target = next(slot for slot in TARGET_SLOTS if slot != target and slot not in occupied)
                place_fixture(item, target)
                assert not runtime.step(action).diagnostics.stop_execution
                final = runtime.step(action)
                stop = item == wrong_item or index == 1
                assert final.diagnostics.stop_execution == stop, final.diagnostics
                if stop:
                    break
            expected_success = wrong_item is None
            assert final.terminal and final.diagnostics.stop_execution
            assert final.diagnostics.success == expected_success, final.diagnostics
            assert final.diagnostics.failure_code == (None if expected_success else "wrong_target_placement")
            assert final.diagnostics.details["evaluation"]["terminal_step"] == env.timestep
            if wrong_item is not None:
                assert final.diagnostics.details["evaluation"]["failure_item"] == wrong_item

            stopped_time, stopped_step = float(env.sim.data.time), env.timestep
            stopped_qpos = env.sim.data.qpos.copy()
            try:
                runtime.step(action)
            except RuntimeError as exc:
                assert "reset before stepping" in str(exc)
            else:
                raise AssertionError("runtime accepted another action after WR-08 termination")
            assert float(env.sim.data.time) == stopped_time
            assert env.timestep == stopped_step
            assert np.array_equal(env.sim.data.qpos, stopped_qpos)
            assert runtime.diagnostics() == final.diagnostics
finally:
    runtime.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, scenario], cwd=ROOT,
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
