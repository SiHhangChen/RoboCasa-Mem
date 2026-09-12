"""Exercise WR-07 policy observations against actual MuJoCo tray contacts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.simulator
def test_policy_placement_uses_real_contacts_and_cannot_skip_pickup():
    code = """
import numpy as np
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wr07.domain.contracts import ITEM_ID, SOURCE_SLOTS, TARGET_SLOTS
from bench.tasks.wr07.evaluation import WR07EvaluationSnapshot

episode = TaskCatalog.resolve(EpisodeRequest(task_id="WR-07", seed=0, split="evaluation"))
api = resolve_builtin_task_plugin("WR-07").load_runtime_api()
runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False)
try:
    env = runtime.env
    assert not env.require_expert_protocol
    assert runtime.initialized.runner is None
    assert not runtime.diagnostics().success
    initial = env.evaluation_snapshot()
    assert "reason" not in initial.details["physical"]["grasp"], initial.details
    assert "reason" not in initial.details["physical"]["support"], initial.details
    assert not initial.item_picked

    # This radius remains an expert-only setting. Even an extremely small
    # value must not affect policy placement based on actual contacts.
    env.plate_threshold = 1e-9
    expected = env.target_slot_for_source_color()
    for slot in TARGET_SLOTS:
        plate = env.target_plate_name(slot)
        center = env._obj_pos(plate).copy()
        position = center.copy()
        position[2] += float(env.objects[plate].top_offset[-1])
        position[2] -= float(env.objects[ITEM_ID].bottom_offset[-1])
        # Fixture setup, not a policy pickup: let the actual carton settle
        # onto each tray and inspect genuine MuJoCo contact data.
        env.set_object_pose(ITEM_ID, position, 0.0)
        env.sim.data.set_joint_qvel(env.objects[ITEM_ID].joints[0], np.zeros(6))
        for _ in range(250):
            env.sim.step()
        env.sim.forward()
        placed = env.evaluation_snapshot()
        assert placed.item_placed, placed.details
        assert placed.target_slot_attempted == slot, placed.details
        assert placed.target_color_matches == (slot == expected), placed.details
        assert not placed.source_color_memory_latched
        plates = [env.source_plate_name(source) for source in SOURCE_SLOTS]
        plates += [env.target_plate_name(target) for target in TARGET_SLOTS]
        assert placed.details["physical"]["support"]["plate_support"] == {
            name: bool(env.check_contact(env.objects[ITEM_ID], env.objects[name]))
            for name in plates
        }
        for _ in range(3):
            env.timestep += 1
            assert not env._check_success(), "placement cannot bypass pickup history"
            assert not runtime.diagnostics().success

        position[2] += 0.3
        env.set_object_pose(ITEM_ID, position, 0.0)
        airborne = env.evaluation_snapshot()
        assert not airborne.item_placed, airborne.details
        assert not airborne.target_color_matches

    runtime.reset()
    assert not env.require_expert_protocol
    assert not runtime.diagnostics().success
    assert np.allclose(env._evaluation_initial_item_position, env._obj_pos(ITEM_ID))
    assert not any(stage.qualified for stage in env._wr07_evaluator.result().stages.values())

    for slot in TARGET_SLOTS:
        runtime.reset()
        # Seed pickup history as a test fixture; this is not a policy pickup
        # rollout. Placement below still uses real contacts and runtime steps.
        env.timestep += 1
        env._wr07_evaluator.observe(WR07EvaluationSnapshot(
            step_id=env.timestep, item_picked=True,
        ))
        plate = env.target_plate_name(slot)
        position = env._obj_pos(plate).copy()
        position[2] += float(env.objects[plate].top_offset[-1])
        position[2] -= float(env.objects[ITEM_ID].bottom_offset[-1])
        env.set_object_pose(ITEM_ID, position, 0.0)
        env.sim.data.set_joint_qvel(env.objects[ITEM_ID].joints[0], np.zeros(6))
        for _ in range(250):
            env.sim.step()
        env.sim.forward()

        action = np.zeros(13, dtype=np.float32)
        assert not runtime.step(action).diagnostics.stop_execution
        final = runtime.step(action)
        matched = slot == env.target_slot_for_source_color()
        assert final.terminal and final.diagnostics.stop_execution, final.diagnostics
        assert final.diagnostics.success == matched, final.diagnostics
        assert final.diagnostics.failure_code == (None if matched else "wrong_target_placement")
        assert final.diagnostics.details["evaluation"]["terminal_step"] == env.timestep

        stopped_time = float(env.sim.data.time)
        stopped_step = env.timestep
        stopped_qpos = env.sim.data.qpos.copy()
        try:
            runtime.step(action)
        except RuntimeError as exc:
            assert "reset before stepping" in str(exc)
        else:
            raise AssertionError("runtime accepted an action after destination placement")
        assert env.timestep == stopped_step
        assert float(env.sim.data.time) == stopped_time
        assert np.array_equal(env.sim.data.qpos, stopped_qpos)
        assert runtime.diagnostics() == final.diagnostics
finally:
    runtime.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
