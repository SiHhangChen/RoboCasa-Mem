"""Explicit diagnostic success is authoritative without a second API query."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.simulator
@pytest.mark.parametrize("task_id", ("WA-01", "WA-05", "WR-08"))
def test_real_policy_runtime_does_not_recheck_diagnostic_success(task_id):
    code = """
from dataclasses import replace
import sys
import numpy as np
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest

task_id = sys.argv[1]
episode = TaskCatalog.resolve(EpisodeRequest(task_id=task_id, seed=0, split="evaluation"))
api = resolve_builtin_task_plugin(task_id).load_runtime_api()
original_check = api.check_success
calls = []
def counted_check(*args):
    calls.append(args)
    return original_check(*args)
api = replace(api, check_success=counted_check)
runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False)
try:
    calls.clear()
    for _ in range(2):
        transition = runtime.step(np.zeros(13, dtype=np.float32))
        assert not calls, "runtime repeated a success query after explicit diagnostics"
        assert transition.diagnostics.success == original_check(runtime.initialized, transition.info)
    assert runtime.diagnostics() == transition.diagnostics
    assert not calls
finally:
    runtime.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, task_id],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
