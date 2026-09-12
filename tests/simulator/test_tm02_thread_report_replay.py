"""Action-only regression against the two recorded TM02 completion orders."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PREFIX_ROOT = ROOT / "outputs/stage_prefixes/tm02_gr00t_r40_v1"


@pytest.mark.simulator
@pytest.mark.parametrize("seed", (1050, 1051))
def test_tm02_v2_preserves_physical_success_and_existing_boundaries(seed: int) -> None:
    if not (PREFIX_ROOT / f"seed_{seed:06d}/manifest.json").is_file():
        pytest.skip("requires the recorded TM02 v1 expert prefixes")
    code = r'''
from dataclasses import replace
from pathlib import Path
import json
import sys

from bench.evaluation.canonical_stage import resolve_suite_episode
from bench.evaluation.expert_prefix import load_expert_prefix
from bench.evaluation.suite import load_eval_suite
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import resolve_builtin_task_plugin
from bench.tasks.tm02.evaluation import evaluate_initialized_runtime

seed = int(sys.argv[1])
old = load_expert_prefix("outputs/stage_prefixes/tm02_gr00t_r40_v1", seed)
suite = load_eval_suite(Path("configs/eval/legacy_suites/tm02_gr00t_r40_v1.yaml"))
episode = resolve_suite_episode(suite, seed)
assert episode.identity == old.episode_identity
api = resolve_builtin_task_plugin("TM02").load_runtime_api()
descriptor = api.build_descriptor(episode, True)
# No policy consumes images in an action replay. Rendering is the only override.
descriptor = replace(descriptor, sensors=replace(
    descriptor.sensors, use_camera_observations=False, has_offscreen_renderer=False,
))
runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(descriptor, expert=False)
try:
    success_step = None
    for step, action in enumerate(old.actions, start=1):
        transition = runtime.step(action)
        diagnostics = transition.diagnostics
        assert diagnostics.failure_code is None, (step, diagnostics.failure_code)
        if diagnostics.success:
            success_step = step
            break
    result = evaluate_initialized_runtime(runtime)
    assert result.success
    assert success_step == old.stage_boundary_steps["success"]
    for name, expected_step in old.stage_boundary_steps.items():
        assert result.stages[name].qualified_step == expected_step, (name, result)
    recorded = old.diagnostics["evaluation"]["semantic_event_steps"]
    first_step = min(recorded["A2"], recorded["B2"])
    assert result.stages["one_thread_completed"].qualified_step == first_step
    print(json.dumps({"seed": seed, "success_step": success_step,
                      "stage_steps": result.to_diagnostics()["stage_qualified_steps"]}))
finally:
    runtime.close()
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(seed)], cwd=ROOT,
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["success_step"] == result["stage_steps"]["success"]
    assert len(result["stage_steps"]) == 5
