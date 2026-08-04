from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MIGRATED_TASK_IDS = (
    "TM01", "TM02", "TM03", "TS01", "TS02", "TS03", "TS04",
    "WA-01", "WA-02", "WR-02", "WR-03", "WR-04",
)


@pytest.mark.simulator
@pytest.mark.parametrize("task_id", MIGRATED_TASK_IDS)
def test_expert_runtime_consumes_resolved_episode(task_id: str) -> None:
    code = """
import json
import sys
from bench.runtime import RoboCasaRuntimeFactory
from bench.execution.runner import StageRunner
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest

task_id = sys.argv[1]
episode = TaskCatalog.resolve(EpisodeRequest(
    task_id=task_id,
    seed=0,
    allow_partial_calibration=(task_id == "TM03"),
))
api = resolve_builtin_task_plugin(task_id).load_runtime_api()
runtime = RoboCasaRuntimeFactory(api).create(episode, execute=False, expert=True)
try:
    print(json.dumps({
        "control_freq": runtime.env.control_freq,
        "camera_width": runtime.initialized.descriptor.sensors.width,
        "camera_height": runtime.initialized.descriptor.sensors.height,
        "cameras": list(runtime.initialized.descriptor.sensors.camera_names),
        "episode_identity": episode.identity,
        "observation_profile_id": runtime.initialized.descriptor.sensors.profile_id,
        "runner_type": type(runtime.initialized.runner).__name__,
        "has_navigation": runtime.initialized.runner.navigation is not None,
        "has_manipulation": runtime.initialized.runner.manipulation is not None,
    }, sort_keys=True))
finally:
    runtime.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, task_id],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["control_freq"] == 20
    assert (result["camera_height"], result["camera_width"]) == (224, 224)
    assert result["cameras"] == [
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ]
    assert result["episode_identity"].startswith("sha256:")
    assert result["observation_profile_id"] == "rgbd_v3"
    assert result["runner_type"] == "StageRunner"
    assert result["has_navigation"] is True
    assert result["has_manipulation"] is True
