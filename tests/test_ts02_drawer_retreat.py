from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.manipulation import ManipulationRuntimeConfig
from bench.tasks.ts02.execution import handlers


@pytest.mark.parametrize("reopens", [False, "retreat", "inward"])
def test_release_precedes_horizontal_retreat_and_return(monkeypatch, reopens):
    calls = []
    fraction = [0.0]
    fixture = "cabinet_slot_1"
    env = SimpleNamespace(
        fixture_names=(fixture,), robots=[object()], closed_threshold=0.08,
        cabinet_for_slot=lambda name: SimpleNamespace(rot=np.pi),
        slot_open_fraction=lambda name: fraction[0],
    )
    pose = np.eye(4)
    pose[:3, 3] = [4.85, -2.157, 1.10033]
    monkeypatch.setattr(handlers, "matrix_from_site_pose", lambda *args: pose.copy())
    monkeypatch.setattr(handlers, "matrix_from_base_obs", lambda obs: np.eye(4))
    monkeypatch.setattr(handlers, "pose_from_matrix", lambda target: target)
    monkeypatch.setattr(handlers, "locked_base_post_step_callback", lambda env: None)
    original = ManipulationRuntimeConfig(planner_args={"preserve_ik_branch_stages": ["existing"]})
    executor = SimpleNamespace(config=original, obs={})

    def release(**kwargs):
        calls.append("release")
        assert kwargs["gripper_action"] == -1.0
        return {"obs": {}, "steps": kwargs["steps"]}

    def retreat(target, **kwargs):
        phase = "inward" if kwargs["stage"].endswith("_inward") else "retreat"
        calls.append(phase)
        if phase == "retreat":
            np.testing.assert_allclose(target[:3, 3], pose[:3, 3] + [0, 0.05, 0])
            assert executor.config.planner_args["preserve_ik_branch_stages"] == [
                "existing", kwargs["stage"],
            ]
        else:
            delta = -pose[:2, 3]
            np.testing.assert_allclose(target[:2, 3], pose[:2, 3] + delta / np.linalg.norm(delta) * 0.15)
            assert target[2, 3] == pose[2, 3]
            assert executor.config is original
        np.testing.assert_array_equal(target[:3, :3], pose[:3, :3])
        pose[:] = target
        if reopens == phase:
            fraction[0] = 0.75
        return {"pass": True}

    def arm_return(**kwargs):
        calls.append("return")
        return {"pass": True}

    executor.run = retreat
    controller = SimpleNamespace(
        env=env, runtime_config=original, obs={}, stage_executor=executor,
        refresh_obs=lambda: None, _sync_obs=lambda obs: None,
        driver=SimpleNamespace(current_arm_qpos=lambda: np.zeros(7), run_gripper=release),
        return_arm_to_regular_pose=arm_return,
    )
    result = handlers.retreat_and_return_arm_after_drawer(controller, fixture)
    expected = ["release", "retreat"]
    if reopens != "retreat":
        expected.append("inward")
    if not reopens:
        expected.append("return")
    assert calls == expected
    assert result["pass"] is (not reopens)
    assert executor.config is original
    assert original.planner_args == {"preserve_ik_branch_stages": ["existing"]}
