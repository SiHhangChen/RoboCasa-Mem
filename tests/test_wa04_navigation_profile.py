from __future__ import annotations

import pytest

from bench.tasks.wa04.config import load_task_config
from bench.tasks.wa04.execution.navigation import build_navigation_config


def test_wa04_navigation_uses_task_owned_velocity_profile() -> None:
    task = load_task_config()
    config = build_navigation_config(
        dict(task.navigation),
        manual_base_path_file=task.path(task.resources["manual_base_path"]),
        station_base_pose_file=task.path(task.resources["station_poses"]),
    )

    assert config.base_velocity_profile_enabled is True
    assert config.base_profile_v_max == pytest.approx(0.75)
    assert config.base_profile_a_max == pytest.approx(1.8)
    assert config.base_profile_a_dec == pytest.approx(1.5)
    assert config.base_profile_w_max == pytest.approx(1.35)
    assert config.base_profile_alpha_max == pytest.approx(3.5)
    assert config.base_profile_alpha_dec == pytest.approx(3.0)
