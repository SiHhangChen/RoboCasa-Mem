from types import SimpleNamespace

import pytest

from bench.execution.manipulation import ManipulationRuntimeConfig
from bench.tasks.tm01.execution.drawer_physics import (
    DrawerPushCloseConfig,
    _drawer_close_execution_config,
)


def test_close_slows_execution_and_restores_config():
    original = ManipulationRuntimeConfig()
    executor = SimpleNamespace(config=original)
    with _drawer_close_execution_config(executor, DrawerPushCloseConfig()):
        assert executor.config.max_joint_step == 0.01
        assert executor.config.settle_steps == 30
    assert executor.config is original


def test_close_preserves_more_conservative_settings_and_restores_on_error():
    original = ManipulationRuntimeConfig(max_joint_step=0.005, settle_steps=40)
    executor = SimpleNamespace(config=original)
    with pytest.raises(RuntimeError):
        with _drawer_close_execution_config(executor, DrawerPushCloseConfig()):
            assert executor.config.max_joint_step == 0.005
            assert executor.config.settle_steps == 40
            raise RuntimeError("execution failed")
    assert executor.config is original


@pytest.mark.parametrize("step", [0, -0.01, float("nan"), float("inf")])
def test_close_rejects_invalid_joint_step(step):
    original = ManipulationRuntimeConfig()
    executor = SimpleNamespace(config=original)
    with pytest.raises(ValueError):
        with _drawer_close_execution_config(
            executor, DrawerPushCloseConfig(max_joint_step=step)
        ):
            pass
    assert executor.config is original
