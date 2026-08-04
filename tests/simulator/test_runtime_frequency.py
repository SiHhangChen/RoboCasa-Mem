from __future__ import annotations

import pytest

from bench.runtime import CONTROL_FREQUENCY_HZ


@pytest.mark.simulator
def test_robocasa_environment_factory_forwards_explicit_control_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.env.robocasa_runtime as runtime

    captured = {}

    class Robosuite:
        @staticmethod
        def make(**kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(runtime, "bootstrap_local_robocasa_sources", lambda: None)
    monkeypatch.setattr(runtime, "register_bench_tasks", lambda: None)
    monkeypatch.setitem(__import__("sys").modules, "robosuite", Robosuite())

    runtime.make_robocasa_env(
        env_name="PilotTask",
        split="pretrain",
        seed=0,
        controller_config={},
    )

    assert captured["control_freq"] == CONTROL_FREQUENCY_HZ == 20
