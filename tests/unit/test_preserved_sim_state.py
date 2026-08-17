"""Regression tests for state-transparent planning probes."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from bench.execution.planning.sim_state import preserved_integration_state


def test_preserved_integration_state_restores_after_forward(monkeypatch) -> None:
    calls: list[str] = []
    stored = np.array([1.0, 2.0, 3.0])
    live = stored.copy()

    def state_size(model, spec):
        return 3

    def get_state(model, data, target, spec):
        target[:] = live

    def set_state(model, data, source, spec):
        calls.append("set")
        live[:] = source

    def forward(model, data):
        calls.append("forward")
        live[-1] = 99.0

    fake_mujoco = SimpleNamespace(
        mjtState=SimpleNamespace(mjSTATE_INTEGRATION=object()),
        mj_stateSize=state_size,
        mj_getState=get_state,
        mj_setState=set_state,
        mj_forward=forward,
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    sim = SimpleNamespace(
        model=SimpleNamespace(_model=object()),
        data=SimpleNamespace(_data=object()),
    )

    with preserved_integration_state(sim):
        live[:] = -1.0

    np.testing.assert_array_equal(live, stored)
    assert calls == ["set", "forward", "set"]
