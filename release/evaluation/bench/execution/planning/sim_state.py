"""State-transparent temporary MuJoCo probes.

Expert planning is allowed to inspect hypothetical robot configurations, but
those inspections must not change the subsequent action-driven dynamics.  The
RoboSuite compatibility ``MjSimState`` contains only time, qpos, and qvel;
MuJoCo's constraint warm-start and other integration fields must be preserved
as well for an action-only replay to reproduce the recorded rollout.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


@contextmanager
def preserved_integration_state(sim: Any) -> Iterator[None]:
    """Restore every MuJoCo integration field after a temporary probe.

    A forward pass is needed to rebuild derived poses after restoring qpos.
    It can overwrite solver state, so the integration snapshot is applied a
    second time after ``mj_forward``.  The next ``mj_step`` then starts from
    the exact state that existed before the probe.
    """

    try:
        import mujoco

        model = sim.model._model
        data = sim.data._data
    except (ImportError, AttributeError):
        # Compatibility fallback for non-native test doubles / older bindings.
        state = sim.get_state()
        try:
            yield
        finally:
            sim.set_state(state)
            sim.forward()
        return

    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    snapshot = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, snapshot, state_spec)
    try:
        yield
    finally:
        mujoco.mj_setState(model, data, snapshot, state_spec)
        mujoco.mj_forward(model, data)
        # mj_forward recomputes acceleration / constraint solver state.  Put
        # back the recorded warm-start and all user-controlled fields without
        # invalidating the derived position and velocity quantities.
        mujoco.mj_setState(model, data, snapshot, state_spec)


__all__ = ["preserved_integration_state"]
