"""Action-only replay validation for recorded native episodes.

A recorded expert episode is considered qualified only when every transition
comes from the recorded action vector and the task reaches its terminal success
state under the *current* environment/task semantics.  This module is shared by
the recording pipeline and offline dataset audits so both paths use the same
definition of "qualified".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bench.runtime import RoboCasaRuntimeFactory
from bench.runtime.contracts import RuntimeDescriptor, TaskRuntimeAPI
from bench.runtime.data_contract import ACTION_DIM

REPLAY_MODE = "recorded_actions_action_only_physics"


@dataclass(frozen=True, slots=True)
class ActionReplayValidation:
    """Result of replaying one recorded native episode action by action."""

    success: bool
    recorded_steps: int
    consumed_steps: int
    success_step: int | None = None
    failure_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    mode: str = REPLAY_MODE

    @property
    def qualified(self) -> bool:
        return self.success

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_recorded_actions(trajectory_path: str | Path) -> np.ndarray:
    """Read the single native action stream ``data/demo_1/actions``."""

    import h5py

    path = Path(trajectory_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing trajectory file: {path}")
    with h5py.File(path, "r") as handle:
        dataset = handle.get("data/demo_1/actions")
        if dataset is None:
            raise ValueError(f"{path} does not contain data/demo_1/actions")
        actions = np.asarray(dataset, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"invalid recorded action shape {actions.shape}, expected (N, {ACTION_DIM})"
        )
    if int(actions.shape[0]) <= 0:
        raise ValueError("recorded episode contains no actions")
    if not np.all(np.isfinite(actions)):
        raise ValueError("recorded actions contain non-finite values")
    return np.ascontiguousarray(actions, dtype=np.float32)


def replay_recorded_actions(
    descriptor: RuntimeDescriptor,
    runtime_api: TaskRuntimeAPI,
    trajectory_path: str | Path,
    *,
    max_steps: int | None = None,
) -> ActionReplayValidation:
    """Replay recorded actions in a fresh non-expert runtime.

    The fresh runtime is built from the exact descriptor used at recording time,
    so episode identity, scene seed, environment kwargs, and task semantics are
    identical.  Success is decided only by the final transition diagnostics; an
    early ``success`` that is not preserved at the end of the action stream is
    treated as a failure.
    """

    actions = load_recorded_actions(trajectory_path)
    recorded_steps = int(actions.shape[0])
    if max_steps is not None:
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        actions = actions[: min(recorded_steps, int(max_steps))]

    runtime = RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
        descriptor, expert=False
    )
    consumed = 0
    success_step = None
    diagnostics: dict[str, Any] = {}
    try:
        for step, action in enumerate(actions, start=1):
            transition = runtime.step(action)
            consumed = step
            diagnostics = dict(transition.diagnostics.details or {})
            if bool(transition.diagnostics.success) and success_step is None:
                success_step = step
        success = bool(diagnostics.get("success") is True)
        return ActionReplayValidation(
            success=success,
            recorded_steps=recorded_steps,
            consumed_steps=int(consumed),
            success_step=success_step,
            failure_reason=(
                None
                if success
                else (
                    "recorded_actions_exhausted"
                    if consumed == recorded_steps
                    else "replay_stopped_early"
                )
            ),
            diagnostics=diagnostics,
        )
    finally:
        runtime.close()


__all__ = [
    "ActionReplayValidation",
    "REPLAY_MODE",
    "load_recorded_actions",
    "replay_recorded_actions",
]
