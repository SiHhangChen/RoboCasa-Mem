"""Frozen low-level expert trajectories for stage-conditioned evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from bench.runtime.data_contract import ACTION_DIM, ACTION_SCHEMA, validate_action


EXPERT_PREFIX_SCHEMA = "membench.expert_prefix/v1"


@dataclass(frozen=True, slots=True)
class ExpertPrefixArtifact:
    task_id: str
    seed: int
    episode_identity: str
    expert_plan_version: str
    stage_boundary_steps: Mapping[str, int]
    actions: np.ndarray
    success: bool
    stage_order: tuple[str, ...] | None = None
    failure_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = _actions(self.actions)
        boundaries = {str(key): int(value) for key, value in self.stage_boundary_steps.items()}
        stage_order = tuple(
            str(value)
            for value in (
                tuple(boundaries) if self.stage_order is None else self.stage_order
            )
        )
        if len(stage_order) != len(set(stage_order)) or set(stage_order) != set(boundaries):
            raise ValueError("expert prefix stage_order must match stage boundaries")
        if any(step < 0 or step > int(actions.shape[0]) for step in boundaries.values()):
            raise ValueError("expert prefix stage boundaries must index recorded actions")
        ordered = [boundaries[stage_id] for stage_id in stage_order]
        if ordered != sorted(ordered):
            raise ValueError("expert prefix stage boundaries must be monotonic")
        object.__setattr__(self, "task_id", str(self.task_id))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "episode_identity", str(self.episode_identity))
        object.__setattr__(self, "expert_plan_version", str(self.expert_plan_version))
        object.__setattr__(self, "stage_boundary_steps", boundaries)
        object.__setattr__(self, "stage_order", stage_order)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics or {}))
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    def prefix_for_target(
        self,
        stage_order: Sequence[str],
        target_stage: str,
    ) -> np.ndarray:
        order = tuple(str(value) for value in stage_order)
        target = str(target_stage)
        try:
            target_index = order.index(target)
        except ValueError as exc:
            raise KeyError(f"unknown target stage {target!r}; expected one of {order}") from exc
        if target_index == 0:
            return np.empty((0, ACTION_DIM), dtype=np.float32)
        prefix_stage = order[target_index - 1]
        try:
            boundary = self.stage_boundary_steps[prefix_stage]
        except KeyError as exc:
            raise ValueError(
                f"expert artifact has no boundary for required prefix stage {prefix_stage!r}"
            ) from exc
        return np.ascontiguousarray(self.actions[:boundary], dtype=np.float32)

    def manifest(self) -> dict[str, Any]:
        action_hash = hashlib.sha256(self.actions.tobytes(order="C")).hexdigest()
        return {
            "schema_version": EXPERT_PREFIX_SCHEMA,
            "task_id": self.task_id,
            "seed": self.seed,
            "episode_identity": self.episode_identity,
            "expert_plan_version": self.expert_plan_version,
            "action_schema": ACTION_SCHEMA,
            "action_count": int(self.actions.shape[0]),
            "action_sha256": action_hash,
            "stage_order": list(self.stage_order or ()),
            "stage_boundary_steps": dict(self.stage_boundary_steps),
            "success": bool(self.success),
            "failure_reason": self.failure_reason,
            "diagnostics": _jsonable(self.diagnostics),
            "provenance": _jsonable(self.provenance),
        }


def capture_expert_trajectory(
    runtime: Any,
    plan: Any,
    *,
    stage_ids: Sequence[str],
    stage_state_builder: Callable[[Any], Any],
    stage_completed: Callable[[Any, str], bool],
    stage_boundary_builder: (
        Callable[[Any, Any, Mapping[str, int]], Mapping[str, int]] | None
    ) = None,
    provenance: Mapping[str, Any] | None = None,
) -> ExpertPrefixArtifact:
    """Run the existing expert while tapping every low-level ``env.step``."""

    initialized = getattr(runtime, "initialized", None)
    runner = getattr(initialized, "runner", None)
    env = getattr(runtime, "env", None)
    if runner is None or not callable(getattr(runner, "run", None)):
        raise TypeError("expert prefix capture requires an expert runtime with StageRunner")
    if env is None or not callable(getattr(env, "step", None)):
        raise TypeError("expert prefix capture requires a runtime exposing env.step")

    actions: list[np.ndarray] = []
    boundaries: dict[str, int] = {}
    expert_stage_end_steps: dict[str, int] = {}
    original_step = env.step

    def tapped_step(action: Any) -> Any:
        action_value = validate_action(action)
        result = original_step(action_value)
        actions.append(action_value.copy())
        state = stage_state_builder(runtime)
        for stage_id in stage_ids:
            if stage_id not in boundaries and stage_completed(state, stage_id):
                boundaries[str(stage_id)] = len(actions)
        return result

    def stage_finished(index: int, stage: Any, result: Any) -> None:
        if not bool(getattr(result, "success", False)):
            return
        stage_name = str(getattr(stage, "name", "")).strip() or f"stage_{index}"
        if stage_name in expert_stage_end_steps:
            raise ValueError(f"duplicate expert stage name during capture: {stage_name!r}")
        expert_stage_end_steps[stage_name] = len(actions)

    env.step = tapped_step
    expert_result = None
    try:
        expert_result = runner.run(plan, on_stage_end=stage_finished)
    finally:
        env.step = original_step

    if stage_boundary_builder is not None:
        resolved = {
            str(stage_id): int(step)
            for stage_id, step in stage_boundary_builder(
                runtime,
                plan,
                expert_stage_end_steps,
            ).items()
        }
        unknown = sorted(set(resolved) - {str(stage_id) for stage_id in stage_ids})
        if unknown:
            raise ValueError(
                f"expert stage boundary builder returned unknown stages: {unknown}"
            )
        invalid = {
            stage_id: step
            for stage_id, step in resolved.items()
            if step < 0 or step > len(actions)
        }
        if invalid:
            raise ValueError(f"expert stage boundary builder returned invalid steps: {invalid}")
        boundaries.update(resolved)

    diagnostics_obj = runtime.diagnostics()
    diagnostics = {
        "success": bool(diagnostics_obj.success),
        "terminal": bool(diagnostics_obj.terminal),
        "failure_code": diagnostics_obj.failure_code,
        **dict(diagnostics_obj.details or {}),
        "expert_stage_end_steps": dict(expert_stage_end_steps),
    }
    success = bool(expert_result.success and diagnostics_obj.success)
    failure_reason = None if success else str(
        diagnostics_obj.failure_code or "expert_plan_failed"
    )
    return ExpertPrefixArtifact(
        task_id=str(runtime.episode.task_id),
        seed=int(runtime.episode.seed),
        episode_identity=str(runtime.episode.identity),
        expert_plan_version=str(getattr(plan, "version", "unknown")),
        stage_boundary_steps=boundaries,
        actions=np.stack(actions) if actions else np.empty((0, ACTION_DIM), dtype=np.float32),
        success=success,
        stage_order=tuple(
            sorted(
                boundaries,
                key=lambda stage_id: (
                    boundaries[stage_id],
                    tuple(str(value) for value in stage_ids).index(stage_id),
                ),
            )
        ),
        failure_reason=failure_reason,
        diagnostics=diagnostics,
        provenance=dict(provenance or {}),
    )


def write_expert_prefix(root: str | Path, artifact: ExpertPrefixArtifact) -> Path:
    episode_dir = Path(root).expanduser().resolve() / f"seed_{artifact.seed:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    actions_path = episode_dir / "actions.npz"
    fd, temp_name = tempfile.mkstemp(prefix=".actions.", suffix=".npz", dir=episode_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, actions=artifact.actions)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, actions_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    _atomic_json(episode_dir / "manifest.json", artifact.manifest())
    return episode_dir


def load_expert_prefix(root: str | Path, seed: int) -> ExpertPrefixArtifact:
    episode_dir = Path(root).expanduser().resolve() / f"seed_{int(seed):06d}"
    manifest = json.loads((episode_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPERT_PREFIX_SCHEMA:
        raise ValueError(f"unsupported expert prefix schema: {manifest.get('schema_version')!r}")
    with np.load(episode_dir / "actions.npz") as payload:
        actions = _actions(payload["actions"])
    action_hash = hashlib.sha256(actions.tobytes(order="C")).hexdigest()
    if action_hash != manifest.get("action_sha256"):
        raise ValueError(f"expert prefix action hash mismatch: {episode_dir}")
    if int(actions.shape[0]) != int(manifest.get("action_count", -1)):
        raise ValueError(f"expert prefix action count mismatch: {episode_dir}")
    return ExpertPrefixArtifact(
        task_id=str(manifest["task_id"]),
        seed=int(manifest["seed"]),
        episode_identity=str(manifest["episode_identity"]),
        expert_plan_version=str(manifest["expert_plan_version"]),
        stage_boundary_steps=dict(manifest["stage_boundary_steps"]),
        actions=actions,
        success=bool(manifest["success"]),
        stage_order=(
            None
            if manifest.get("stage_order") is None
            else tuple(str(value) for value in manifest["stage_order"])
        ),
        failure_reason=manifest.get("failure_reason"),
        diagnostics=dict(manifest.get("diagnostics") or {}),
        provenance=dict(manifest.get("provenance") or {}),
    )


def _actions(value: Any) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] <= 0:
        raise ValueError(f"expert actions must have shape (N, {ACTION_DIM}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("expert actions contain non-finite values")
    return np.ascontiguousarray(actions)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "EXPERT_PREFIX_SCHEMA",
    "ExpertPrefixArtifact",
    "capture_expert_trajectory",
    "load_expert_prefix",
    "write_expert_prefix",
]
