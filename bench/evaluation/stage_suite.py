"""Configuration for stage-conditioned evaluation campaigns."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from bench.evaluation.suite import EvalSuiteConfig, load_eval_suite


@dataclass(frozen=True, slots=True)
class StageTargetConfig:
    id: str
    source: str
    model_step_budget: int


@dataclass(frozen=True, slots=True)
class StageEvalSuiteConfig:
    stage_suite_version: str
    task_id: str
    base_suite: EvalSuiteConfig
    targets: tuple[StageTargetConfig, ...]
    expert_prefix_root: Path
    source_path: Path
    content_hash: str

    def target(self, stage_id: str) -> StageTargetConfig:
        for target in self.targets:
            if target.id == str(stage_id):
                return target
        raise KeyError(f"unknown target stage {stage_id!r}; expected {self.stage_ids}")

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(target.id for target in self.targets)

    @property
    def budgets(self) -> dict[str, int]:
        return {target.id: target.model_step_budget for target in self.targets}


def load_stage_eval_suite(path: str | Path) -> StageEvalSuiteConfig:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"stage evaluation suite must be a YAML map: {source}")
    allowed = {"stage_suite_version", "task_id", "base_suite", "expert_prefix_root", "stages"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown stage evaluation suite fields: {unknown}")
    base_path = _path(payload.get("base_suite"), source=source, relative_to_source=True)
    prefix_root = _path(payload.get("expert_prefix_root"), source=source, relative_to_source=False)
    base_suite = load_eval_suite(base_path)
    task_id = _string(payload.get("task_id"), "task_id")
    if task_id != base_suite.task_id:
        raise ValueError(
            f"stage suite task {task_id!r} does not match base suite {base_suite.task_id!r}"
        )
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stage suite stages must be a non-empty list")
    targets = tuple(_target(item) for item in raw_stages)
    ids = tuple(target.id for target in targets)
    if len(ids) != len(set(ids)):
        raise ValueError("stage suite stage ids must be unique")
    if targets[0].source != "main_rollout":
        raise ValueError("the first stage must use source=main_rollout")
    if any(target.source != "expert_prefix" for target in targets[1:]):
        raise ValueError("all stages after the first must use source=expert_prefix")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return StageEvalSuiteConfig(
        stage_suite_version=_string(payload.get("stage_suite_version"), "stage_suite_version"),
        task_id=task_id,
        base_suite=base_suite,
        targets=targets,
        expert_prefix_root=prefix_root,
        source_path=source,
        content_hash=f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}",
    )


def _target(value: Any) -> StageTargetConfig:
    if not isinstance(value, Mapping):
        raise ValueError("each stage suite stage must be a map")
    unknown = sorted(set(value) - {"id", "source", "model_step_budget"})
    if unknown:
        raise ValueError(f"unknown stage target fields: {unknown}")
    source = _string(value.get("source"), "stage.source")
    if source not in {"main_rollout", "expert_prefix"}:
        raise ValueError("stage.source must be main_rollout or expert_prefix")
    budget = value.get("model_step_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("stage.model_step_budget must be a positive integer")
    return StageTargetConfig(
        id=_string(value.get("id"), "stage.id"),
        source=source,
        model_step_budget=int(budget),
    )


def _path(value: Any, *, source: Path, relative_to_source: bool) -> Path:
    text = _string(value, "path")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = source.parent if relative_to_source else Path(__file__).resolve().parents[2]
    return (base / path).resolve()


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


__all__ = [
    "StageEvalSuiteConfig",
    "StageTargetConfig",
    "load_stage_eval_suite",
]
