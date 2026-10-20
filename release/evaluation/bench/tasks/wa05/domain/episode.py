"""Deterministic, simulator-free WA-05 episode decisions."""

from __future__ import annotations
from typing import Any
from bench.tasks.wa05.domain.contracts import DISPLAY_NAME, ENV_NAME, TASK_ID, TEMPERATURES, EpisodePlan, SceneConfig, TaskDefinition

def validate_temperature(value: Any) -> str:
    temperature = str(value).strip().lower()
    if temperature not in TEMPERATURES:
        raise ValueError(f"invalid WA-05 temperature: {value!r}")
    return temperature

def expert_temperature_for_seed(seed: int) -> str:
    return TEMPERATURES[int(seed) & 1]

def resolve_task_definition(**values: Any) -> TaskDefinition:
    task_id = str(values.get("id", TASK_ID))
    if task_id != TASK_ID:
        raise ValueError(f"WA-05 config must use task.id={TASK_ID!r}")
    return TaskDefinition(task_id=task_id, display_name=str(values.get("display_name", DISPLAY_NAME)), description=str(values.get("description", "Fill one cup with either temperature, stage it, then press the matching button.")), success_conditions=tuple(str(item) for item in values.get("success_conditions", ("single_cup_filled", "single_temperature_fill", "cup_staged_before_answer", "physical_answer_button_pressed", "answer_matches_filled_temperature"))))

def resolve_scene_config(*, seed: int, env_split: str, **values: Any) -> SceneConfig:
    fixed_scene = bool(values.get("fixed_scene", True))
    return SceneConfig(env_name=str(values.get("env_name", ENV_NAME)), env_split=str(env_split), layout_id=(15 if fixed_scene else _optional_int(values.get("layout_id"))), style_id=(34 if fixed_scene else _optional_int(values.get("style_id"))), scene_seed=int(values.get("scene_seed", 0)) if fixed_scene else int(seed), fixed_scene=fixed_scene)

def resolve_episode_plan(*, seed: int, **values: Any) -> EpisodePlan:
    configured = values.get("expert_fill_temperature")
    temperature = expert_temperature_for_seed(seed) if configured is None or str(configured).strip().lower() in {"", "none", "random"} else validate_temperature(configured)
    return EpisodePlan(seed=int(seed), cup_id="cup_A", expert_fill_temperature=temperature, service_slot="slot_1", max_horizon=int(values.get("max_horizon", 4000)))

def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)

__all__ = ["expert_temperature_for_seed", "resolve_episode_plan", "resolve_scene_config", "resolve_task_definition", "validate_temperature"]
