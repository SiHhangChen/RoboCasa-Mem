"""Deterministic, simulator-free WA-01 episode resolution."""

from __future__ import annotations

from typing import Any

from bench.tasks.episode_support import resolve_canonical_episode
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode
from bench.tasks.wa01.config import (
    DEFAULT_TASK_CONFIG_FILE,
    load_task_config,
    resolve_task_options,
)
from bench.tasks.wa01.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wa01.prompts import resolve_prompt_spec


def resolve_episode(request: EpisodeRequest | None = None, **kwargs: Any) -> ResolvedEpisode:
    if request is None:
        request = EpisodeRequest(task_id="WA-01", **kwargs)
    return resolve_canonical_episode(
        request,
        task_id="WA-01",
        task_version="1.0.0",
        default_task_config_file=DEFAULT_TASK_CONFIG_FILE,
        load_task_config=load_task_config,
        resolve_task_options=resolve_task_options,
        resolve_scene=lambda config, req: resolve_scene_config(
            seed=req.seed, env_split=req.env_split, **dict(config.scene)
        ),
        resolve_plan=lambda config, req: resolve_episode_plan(
            seed=req.seed, **{**dict(config.episode), **dict(req.options)}
        ),
        resolve_prompt=lambda plan, req: resolve_prompt_spec(
            sugar_target_mode=plan.sugar_target_mode,
            variant=req.variant,
        ),
    )


__all__ = ["resolve_episode"]
