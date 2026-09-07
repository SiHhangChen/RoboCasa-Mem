"""Deterministic, simulator-free WR-08 episode resolution."""

from __future__ import annotations

from typing import Any

from bench.tasks.episode_support import resolve_canonical_episode
from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode
from bench.tasks.wr08.config import DEFAULT_TASK_CONFIG_FILE, load_task_config
from bench.tasks.wr08.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.wr08.prompts import resolve_prompt_spec


def resolve_episode(request: EpisodeRequest | None = None, **kwargs: Any) -> ResolvedEpisode:
    if request is None:
        request = EpisodeRequest(task_id="WR-08", **kwargs)
    return resolve_canonical_episode(
        request,
        task_id="WR-08",
        task_version="2.0.0",
        default_task_config_file=DEFAULT_TASK_CONFIG_FILE,
        load_task_config=load_task_config,
        resolve_scene=lambda config, req: resolve_scene_config(
            seed=req.seed, env_split=req.env_split, **dict(config.scene)
        ),
        resolve_plan=lambda config, req: resolve_episode_plan(
            seed=req.seed, **{**dict(config.episode), **dict(req.options)}
        ),
        resolve_prompt=lambda plan, req: resolve_prompt_spec(variant=req.variant),
    )


__all__ = ["resolve_episode"]
