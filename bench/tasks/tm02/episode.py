"""Deterministic, simulator-free TM02 episode resolution."""

from __future__ import annotations

from typing import Any

from bench.tasks.contracts import EpisodeRequest, ResolvedEpisode
from bench.tasks.episode_support import resolve_canonical_episode
from bench.tasks.tm02.config import (
    CALIBRATION_RESOURCE_NAMES,
    DEFAULT_TASK_CONFIG_FILE,
    load_task_config,
    resolve_task_options,
)
from bench.tasks.tm02.domain.episode import resolve_episode_plan, resolve_scene_config
from bench.tasks.tm02.prompts import resolve_prompt_spec


def resolve_episode(request: EpisodeRequest | None = None, **kwargs: Any) -> ResolvedEpisode:
    if request is None:
        request = EpisodeRequest(task_id="TM-02", **kwargs)
    return resolve_canonical_episode(
        request,
        task_id="TM-02",
        task_version="1.0.0",
        default_task_config_file=DEFAULT_TASK_CONFIG_FILE,
        load_task_config=load_task_config,
        resolve_task_options=resolve_task_options,
        task_validated_resource_names=CALIBRATION_RESOURCE_NAMES,
        resolve_scene=lambda config, req: resolve_scene_config(
            env_split=req.env_split, **dict(config.scene)
        ),
        resolve_plan=lambda config, req: resolve_episode_plan(
            seed=req.seed, **{**dict(config.episode), **dict(req.options)}
        ),
        resolve_prompt=lambda plan, req: resolve_prompt_spec(variant=req.variant),
    )


__all__ = ["resolve_episode"]
