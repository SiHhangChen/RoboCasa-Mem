"""Resolve WX-02 domain targets into controller-facing world geometry."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.planning import (
    make_manual_object_pick_target_poses,
    matrix_from_base_obs,
)
from bench.tasks.wx02.domain.contracts import CUP_NAMES, EpisodePlan, EpisodeTargets


def resolve_episode_targets(env: Any, episode: EpisodePlan) -> EpisodeTargets:
    """Bind pure relative episode offsets to reset-time original world poses."""

    original_tool = env_pose(
        env,
        "original_tool_pose",
        fallback_object="cleaning_tool",
    )
    original_objects = {
        object_name: env_pose(
            env,
            f"original_{object_name}_pose",
            fallback_object=object_name,
        )
        for object_name in CUP_NAMES
    }
    if original_tool is None or any(
        pose is None for pose in original_objects.values()
    ):
        raise ValueError("WX-02 original poses are unavailable after initialization")
    resolved_originals = {
        name: np.asarray(pose, dtype=float) for name, pose in original_objects.items()
    }
    temporary_objects = {}
    for object_name, target_xy in episode.temporary_object_xy.items():
        temporary = resolved_originals[object_name].copy()
        temporary[:2] = np.asarray(target_xy, dtype=float)
        temporary_objects[object_name] = temporary
    staged_tool = np.asarray(
        [episode.staged_tool_xy[0], episode.staged_tool_xy[1], original_tool[2]],
        dtype=float,
    )
    return_region = np.asarray(original_tool, dtype=float).copy()
    return_region[:2] = np.asarray(
        env.cleaning_tool_return_region_center_xy,
        dtype=float,
    )
    return EpisodeTargets(
        original_tool_pose=_tuple3(original_tool),
        original_object_poses={
            name: _tuple3(pose) for name, pose in resolved_originals.items()
        },
        temporary_object_poses={
            name: _tuple3(pose) for name, pose in temporary_objects.items()
        },
        staged_tool_pose=_tuple3(staged_tool),
        post_wipe_tool_pose=_tuple3(staged_tool),
        cleaning_tool_return_region_pose=_tuple3(return_region),
        stain_center_poses={
            name: _tuple3(pose) for name, pose in resolved_originals.items()
        },
    )


def make_cup_pick_payload_selector(
    controller: Any,
    *,
    manual_candidates_file: str | Path,
    maximum_base_x: float,
    dry_run_only: bool,
) -> Any:
    """Select a reachable task-owned top-down single-wall cup grasp.

    Execute mode always plans against the live cup pose. Dry-run mode retains
    WX-02's bounded-reach proxy, but applies that proxy before evaluating the
    same task-owned grasp candidates.
    """

    candidate_source, candidates = _load_blocking_cup_grasp_candidates(
        manual_candidates_file
    )

    def selector(request: dict[str, Any]) -> dict[str, Any] | None:
        object_name = str(request.get("object_name", ""))
        if object_name not in CUP_NAMES:
            return None
        kwargs = dict(request.get("kwargs") or {})
        execute = bool(controller.runtime_config.execute)
        object_position = _object_position(
            request["env"],
            object_name,
            override=None if execute else kwargs.get("object_pos_world"),
        )
        adjusted = object_position
        adjustment = {"applied": False, "reason": "execute_true"}
        if not execute:
            adjusted, adjustment = accessible_object_pos_world(
                request["obs"],
                object_position,
                max_base_x=float(
                    kwargs.get("object_max_base_x", maximum_base_x)
                ),
            )
        config = request["pick_place_config"]
        grasp_strategy = str(
            kwargs.get("grasp_strategy", "single-wall-topdown")
        )
        if grasp_strategy != "single-wall-topdown":
            raise ValueError(
                "WX-02 cups require grasp_strategy='single-wall-topdown'"
            )
        reachable: list[
            tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]
        ] = []
        records: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            payload = make_manual_object_pick_target_poses(
                request["env"],
                request["obs"],
                object_name=object_name,
                grasp_pose_file=candidate_source,
                grasp_pose_payload=dict(candidate),
                pregrasp_distance=None,
                lift_height=float(
                    kwargs.get("lift_height", config.lift_height_for_object(is_mug=True))
                ),
                object_pos_world=adjusted,
            )
            dry_sequence = request.get("dry_run_pick_sequence")
            validation = (
                dry_sequence(payload, object_name=object_name)
                if callable(dry_sequence)
                else request["dry_run_stage"](
                    payload["poses"]["pregrasp"]["base"],
                    stage="pregrasp",
                    gripper_action=-1.0,
                )
            )
            score = _plan_score(validation, index=index)
            record = {
                "index": index,
                "name": str(candidate["template_name"]),
                "wall_azimuth_deg": float(candidate["wall_azimuth_deg"]),
                "pregrasp_base": payload["poses"]["pregrasp"]["base"],
                "pass": bool(validation.get("pass", False)),
                "score": list(score),
                "validation": _validation_summary(validation),
            }
            records.append(record)
            if record["pass"]:
                reachable.append((score, payload, record))
        if not reachable:
            return {
                "payload": None,
                "selection": {
                    "pass": False,
                    "mode": "wx02_single_wall_topdown",
                    "candidate_source": str(candidate_source),
                    "candidates": records,
                    "object_pos_world": object_position.tolist(),
                    "adjusted_object_pos_world": adjusted.tolist(),
                    "adjustment": adjustment,
                    "reason": "no_reachable_cup_single_wall_grasp",
                },
            }
        score, payload, selected = min(reachable, key=lambda item: item[0])
        return {
            "payload": payload,
            "selection": {
                "enabled": True,
                "mode": "wx02_single_wall_topdown",
                "dry_run_only": bool(dry_run_only),
                "candidate_source": str(candidate_source),
                "selected_name": selected["name"],
                "selected_wall_azimuth_deg": selected["wall_azimuth_deg"],
                "selected_score": list(score),
                "candidates": records,
                "object_pos_world": object_position.tolist(),
                "adjusted_object_pos_world": adjusted.tolist(),
                "adjustment": adjustment,
                "pass": True,
            },
        }

    return selector


def _load_blocking_cup_grasp_candidates(
    path: str | Path,
) -> tuple[Path, tuple[Mapping[str, Any], ...]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"WX-02 cup grasp resource has no candidates: {source}")
    if payload.get("grasp_strategy") != "single-wall-topdown":
        raise ValueError(
            f"WX-02 cup grasp resource must use single-wall-topdown: {source}"
        )
    pregrasp = payload.get("pregrasp_offset_tcp", [0.0, 0.0, -0.12])
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping) or not isinstance(
            raw.get("eef_site_in_object"), Mapping
        ):
            raise ValueError(f"invalid WX-02 cup grasp candidate {index} in {source}")
        wall_azimuth_deg = float(raw.get("wall_azimuth_deg", float("nan")))
        if not np.isfinite(wall_azimuth_deg):
            raise ValueError(
                f"WX-02 cup grasp candidate {index} has no finite wall azimuth"
            )
        result.append(
            {
                "schema_version": 2,
                "asset": payload.get("asset", "robocasa cup_5"),
                "object_name": "cup",
                "source_file": str(source),
                "pregrasp_offset_tcp": pregrasp,
                "template_name": str(raw.get("name", index)),
                "wall_azimuth_deg": wall_azimuth_deg,
                "selected": {"eef_site_in_object": dict(raw["eef_site_in_object"])},
            }
        )
    return source, tuple(result)


def _plan_score(validation: Mapping[str, Any], *, index: int) -> tuple[float, ...]:
    if not bool(validation.get("pass", False)):
        return (1.0e12, 1.0e12, float(index))
    stages = [
        item for item in validation.get("stages", ()) if isinstance(item, Mapping)
    ]
    lengths = [
        float((item.get("plan") or {}).get("trajectory_len", 1.0e12))
        for item in stages
    ]
    return max(lengths, default=1.0e12), sum(lengths), float(index)


def _validation_summary(validation: Mapping[str, Any]) -> dict[str, Any]:
    stages = [
        item for item in validation.get("stages", ()) if isinstance(item, Mapping)
    ]
    return {
        "pass": bool(validation.get("pass", False)),
        "reason": validation.get("reason"),
        "stages": [
            {
                "stage": item.get("stage"),
                "pass": bool(item.get("pass", False)),
                "reason": item.get("reason"),
                "plan_status": (item.get("plan") or {}).get("status"),
            }
            for item in stages
        ],
    }


def accessible_object_pos_world(
    obs: dict[str, Any],
    object_pos_world: Any,
    *,
    max_base_x: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp a planning-only target to a maximum forward Base reach."""

    base_world = matrix_from_base_obs(obs)
    world_base = np.linalg.inv(base_world)
    object_h = np.ones(4, dtype=float)
    object_h[:3] = np.asarray(object_pos_world, dtype=float)[:3]
    object_base = world_base @ object_h
    adjusted_base = object_base.copy()
    applied = bool(adjusted_base[0] > float(max_base_x))
    if applied:
        adjusted_base[0] = float(max_base_x)
    adjusted_world = base_world @ adjusted_base
    return adjusted_world[:3], {
        "applied": applied,
        "max_base_x": float(max_base_x),
        "object_pos_base": object_base[:3].tolist(),
        "adjusted_object_pos_base": adjusted_base[:3].tolist(),
    }


def env_pose(
    env: Any, attribute: str, *, fallback_object: str | None = None
) -> np.ndarray | None:
    """Read a finite XYZ environment attribute with an object-body fallback."""

    value = getattr(env, attribute, None)
    pose = _optional_pose3(value)
    if pose is not None:
        return pose
    if fallback_object is None:
        return None
    try:
        return np.asarray(
            env.sim.data.body_xpos[env.obj_body_id[fallback_object]],
            dtype=float,
        )[:3]
    except Exception:
        return None


def _object_position(env: Any, object_name: str, *, override: Any = None) -> np.ndarray:
    if override is not None:
        pose = _optional_pose3(override)
        if pose is not None:
            return pose
    return np.asarray(
        env.sim.data.body_xpos[env.obj_body_id[object_name]], dtype=float
    )[:3]


def _optional_pose3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return array[:3]


def _tuple3(value: Any) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0]), float(array[1]), float(array[2])


__all__ = [
    "accessible_object_pos_world",
    "env_pose",
    "make_cup_pick_payload_selector",
    "resolve_episode_targets",
]
