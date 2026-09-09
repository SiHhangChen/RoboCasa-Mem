"""Bind WR-08 episode facts to live world geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.execution.planning import make_object_pick_target_poses, matrix_from_base_obs
from bench.tasks.wr08.domain.contracts import EpisodePlan, EpisodeTargets, ITEM_IDS

def build_pick_payload_selector():
    """Try the standard trajectory first, then a lower shelf-safe pick.

    The fallback is eligible only after a real collision-preflight rejection.
    It preserves the grasp center, yaw candidates and three-segment checks;
    both carton types use the same rule, with no seed-specific branches.
    """
    select_height = _build_fixed_height_pick_payload_selector()

    def selector(request: dict[str, Any]) -> dict[str, Any] | None:
        primary = select_height(request)
        if primary is None or primary.get("payload") is not None:
            return primary
        selection = primary.get("selection") or {}
        has_collision = any(
            bool((stage.get("collision_preflight") or {}).get("collision"))
            for candidate in selection.get("candidates", ())
            for stage in candidate.get("stages", ())
        )
        if not has_collision:
            return primary
        kwargs = dict(request.get("kwargs") or {})
        pregrasp = float(kwargs.get("pregrasp_height", request["pick_place_config"].pregrasp_height))
        lift = float(request["lift_height"])
        lower_pregrasp, lower_lift = min(pregrasp, 0.03), min(lift, 0.06)
        if (lower_pregrasp, lower_lift) == (pregrasp, lift):
            return primary
        fallback = select_height({
            **request,
            "kwargs": {**kwargs, "pregrasp_height": lower_pregrasp},
            "lift_height": lower_lift,
        })
        if fallback is None:
            return primary
        fallback["selection"]["clearance_fallback"] = {
            "reason": "standard_pick_rejected_by_collision_preflight",
            "pregrasp_height_m": lower_pregrasp,
            "lift_height_m": lower_lift,
            "primary_selection": selection,
        }
        return fallback

    return selector


def _build_fixed_height_pick_payload_selector():
    """Compare full pick plans without changing the live carton or robot.

    Like WR-07, targets follow the object's current orientation. Different
    top-down wrist directions can reach the same grasp center through very
    different arm paths. Planning checks reachability, not contact dynamics;
    shelf preflight and the executed pick must still pass collision and grasp
    checks. Planning alone does not establish a physical grasp or lift.
    Explicitly calibrated single-pose files retain the generic controller path.
    """

    def selector(request: dict[str, Any]) -> dict[str, Any] | None:
        name = str(request.get("object_name", ""))
        kwargs = dict(request.get("kwargs") or {})
        raw_yaws = kwargs.get("grasp_yaw_offsets_deg")
        if name not in ITEM_IDS or raw_yaws is None:
            return None
        if kwargs.get("manual_grasp_pose_file") or kwargs.get("grasp_pose_file"):
            return None
        yaws = np.asarray(raw_yaws, dtype=float)
        if yaws.ndim != 1 or yaws.size == 0 or not np.all(np.isfinite(yaws)):
            raise ValueError("WR-08 grasp_yaw_offsets_deg must be a nonempty finite list")
        config = request["pick_place_config"]
        orientation = str(kwargs.get("orientation_mode", config.orientation_mode))
        if orientation not in {"object-x-topdown", "object-y-topdown"}:
            raise ValueError("WR-08 multi-direction grasp requires a top-down orientation")
        dry_sequence = request["dry_run_pick_sequence"]
        records: list[dict[str, Any]] = []
        passing: list[tuple[tuple[int, int, int], dict[str, Any], dict[str, Any]]] = []
        for index, yaw in enumerate(yaws):
            payload = make_object_pick_target_poses(
                request["env"],
                request["obs"],
                object_name=name,
                orientation_mode=orientation,
                pregrasp_height=float(kwargs.get("pregrasp_height", config.pregrasp_height)),
                grasp_clearance=float(kwargs.get("grasp_clearance", config.grasp_clearance)),
                lift_height=float(request["lift_height"]),
                grasp_yaw_offset_deg=float(yaw),
                object_pos_world=kwargs.get("object_pos_world"),
                grasp_offset_x=float(kwargs.get("grasp_offset_x", 0.0)),
                grasp_offset_y=float(kwargs.get("grasp_offset_y", 0.0)),
                grasp_offset_z=float(kwargs.get("grasp_offset_z", 0.0)),
            )
            validation = dry_sequence(payload, object_name=name)
            stages = [
                {
                    "stage": stage.get("stage"),
                    "pass": bool(stage.get("pass", False)),
                    "status": (stage.get("plan") or {}).get("status"),
                    "trajectory_len": (stage.get("plan") or {}).get("trajectory_len"),
                    "failure_reason": stage.get("failure_reason"),
                    "collision_preflight": stage.get("collision_preflight"),
                }
                for stage in validation.get("stages", ())
                if isinstance(stage, Mapping)
            ]
            passed = bool(validation.get("pass", False)) and len(stages) == 3
            lengths = [int(stage["trajectory_len"] or 0) for stage in stages]
            score = (max(lengths, default=0), sum(lengths), index)
            record = {
                "index": index,
                "yaw_offset_deg": float(yaw),
                "pass": passed,
                "score": list(score) if passed else None,
                "stages": stages,
            }
            records.append(record)
            if passed:
                passing.append((score, payload, record))
        selection = {
            "mode": "wr08_object_relative_topdown_candidates",
            "validation_scope": "plan_reachability_and_shelf_collision_preflight",
            "score_order": ["max_stage_trajectory_len", "total_trajectory_len", "candidate_index"],
            "candidates": records,
            "pass": bool(passing),
        }
        if not passing:
            selection["reason"] = "no_reachable_topdown_grasp_candidate"
            return {"payload": None, "selection": selection}
        score, payload, chosen = min(passing, key=lambda entry: entry[0])
        selection.update({
            "selected_index": chosen["index"],
            "selected_yaw_offset_deg": chosen["yaw_offset_deg"],
            "selected_score": list(score),
        })
        return {"payload": payload, "selection": selection}

    return selector


def resolve_episode_targets(env: Any, episode: EpisodePlan) -> EpisodeTargets:
    return EpisodeTargets(
        target_plate_pose_by_item={
            item: _tuple3(env.target_plate_pose(episode.target_slot_by_color[episode.source_color_by_item[item]], item))
            for item in ITEM_IDS
        },
    )

def accessible_object_pos_world(obs: dict[str, Any], object_pos_world: Any, *, max_base_x: float) -> tuple[np.ndarray, dict[str, Any]]:
    base_world = matrix_from_base_obs(obs)
    object_h = np.ones(4, dtype=float)
    object_h[:3] = np.asarray(object_pos_world, dtype=float)[:3]
    object_base = np.linalg.inv(base_world) @ object_h
    adjusted = object_base.copy()
    applied = bool(adjusted[0] > float(max_base_x))
    if applied:
        adjusted[0] = float(max_base_x)
    world = base_world @ adjusted
    return world[:3], {"applied": applied, "max_base_x": float(max_base_x), "object_pos_base": object_base[:3].tolist(), "adjusted_object_pos_base": adjusted[:3].tolist()}

def _tuple3(value: Any) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        raise ValueError("WR-08 world pose must contain finite XYZ")
    return tuple(float(v) for v in arr[:3])

__all__ = ["build_pick_payload_selector", "accessible_object_pos_world", "resolve_episode_targets"]
