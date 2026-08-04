"""Live TS01 grasp and saucepan placement geometry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bench.execution.planning import make_object_pick_target_poses
from bench.tasks.ts01.domain.contracts import RECEPTACLE_NAME, SUGAR_PLATE_NAME
from bench.tasks.ts01.domain.geometry import (
    SUGAR_RELEASE_REGION_NAMES,
    sugar_release_offset_is_inside,
)


DEFAULT_GRASP_YAW_OFFSETS_DEG = (0.0, 90.0, 180.0, 270.0)


def live_saucepan_anchor(env: Any) -> np.ndarray:
    if hasattr(env, "saucepan_place_pose"):
        return np.asarray(env.saucepan_place_pose(z_offset=0.0, xy_offset=(0.0, 0.0)), dtype=float)
    return np.asarray(env.sim.data.body_xpos[env.obj_body_id[RECEPTACLE_NAME]], dtype=float)


def live_saucepan_pose(env: Any) -> np.ndarray:
    """Read the saucepan body pose directly from the current MuJoCo state."""

    body_id = env.obj_body_id[RECEPTACLE_NAME]
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
    pose[:3, :3] = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    return pose


def saucepan_pose_guard(
    env: Any,
    reference_pose_world: np.ndarray,
    *,
    max_translation_m: float,
    max_tilt_deg: float,
) -> dict[str, Any]:
    """Check that the saucepan has not been pushed or tipped before release."""

    reference = np.asarray(reference_pose_world, dtype=float).reshape(4, 4)
    current = live_saucepan_pose(env)
    translation = float(np.linalg.norm(current[:3, 3] - reference[:3, 3]))
    reference_up = reference[:3, 2] / max(
        float(np.linalg.norm(reference[:3, 2])), 1e-12
    )
    current_up = current[:3, 2] / max(float(np.linalg.norm(current[:3, 2])), 1e-12)
    cosine = float(np.clip(np.dot(reference_up, current_up), -1.0, 1.0))
    tilt = float(np.rad2deg(np.arccos(cosine)))
    translation_limit = float(max_translation_m)
    tilt_limit = float(max_tilt_deg)
    finite = bool(np.isfinite(translation) and np.isfinite(tilt))
    passed = bool(
        finite
        and translation <= translation_limit
        and tilt <= tilt_limit
    )
    return {
        "pass": passed,
        "translation_m": translation,
        "max_translation_m": translation_limit,
        "tilt_deg": tilt,
        "max_tilt_deg": tilt_limit,
        "reference_position_world": reference[:3, 3].astype(float).tolist(),
        "current_position_world": current[:3, 3].astype(float).tolist(),
        "reason": None if passed else "saucepan_pose_changed_before_release",
    }


def build_saucepan_release_candidate_groups(
    anchor_world: Any,
    *,
    region_order: Any,
    release_regions_xy: Mapping[str, Any],
    yaw_offsets_deg: Any,
    max_radius_m: float,
) -> list[dict[str, Any]]:
    """Expand ordered regions into fixed point-major, yaw-minor candidates."""

    anchor = np.asarray(anchor_world, dtype=float).reshape(-1)
    if anchor.size != 3 or not np.all(np.isfinite(anchor)):
        raise ValueError("TS01 saucepan anchor must be a finite xyz position")
    order = tuple(str(value).strip().upper() for value in region_order)
    if len(order) != len(SUGAR_RELEASE_REGION_NAMES) or set(order) != set(
        SUGAR_RELEASE_REGION_NAMES
    ):
        raise ValueError(
            "TS01 release_region_order must be a permutation of "
            f"{SUGAR_RELEASE_REGION_NAMES}"
        )
    yaws = _float_sequence(yaw_offsets_deg)
    groups: list[dict[str, Any]] = []
    fixed_candidate_order = 0
    for fallback_rank, region_name in enumerate(order):
        raw_offsets = release_regions_xy.get(region_name)
        offsets = np.asarray(raw_offsets, dtype=float)
        if offsets.shape != (3, 2) or not np.all(np.isfinite(offsets)):
            raise ValueError(
                f"TS01 release region {region_name!r} must contain three finite XY points"
            )
        candidates = []
        for point_index, offset in enumerate(offsets):
            inside = sugar_release_offset_is_inside(
                offset,
                max_radius_m=float(max_radius_m),
            )
            for yaw_index, yaw in enumerate(yaws):
                release_xy_world = anchor[:2] + offset
                candidates.append(
                    {
                        "target_pos_world": anchor.astype(float).tolist(),
                        "release_offset_x": float(offset[0]),
                        "release_offset_y": float(offset[1]),
                        "placement_base_yaw_deg": float(yaw),
                        "release_target_inside_saucepan": bool(inside),
                        "metadata": {
                            "region_name": region_name,
                            "fallback_rank": int(fallback_rank),
                            "point_index": int(point_index),
                            "yaw_index": int(yaw_index),
                            "yaw_offset_deg": float(yaw),
                            "yaw_mode": "base_cardinal_topdown",
                            "release_offset_xy": offset.astype(float).tolist(),
                            "release_xy_world": release_xy_world.astype(float).tolist(),
                            "fixed_candidate_order": int(fixed_candidate_order),
                        },
                    }
                )
                fixed_candidate_order += 1
        groups.append(
            {
                "group_name": region_name,
                "fallback_rank": int(fallback_rank),
                "candidates": candidates,
            }
        )
    return groups


def build_pick_payload_selector(yaw_offsets_deg: Any = DEFAULT_GRASP_YAW_OFFSETS_DEG):
    defaults = _float_sequence(yaw_offsets_deg)

    def selector(request: dict[str, Any]) -> dict[str, Any] | None:
        name = str(request.get("object_name", ""))
        if name not in set(getattr(request.get("env"), "sugar_names", ())):
            return None
        kwargs = dict(request.get("kwargs") or {})
        config = request["pick_place_config"]
        offsets = _float_sequence(kwargs.get("grasp_yaw_offsets_deg", defaults))
        preferred = preferred_tray_grasp_yaw_offset_deg(request["env"], name)
        candidates: list[dict[str, Any]] = []
        reachable: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
        for index, yaw in enumerate(offsets):
            payload = make_object_pick_target_poses(
                request["env"],
                request["obs"],
                object_name=name,
                orientation_mode=str(kwargs.get("orientation_mode", config.orientation_mode)),
                pregrasp_height=float(kwargs.get("pregrasp_height", config.pregrasp_height)),
                grasp_clearance=float(kwargs.get("grasp_clearance", config.grasp_clearance)),
                lift_height=float(request.get("lift_height", config.lift_height)),
                grasp_yaw_offset_deg=float(yaw),
            )
            dry_sequence = request.get("dry_run_pick_sequence")
            validation = (
                dry_sequence(payload, object_name=name)
                if callable(dry_sequence)
                else request["dry_run_stage"](
                    payload["poses"]["pregrasp"]["base"],
                    stage="pregrasp",
                    gripper_action=-1.0,
                )
            )
            score = (_yaw_distance(float(yaw), preferred), *_plan_score(validation, index))
            record = {
                "index": index,
                "yaw_offset_deg": float(yaw),
                "preferred_yaw_offset_deg": preferred,
                "pass": bool(validation.get("pass", False)),
                "score": list(score),
            }
            candidates.append(record)
            if record["pass"]:
                reachable.append((score, payload, record))
        if not reachable:
            return {"payload": None, "selection": {"pass": False, "candidates": candidates}}
        score, payload, selected = min(reachable, key=lambda item: item[0])
        return {
            "payload": payload,
            "selection": {
                "pass": True,
                "mode": "four_direction_tray_side_preference",
                "selected_yaw_offset_deg": selected["yaw_offset_deg"],
                "selected_score": list(score),
                "candidates": candidates,
            },
        }

    return selector


def preferred_tray_grasp_yaw_offset_deg(env: Any, object_name: str) -> float:
    try:
        object_pos = np.asarray(env.sim.data.body_xpos[env.obj_body_id[object_name]], dtype=float)
        plate_id = env.obj_body_id[SUGAR_PLATE_NAME]
        plate_pos = np.asarray(env.sim.data.body_xpos[plate_id], dtype=float)
        plate_rot = np.asarray(env.sim.data.body_xmat[plate_id], dtype=float).reshape(3, 3)
        return 270.0 if float((plate_rot.T @ (object_pos - plate_pos))[0]) > 0.04 else 0.0
    except Exception:
        return 0.0


def _plan_score(validation: Mapping[str, Any], index: int) -> tuple[float, ...]:
    if not bool(validation.get("pass", False)):
        return (1.0e12, 1.0e12, float(index))
    stages = [item for item in validation.get("stages", ()) if isinstance(item, Mapping)]
    lengths = [float((item.get("plan") or {}).get("trajectory_len", 1.0e12)) for item in stages]
    return (max(lengths, default=1.0e12), sum(lengths), float(index))


def _yaw_distance(value: float, target: float) -> float:
    return abs((value - target + 180.0) % 360.0 - 180.0)


def _float_sequence(value: Any) -> tuple[float, ...]:
    raw = value.replace(";", ",").split(",") if isinstance(value, str) else value
    result = tuple(float(item) for item in raw)
    if not result:
        raise ValueError("TS01 grasp yaw offsets must not be empty")
    return result


__all__ = [
    "DEFAULT_GRASP_YAW_OFFSETS_DEG",
    "build_saucepan_release_candidate_groups",
    "build_pick_payload_selector",
    "live_saucepan_anchor",
    "live_saucepan_pose",
    "preferred_tray_grasp_yaw_offset_deg",
    "saucepan_pose_guard",
]
