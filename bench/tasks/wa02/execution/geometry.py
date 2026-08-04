"""Live WA-02 geometry: cup grasps, calibrated anchors, and world/base poses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.planning import (
    make_manual_object_pick_target_poses,
    make_object_pick_target_poses,
    matrix_from_base_obs,
    matrix_from_json_pose,
    pose_from_matrix,
)
from bench.tasks.wa02.domain.contracts import CUP_IDS, SERVICE_SLOTS


DEFAULT_CUP_GRASP_YAW_OFFSETS_DEG = (0.0, -45.0, 45.0, -90.0, 90.0)


def build_pick_payload_selector(
    yaw_offsets_deg: Any = DEFAULT_CUP_GRASP_YAW_OFFSETS_DEG,
    *,
    manual_candidates_file: str | Path | None = None,
):
    """Choose the lowest-cost reachable live cup grasp from fixed yaw options.

    The selector uses only current geometry and dry-run planning.  It does not
    access the episode's hot/cold labels, so a grasp direction cannot encode a
    temperature assignment.
    """

    defaults = _float_sequence(yaw_offsets_deg)
    manual_source, manual_candidates = _load_manual_candidates(manual_candidates_file)

    def selector(request: dict[str, Any]) -> dict[str, Any] | None:
        cup_id = str(request.get("object_name", ""))
        if cup_id not in CUP_IDS:
            return None
        kwargs = dict(request.get("kwargs") or {})
        config = request["pick_place_config"]
        orientation_mode = str(
            kwargs.get("orientation_mode", config.mug_orientation_mode)
        )
        if manual_candidates and orientation_mode in {"mug-body-side", "side-from-base"}:
            return _select_manual_cup_grasp(
                request,
                candidates=manual_candidates,
                source=manual_source,
            )
        offsets = _float_sequence(kwargs.get("grasp_yaw_offsets_deg", defaults))
        approach_sector = _parse_grasp_approach_sector(
            kwargs.get("grasp_approach_sector")
        )
        candidates: list[dict[str, Any]] = []
        reachable: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
        for index, yaw in enumerate(offsets):
            payload = make_object_pick_target_poses(
                request["env"],
                request["obs"],
                object_name=cup_id,
                orientation_mode=orientation_mode,
                pregrasp_height=float(
                    kwargs.get("pregrasp_height", config.pregrasp_height)
                ),
                grasp_clearance=float(
                    kwargs.get("grasp_clearance", config.grasp_clearance)
                ),
                lift_height=float(request.get("lift_height", config.lift_height)),
                grasp_yaw_offset_deg=float(yaw),
            )
            sector_check = _evaluate_grasp_approach_sector(payload, approach_sector)
            if sector_check is not None and not sector_check["pass"]:
                candidates.append(
                    {
                        "index": int(index),
                        "yaw_offset_deg": float(yaw),
                        "pass": False,
                        "score": [1.0e12, 1.0e12, float(index)],
                        "approach_sector": sector_check,
                        "failure_reason": "pregrasp_outside_allowed_world_sector",
                    }
                )
                continue
            dry_sequence = request.get("dry_run_pick_sequence")
            validation = (
                dry_sequence(payload, object_name=cup_id)
                if callable(dry_sequence)
                else request["dry_run_stage"](
                    payload["poses"]["pregrasp"]["base"],
                    stage="pregrasp",
                    gripper_action=-1.0,
                )
            )
            score = _plan_score(validation, index=index)
            record = {
                "index": int(index),
                "yaw_offset_deg": float(yaw),
                "pass": bool(validation.get("pass", False)),
                "score": list(score),
                "approach_sector": sector_check,
            }
            candidates.append(record)
            if record["pass"]:
                reachable.append((score, payload, record))
        if not reachable:
            return {
                "payload": None,
                "selection": {
                    "pass": False,
                    "mode": "live_multi_yaw_cup_grasp",
                    "grasp_approach_sector": approach_sector,
                    "candidates": candidates,
                    "reason": "no_reachable_cup_grasp_candidate",
                },
            }
        score, payload, selected = min(reachable, key=lambda item: item[0])
        return {
            "payload": payload,
            "selection": {
                "pass": True,
                "mode": "live_multi_yaw_cup_grasp",
                "grasp_approach_sector": approach_sector,
                "selected_yaw_offset_deg": selected["yaw_offset_deg"],
                "selected_score": list(score),
                "candidates": candidates,
            },
        }

    return selector


def _select_manual_cup_grasp(
    request: Mapping[str, Any],
    *,
    candidates: tuple[Mapping[str, Any], ...],
    source: Path | None,
) -> dict[str, Any]:
    cup_id = str(request["object_name"])
    kwargs = dict(request.get("kwargs") or {})
    config = request["pick_place_config"]
    dry_sequence = request.get("dry_run_pick_sequence")
    dry_stage = request["dry_run_stage"]
    approach_sector = _parse_grasp_approach_sector(
        kwargs.get("grasp_approach_sector")
    )
    reachable: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        payload = make_manual_object_pick_target_poses(
            request["env"],
            request["obs"],
            object_name=cup_id,
            grasp_pose_file=source,
            grasp_pose_payload=dict(candidate),
            pregrasp_distance=None,
            lift_height=float(
                kwargs.get("lift_height", config.lift_height_for_object(is_mug=True))
            ),
            object_pos_world=kwargs.get("object_pos_world"),
        )
        sector_check = _evaluate_grasp_approach_sector(payload, approach_sector)
        if sector_check is not None and not sector_check["pass"]:
            records.append(
                {
                    "index": int(index),
                    "name": str(candidate.get("template_name", index)),
                    "pass": False,
                    "score": [1.0e12, 1.0e12, float(index)],
                    "approach_sector": sector_check,
                    "failure_reason": "pregrasp_outside_allowed_world_sector",
                }
            )
            continue
        validation = (
            dry_sequence(payload, object_name=cup_id)
            if callable(dry_sequence)
            else dry_stage(
                payload["poses"]["pregrasp"]["base"],
                stage="pregrasp",
                gripper_action=-1.0,
            )
        )
        score = _plan_score(validation, index=index)
        record = {
            "index": int(index),
            "name": str(candidate.get("template_name", index)),
            "pass": bool(validation.get("pass", False)),
            "score": list(score),
            "approach_sector": sector_check,
        }
        records.append(record)
        if record["pass"]:
            reachable.append((score, payload, record))
    if not reachable:
        return {
            "payload": None,
            "selection": {
                "pass": False,
                "mode": "task_owned_manual_cup_grasp",
                "candidate_source": None if source is None else str(source),
                "grasp_approach_sector": approach_sector,
                "candidates": records,
                "reason": "no_reachable_manual_cup_grasp_candidate",
            },
        }
    score, payload, selected = min(reachable, key=lambda item: item[0])
    return {
        "payload": payload,
        "selection": {
            "pass": True,
            "mode": "task_owned_manual_cup_grasp",
            "candidate_source": None if source is None else str(source),
            "grasp_approach_sector": approach_sector,
            "selected_name": selected["name"],
            "selected_score": list(score),
            "candidates": records,
        },
    }


def _load_manual_candidates(
    path: str | Path | None,
) -> tuple[Path | None, tuple[Mapping[str, Any], ...]]:
    if path is None:
        return None, ()
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError(f"WA-02 cup grasp resource has no candidates: {source}")
    pregrasp = payload.get("pregrasp_offset_tcp", [0.0, 0.0, -0.075])
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping) or not isinstance(
            raw.get("eef_site_in_object"), Mapping
        ):
            raise ValueError(f"invalid WA-02 cup grasp candidate {index} in {source}")
        result.append(
            {
                "schema_version": 2,
                "asset": payload.get("asset", "robocasa cup_2"),
                "object_name": "water_cup",
                "source_file": str(source),
                "pregrasp_offset_tcp": pregrasp,
                "template_name": str(raw.get("name", index)),
                "selected": {"eef_site_in_object": dict(raw["eef_site_in_object"])},
            }
        )
    return source, tuple(result)


def _parse_grasp_approach_sector(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("WA-02 grasp_approach_sector must be a mapping")
    center = float(value.get("center_deg", 180.0))
    half_width = float(value.get("half_width_deg", 60.0))
    if not np.isfinite(center) or not np.isfinite(half_width):
        raise ValueError("WA-02 grasp approach sector values must be finite")
    if not 0.0 < half_width <= 180.0:
        raise ValueError("WA-02 grasp approach half-width must be in (0, 180]")
    return {"center_deg": center, "half_width_deg": half_width}


def _evaluate_grasp_approach_sector(
    payload: Mapping[str, Any],
    sector: Mapping[str, float] | None,
) -> dict[str, Any] | None:
    """Check the world-frame cup-center-to-pregrasp azimuth."""

    if sector is None:
        return None
    poses = payload.get("poses")
    if not isinstance(poses, Mapping):
        raise ValueError("WA-02 pick payload is missing poses")
    pregrasp = matrix_from_json_pose(dict(_mapping(poses, "pregrasp"))["world"])
    grasp = matrix_from_json_pose(dict(_mapping(poses, "grasp"))["world"])
    radial = np.asarray(pregrasp[:2, 3] - grasp[:2, 3], dtype=float)
    standoff = float(np.linalg.norm(radial))
    if standoff <= 1.0e-8:
        raise ValueError("WA-02 cup pregrasp has no horizontal standoff")
    side_angle = float(np.degrees(np.arctan2(radial[1], radial[0])))
    center = float(sector["center_deg"])
    half_width = float(sector["half_width_deg"])
    delta = abs(float((side_angle - center + 180.0) % 360.0 - 180.0))
    motion_angle = float((side_angle + 360.0) % 360.0 - 180.0)
    return {
        "pass": bool(delta <= half_width + 1.0e-9),
        "frame": "world",
        "angle_reference": "cup_center_to_pregrasp",
        "pregrasp_side_angle_world_deg": side_angle,
        "gripper_motion_angle_world_deg": motion_angle,
        "sector_center_deg": center,
        "sector_half_width_deg": half_width,
        "angular_distance_from_center_deg": delta,
        "horizontal_standoff_m": standoff,
    }


def load_anchor_pose(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a task-owned pose JSON accepting either matrix or p/q_xyzw form."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"WA-02 anchor pose must be a JSON object: {source}")
    raw_pose = _mapping(_mapping(payload, "pose"), "world")
    pose = matrix_from_json_pose(dict(raw_pose))
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"WA-02 anchor pose is not a finite 4x4 transform: {source}")
    return np.asarray(pose, dtype=float), dict(payload)


def pose_world_to_base(obs: Mapping[str, Any], pose_world: np.ndarray) -> dict[str, Any]:
    """Represent a world-frame EEF target in the controller's current base frame."""

    base_world = matrix_from_base_obs(dict(obs))
    return pose_from_matrix(np.linalg.inv(base_world) @ np.asarray(pose_world, dtype=float))


def topdown_eef_rotation_world(obs: Mapping[str, Any]) -> np.ndarray:
    """Return a downward-facing EEF orientation aligned to the current base."""

    return matrix_from_base_obs(dict(obs))[:3, :3] @ np.diag([1.0, -1.0, -1.0])


def button_press_pose_world(
    obs: Mapping[str, Any],
    *,
    button_position_world: Any,
    z_offset: float,
) -> np.ndarray:
    """Create a downwards press pose at one live water-button position."""

    position = _vec3(button_position_world, "button_position_world")
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = topdown_eef_rotation_world(obs)
    pose[:3, 3] = position + np.asarray([0.0, 0.0, float(z_offset)], dtype=float)
    return pose


def resolve_held_fill_pose(
    controller: Any,
    cup_id: str,
    *,
    cup_z_offset_from_water: float,
    water_site_offset_x: float = 0.0,
    water_site_offset_y: float = 0.0,
    sink_fill_pose_file: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the live under-water EEF target.

    A calibrated JSON pose supplies the complete EEF transform and is
    translated by its saved anchor offset onto the live sink water site.  The
    live held-cup transform remains available for the geometry diagnostics and
    for the calibration-free fallback.
    """

    if cup_id not in CUP_IDS:
        raise ValueError(f"invalid WA-02 cup id: {cup_id!r}")
    env = controller.env
    water_site = _vec3(env.water_site_position(), "sink water site")
    current_eef = np.asarray(
        controller.stage_executor.current_eef_site_world(), dtype=float
    ).reshape(4, 4)
    cup_position = current_held_or_object_position_world(controller, cup_id)
    local_cup_from_eef = current_eef[:3, :3].T @ (
        cup_position - current_eef[:3, 3]
    )
    target_rotation = np.asarray(current_eef[:3, :3], dtype=float)
    calibrated_target: np.ndarray | None = None
    calibration: dict[str, Any] | None = None
    if sink_fill_pose_file not in (None, "", "none"):
        calibrated, payload = load_anchor_pose(Path(str(sink_fill_pose_file)))
        target_rotation = np.asarray(calibrated[:3, :3], dtype=float)
        pose_payload = dict(payload.get("pose") or {})
        offset_value = pose_payload.get("offset_from_anchor_world")
        if offset_value is None:
            saved_anchor = dict(payload.get("anchor") or {}).get("world")
            if saved_anchor is not None:
                offset_value = (
                    np.asarray(calibrated[:3, 3], dtype=float)
                    - _vec3(saved_anchor, "saved sink anchor")
                )
        if offset_value is None:
            raise ValueError(
                "WA-02 sink fill calibration requires "
                "pose.offset_from_anchor_world or anchor.world"
            )
        anchor_offset = _vec3(offset_value, "sink fill anchor offset")
        calibrated_target = np.asarray(calibrated, dtype=float).copy()
        calibrated_target[:3, 3] = water_site + anchor_offset
        calibration = {
            "file": str(sink_fill_pose_file),
            "anchor": payload.get("anchor"),
            "world_pose": pose_from_matrix(calibrated),
            "offset_from_anchor_world": anchor_offset.astype(float).tolist(),
            "position_mode": "full_eef_pose_reanchored_to_live_water_site",
        }
    desired_cup = water_site + np.asarray(
        [
            float(water_site_offset_x),
            float(water_site_offset_y),
            float(cup_z_offset_from_water),
        ],
        dtype=float,
    )
    if calibrated_target is not None:
        target = calibrated_target
    else:
        target = np.eye(4, dtype=float)
        target[:3, :3] = target_rotation
        target[:3, 3] = desired_cup - target_rotation @ local_cup_from_eef
    return {
        "target_eef_world": target,
        "water_site_world": water_site,
        "desired_cup_world": desired_cup,
        "current_cup_world": cup_position,
        "local_cup_from_eef": local_cup_from_eef,
        "calibration": calibration,
    }


def current_held_or_object_position_world(controller: Any, cup_id: str) -> np.ndarray:
    """Read the carried cup's actual/virtual world position without labels."""

    current_eef = np.asarray(
        controller.stage_executor.current_eef_site_world(), dtype=float
    ).reshape(4, 4)
    virtual_position = getattr(controller, "_virtual_object_pos_world_for_eef", None)
    if callable(virtual_position):
        try:
            value = virtual_position(cup_id, current_eef)
        except (KeyError, TypeError, ValueError):
            value = None
        if value is not None:
            return _vec3(value, f"virtual {cup_id} position")
    env = controller.env
    return _vec3(env.object_position(cup_id), f"{cup_id} position")


def service_slot_position_world(env: Any, slot: str) -> np.ndarray:
    """Read a neutral slot position from the initialized physical environment."""

    if slot not in SERVICE_SLOTS:
        raise ValueError(f"invalid WA-02 service slot: {slot!r}")
    return _vec3(env.service_slot_position(slot), f"service slot {slot}")


def _plan_score(validation: Mapping[str, Any], *, index: int) -> tuple[float, ...]:
    if not bool(validation.get("pass", False)):
        return (1.0e12, 1.0e12, float(index))
    stages = [item for item in validation.get("stages", ()) if isinstance(item, Mapping)]
    lengths = [
        float((item.get("plan") or {}).get("trajectory_len", 1.0e12))
        for item in stages
    ]
    return (max(lengths, default=1.0e12), sum(lengths), float(index))


def _float_sequence(value: Any) -> tuple[float, ...]:
    raw = value.replace(";", ",").split(",") if isinstance(value, str) else value
    result = tuple(float(item) for item in raw)
    if not result:
        raise ValueError("WA-02 grasp yaw offsets must not be empty")
    if not all(np.isfinite(item) for item in result):
        raise ValueError("WA-02 grasp yaw offsets must be finite")
    return result


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"WA-02 anchor pose is missing mapping {key!r}")
    return value


def _vec3(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        raise ValueError(f"WA-02 {label} must be a finite xyz value")
    return np.asarray(array[:3], dtype=float)


__all__ = [
    "DEFAULT_CUP_GRASP_YAW_OFFSETS_DEG",
    "build_pick_payload_selector",
    "button_press_pose_world",
    "current_held_or_object_position_world",
    "load_anchor_pose",
    "pose_world_to_base",
    "resolve_held_fill_pose",
    "service_slot_position_world",
    "topdown_eef_rotation_world",
]
