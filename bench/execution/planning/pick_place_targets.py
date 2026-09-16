"""Target-pose construction for pick/place manipulation primitives.

This module is env-facing but src-independent: it can read a RoboCasa-like env,
yet it must not import task implementations. The returned payloads are plain dict/list
data that downstream MPLib planning can consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.planning.model_alignment import (
    mat_to_quat_xyzw,
    matrix_from_base_obs,
    matrix_from_json_pose,
    matrix_from_site_pose,
    matrix_from_pos_quat_wxyz,
    pose_from_matrix,
    quat_xyzw_to_mat,
)


def matrix_with_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = np.array(rot, dtype=float)
    mat[:3, 3] = np.array(pos, dtype=float)
    return mat


def rot_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


GREEN_CLOSING_TO_PANDA_TCP_ROT = rot_z(-np.pi / 2.0)
GREEN_CLOSING_TO_PANDA_TCP_ROT_4 = np.eye(4)
GREEN_CLOSING_TO_PANDA_TCP_ROT_4[:3, :3] = GREEN_CLOSING_TO_PANDA_TCP_ROT


def green_closing_rot_to_panda_tcp(rot: np.ndarray) -> np.ndarray:
    return np.array(rot, dtype=float) @ GREEN_CLOSING_TO_PANDA_TCP_ROT


def green_closing_pose_to_panda_tcp(pose: np.ndarray) -> np.ndarray:
    return np.array(pose, dtype=float) @ GREEN_CLOSING_TO_PANDA_TCP_ROT_4


def object_pos(env: Any, obj_name: str) -> np.ndarray:
    return np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]], dtype=float)


def object_pose_matrix(env: Any, obj_name: str) -> np.ndarray:
    body_id = env.obj_body_id[obj_name]
    mat = np.eye(4)
    mat[:3, 3] = np.array(env.sim.data.body_xpos[body_id], dtype=float)
    mat[:3, :3] = np.array(env.sim.data.body_xmat[body_id].reshape(3, 3), dtype=float)
    return mat


def object_pose_matrix_with_override(
    env: Any,
    obj_name: str,
    *,
    object_pos_world: np.ndarray | None = None,
    object_pose_world: np.ndarray | None = None,
) -> np.ndarray:
    if object_pose_world is not None:
        return np.array(object_pose_world, dtype=float)
    pose = object_pose_matrix(env, obj_name)
    if object_pos_world is not None:
        pose[:3, 3] = np.array(object_pos_world, dtype=float)
    return pose


def normalize_placement_reference(reference: str | None) -> str:
    value = str(reference or "object_origin").strip().lower().replace("-", "_")
    aliases = {
        "": "object_origin",
        "origin": "object_origin",
        "object": "object_origin",
        "object_origin": "object_origin",
        "object_body": "object_origin",
        "body": "object_origin",
        "body_origin": "object_origin",
        "object_pos": "object_origin",
        "bbox_center": "bbox_center",
        "center": "bbox_center",
        "object_center": "bbox_center",
        "bbox_top_center": "bbox_top_center",
        "top_center": "bbox_top_center",
        "bbox_bottom_center": "bbox_bottom_center",
        "bottom_center": "bbox_bottom_center",
        "cup_bottom_center": "bbox_bottom_center",
        "base_center": "bbox_bottom_center",
    }
    if value not in aliases:
        raise ValueError(f"unsupported placement_reference {reference!r}")
    return aliases[value]


def object_reference_pos(
    env: Any,
    obj_name: str,
    *,
    placement_reference: str | None = None,
    object_pos_world: np.ndarray | None = None,
    object_pose_world: np.ndarray | None = None,
) -> np.ndarray:
    """Return the world position of a named reference point on an object."""
    reference = normalize_placement_reference(placement_reference)
    pose = object_pose_matrix_with_override(
        env,
        obj_name,
        object_pos_world=object_pos_world,
        object_pose_world=object_pose_world,
    )
    if reference == "object_origin":
        return pose[:3, 3].astype(float)

    obj = env.objects[obj_name]
    bbox = object_bbox(
        env,
        obj_name,
        obj,
        pose[:3, 3],
        mat_to_quat_xyzw(pose[:3, :3]),
    )
    return np.array(bbox[reference], dtype=float)


def eef_pose_for_object_reference_target(
    *,
    current_eef_world: np.ndarray,
    current_reference_pos_world: np.ndarray,
    target_reference_pos_world: np.ndarray,
    target_eef_rot_world: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Place an EEF pose so a held object's reference point reaches a target."""
    current_eef = np.array(current_eef_world, dtype=float)
    current_rot = current_eef[:3, :3]
    target_rot = (
        current_rot
        if target_eef_rot_world is None
        else np.array(target_eef_rot_world, dtype=float)
    )
    ref_from_eef_local = current_rot.T @ (
        np.array(current_reference_pos_world, dtype=float).reshape(3) - current_eef[:3, 3]
    )
    target_ref = np.array(target_reference_pos_world, dtype=float).reshape(3)
    target_eef_pos = target_ref - target_rot @ ref_from_eef_local
    return matrix_with_pose(target_eef_pos, target_rot), ref_from_eef_local


def upright_object_rotation(current_object_rot_world: np.ndarray) -> np.ndarray:
    """Make an object's local +z world-upright while preserving a stable yaw."""
    current_rot = np.array(current_object_rot_world, dtype=float).reshape(3, 3)
    world_z = np.array([0.0, 0.0, 1.0], dtype=float)
    projected_x = np.array([current_rot[0, 0], current_rot[1, 0], 0.0], dtype=float)
    if float(np.linalg.norm(projected_x)) < 1e-8:
        projected_y = np.array(
            [current_rot[0, 1], current_rot[1, 1], 0.0],
            dtype=float,
        )
        projected_y /= max(float(np.linalg.norm(projected_y)), 1e-12)
        projected_x = np.cross(projected_y, world_z)
    projected_x /= max(float(np.linalg.norm(projected_x)), 1e-12)
    projected_y = np.cross(world_z, projected_x)
    return np.column_stack((projected_x, projected_y, world_z))


def parse_placement_yaw_offsets_deg(
    *,
    placement_yaw_offset_deg: Any | None = None,
    placement_yaw_offsets_deg: Any | None = None,
) -> list[float] | None:
    raw = placement_yaw_offsets_deg
    if raw is None:
        raw = placement_yaw_offset_deg
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [
            float(item.strip())
            for item in raw.replace(";", ",").split(",")
            if item.strip()
        ]
    elif isinstance(raw, (list, tuple, np.ndarray)):
        values = [float(item) for item in raw]
    else:
        values = [float(raw)]
    finite_values = [value for value in values if np.isfinite(value)]
    return finite_values or None


def parse_placement_approach_angles_deg(value: Any | None = None) -> list[float] | None:
    return parse_placement_yaw_offsets_deg(placement_yaw_offsets_deg=value)


def rotate_world_z(vec: np.ndarray, yaw: float) -> np.ndarray:
    return rot_z(yaw) @ np.array(vec, dtype=float)


def topdown_rotation_from_gripper_y(gripper_y_world: np.ndarray) -> np.ndarray:
    y_axis = np.array([gripper_y_world[0], gripper_y_world[1], 0.0], dtype=float)
    norm = float(np.linalg.norm(y_axis))
    if norm < 1e-8:
        raise ValueError("cannot build top-down grasp rotation from near-vertical y axis")
    y_axis /= norm
    z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def side_rotation_from_approach(approach_world: np.ndarray) -> np.ndarray:
    z_axis = np.array([approach_world[0], approach_world[1], 0.0], dtype=float)
    norm = float(np.linalg.norm(z_axis))
    if norm < 1e-8:
        raise ValueError("cannot build side grasp rotation from near-zero approach vector")
    z_axis /= norm
    x_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def pose_pair_from_world(base_world: np.ndarray, pose_world: np.ndarray) -> dict[str, Any]:
    return {
        "world": pose_from_matrix(pose_world),
        "base": pose_from_matrix(np.linalg.inv(base_world) @ pose_world),
    }


def _manual_eef_in_object(payload: dict[str, Any]) -> np.ndarray:
    selected = payload.get("selected") or {}
    if int(payload.get("schema_version") or 0) >= 2 and isinstance(selected, dict):
        candidate = selected.get("eef_site_in_object")
        if isinstance(candidate, dict):
            return matrix_from_json_pose(candidate)

    tcp_pose = payload.get("tcp_pose_object")
    if int(payload.get("schema_version") or 0) >= 2 and isinstance(tcp_pose, dict):
        return matrix_from_json_pose(tcp_pose)

    candidates = [
        selected.get("eef_site_in_object") if isinstance(selected, dict) else None,
        (payload.get("actual") or {}).get("eef_site_in_object"),
        payload.get("eef_site_in_object"),
        (payload.get("manual_grasp") or {}).get("eef_site_in_object"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, dict):
            return green_closing_pose_to_panda_tcp(matrix_from_json_pose(candidate))
        arr = np.array(candidate, dtype=float)
        if arr.shape == (4, 4):
            return green_closing_pose_to_panda_tcp(arr)
    raise ValueError("manual grasp file does not contain eef_site_in_object")


def _safe_array(value: Any) -> list[float] | None:
    try:
        arr = np.array(value, dtype=float)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr.astype(float).tolist()


def _manual_pregrasp_offset_tcp(
    payload: dict[str, Any],
    pregrasp_distance: float | None,
) -> tuple[np.ndarray, str]:
    if pregrasp_distance is not None:
        return np.array([0.0, 0.0, -float(pregrasp_distance)], dtype=float), "cli_distance"

    value = payload.get("pregrasp_offset_tcp")
    if value is not None:
        offset = np.array(value, dtype=float).reshape(-1)
        if offset.size == 3 and np.all(np.isfinite(offset)):
            return offset.astype(float), "file_pregrasp_offset_tcp"

    return np.array([0.0, 0.0, -0.16], dtype=float), "fallback_default"


def object_bbox(
    env: Any,
    obj_name: str,
    obj: Any,
    pos: np.ndarray,
    quat_xyzw: np.ndarray,
) -> dict[str, Any]:
    points_source = None
    points = None
    try:
        points = np.array(obj.get_bbox_points(trans=pos, rot=quat_xyzw), dtype=float)
        points_source = "get_bbox_points"
    except Exception:
        points = None

    if points is None or points.shape != (8, 3) or not np.all(np.isfinite(points)):
        half_size = None
        size = _safe_array(getattr(obj, "size", None))
        if size is not None:
            half_size = 0.5 * np.array(size, dtype=float)
        else:
            radius = getattr(obj, "horizontal_radius", None)
            top = _safe_array(getattr(obj, "top_offset", None))
            bottom = _safe_array(getattr(obj, "bottom_offset", None))
            if radius is not None and top is not None and bottom is not None:
                height = abs(float(np.array(top)[2] - np.array(bottom)[2]))
                half_size = np.array([float(radius), float(radius), 0.5 * height], dtype=float)
        if half_size is None:
            half_size = np.array([0.04, 0.04, 0.04], dtype=float)
        offsets = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
            ],
            dtype=float,
        )
        points = pos + offsets * half_size
        points_source = "fallback_axis_aligned"

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    size_world = bbox_max - bbox_min
    return {
        "source": points_source,
        "points": points.astype(float).tolist(),
        "min": bbox_min.astype(float).tolist(),
        "max": bbox_max.astype(float).tolist(),
        "center": ((bbox_min + bbox_max) * 0.5).astype(float).tolist(),
        "bbox_center": ((bbox_min + bbox_max) * 0.5).astype(float).tolist(),
        "size": size_world.astype(float).tolist(),
        "top_center": np.array(
            [(bbox_min[0] + bbox_max[0]) * 0.5, (bbox_min[1] + bbox_max[1]) * 0.5, bbox_max[2]],
            dtype=float,
        ).tolist(),
        "bottom_center": np.array(
            [(bbox_min[0] + bbox_max[0]) * 0.5, (bbox_min[1] + bbox_max[1]) * 0.5, bbox_min[2]],
            dtype=float,
        ).tolist(),
        "bbox_top_center": np.array(
            [(bbox_min[0] + bbox_max[0]) * 0.5, (bbox_min[1] + bbox_max[1]) * 0.5, bbox_max[2]],
            dtype=float,
        ).tolist(),
        "bbox_bottom_center": np.array(
            [(bbox_min[0] + bbox_max[0]) * 0.5, (bbox_min[1] + bbox_max[1]) * 0.5, bbox_min[2]],
            dtype=float,
        ).tolist(),
    }


def grasp_rotation(
    env: Any,
    obs: dict[str, Any],
    obj_name: str,
    *,
    orientation_mode: str,
    grasp_yaw_offset_deg: float = 0.0,
) -> np.ndarray:
    current_rot = matrix_from_site_pose(env, env.robots[0])[:3, :3]
    if orientation_mode == "current":
        return current_rot

    body_id = env.obj_body_id[obj_name]
    quat_wxyz = np.array(env.sim.data.body_xquat[body_id], dtype=float)
    obj_rot = matrix_from_pos_quat_wxyz(np.zeros(3, dtype=float), quat_wxyz)[:3, :3]
    if orientation_mode in {"mug-body-side", "side-from-base"}:
        base_world = matrix_from_base_obs(obs)
        obj_xy = object_pos(env, obj_name)[:2]
        approach = obj_xy - base_world[:2, 3]
        try:
            yaw_offset = np.deg2rad(float(grasp_yaw_offset_deg))
            return side_rotation_from_approach(
                rotate_world_z(np.array([approach[0], approach[1], 0.0]), yaw_offset)
            )
        except ValueError:
            return current_rot
    if orientation_mode == "object-x-topdown":
        gripper_y = obj_rot[:, 0]
    elif orientation_mode == "object-y-topdown":
        gripper_y = obj_rot[:, 1]
    else:
        raise ValueError(f"unsupported orientation mode {orientation_mode!r}")

    try:
        yaw_offset = np.deg2rad(float(grasp_yaw_offset_deg))
        return topdown_rotation_from_gripper_y(rotate_world_z(gripper_y, yaw_offset))
    except ValueError:
        return current_rot


def make_receptacle_pick_place_target_poses(
    env: Any,
    obs: dict[str, Any],
    *,
    object_name: str,
    receptacle_name: str,
    orientation_mode: str,
    pregrasp_height: float,
    grasp_clearance: float,
    lift_height: float,
    place_height: float,
    retreat_height: float,
    release_offset_x: float = 0.0,
    release_offset_y: float = 0.0,
    retreat_x: float = -0.10,
    retreat_y: float = -0.18,
    grasp_yaw_offset_deg: float = 0.0,
) -> dict[str, Any]:
    base_world = matrix_from_base_obs(obs)
    body_id = env.obj_body_id[object_name]
    obj = env.objects[object_name]
    pos = object_pos(env, object_name)
    quat_xyzw = np.array(env.sim.data.body_xquat[body_id], dtype=float)[[1, 2, 3, 0]]
    bbox = object_bbox(env, object_name, obj, pos, quat_xyzw)
    top_center = np.array(bbox["top_center"], dtype=float)
    rot = grasp_rotation(
        env,
        obs,
        object_name,
        orientation_mode=orientation_mode,
        grasp_yaw_offset_deg=grasp_yaw_offset_deg,
    )
    receptacle_pos = object_pos(env, receptacle_name)
    release_xy = np.array([release_offset_x, release_offset_y], dtype=float)

    poses_world = {
        "pregrasp": matrix_with_pose(top_center + [0.0, 0.0, pregrasp_height], rot),
        "grasp": matrix_with_pose(top_center + [0.0, 0.0, grasp_clearance], rot),
        "lift": matrix_with_pose(top_center + [0.0, 0.0, grasp_clearance + lift_height], rot),
        "place": matrix_with_pose(
            receptacle_pos + np.array([release_xy[0], release_xy[1], place_height], dtype=float),
            rot,
        ),
        "retreat": matrix_with_pose(
            receptacle_pos + np.array([retreat_x, retreat_y, retreat_height], dtype=float),
            rot,
        ),
    }
    return {
        "object_bbox": bbox,
        "poses": {
            name: pose_pair_from_world(base_world, pose)
            for name, pose in poses_world.items()
        },
    }


def make_object_pick_target_poses(
    env: Any,
    obs: dict[str, Any],
    *,
    object_name: str,
    orientation_mode: str,
    pregrasp_height: float,
    grasp_clearance: float,
    lift_height: float,
    grasp_yaw_offset_deg: float = 0.0,
    object_pos_world: np.ndarray | None = None,
    grasp_offset_x: float = 0.0,
    grasp_offset_y: float = 0.0,
    grasp_offset_z: float = 0.0,
    # Offsets applied in the final gripper frame (after rotation).
    # X is along finger separation (local X of the grasp rot), Y perpendicular.
    # Positive X shifts the TCP so the finger-pad midpoint moves relative to object center.
    # Use small values (e.g. ±0.005 to ±0.015 m) to compensate TCP vs actual pad center.
    grasp_finger_offset_x: float = 0.0,
    grasp_finger_offset_y: float = 0.0,
) -> dict[str, Any]:
    """Build local pick-stage TCP targets from the current base pose."""
    base_world = matrix_from_base_obs(obs)
    body_id = env.obj_body_id[object_name]
    obj = env.objects[object_name]
    pos = (
        np.array(object_pos_world, dtype=float)
        if object_pos_world is not None
        else object_pos(env, object_name)
    )
    quat_xyzw = np.array(env.sim.data.body_xquat[body_id], dtype=float)[[1, 2, 3, 0]]
    bbox = object_bbox(env, object_name, obj, pos, quat_xyzw)
    top_center = np.array(bbox["top_center"], dtype=float)
    bbox_center = np.array(bbox["center"], dtype=float)
    grasp_offset = np.array(
        [float(grasp_offset_x), float(grasp_offset_y), float(grasp_offset_z)],
        dtype=float,
    )
    rot = grasp_rotation(
        env,
        obs,
        object_name,
        orientation_mode=orientation_mode,
        grasp_yaw_offset_deg=grasp_yaw_offset_deg,
    )
    if orientation_mode in {"mug-body-side", "side-from-base"}:
        rot = green_closing_rot_to_panda_tcp(rot)
        grasp_z = (
            bbox_center[2] + float(grasp_clearance)
            if orientation_mode == "side-from-base"
            else top_center[2] + float(grasp_clearance)
        )
        grasp_pos = np.array(
            [bbox_center[0], bbox_center[1], grasp_z],
            dtype=float,
        ) + grasp_offset
        approach_axis = np.array(rot[:3, 2], dtype=float)
        pregrasp_pos = grasp_pos - approach_axis * float(pregrasp_height)
        lift_pos = grasp_pos + np.array([0.0, 0.0, lift_height], dtype=float)
        poses_world = {
            "pregrasp": matrix_with_pose(pregrasp_pos, rot),
            "grasp": matrix_with_pose(grasp_pos, rot),
            "lift": matrix_with_pose(lift_pos, rot),
        }
    else:
        grasp_pos = top_center + np.array([0.0, 0.0, grasp_clearance], dtype=float) + grasp_offset
        # Apply offsets in the final gripper-local frame (after rot).
        # grasp_finger_offset_x moves along the finger-separation axis (rot[:,0]).
        # This lets us shift the TCP so the effective finger-pad midpoint aligns better
        # with the object center, instead of forcing the TCP itself to the bbox center.
        finger_off = np.array(
            [float(grasp_finger_offset_x), float(grasp_finger_offset_y), 0.0],
            dtype=float,
        )
        grasp_pos = grasp_pos + (rot @ finger_off)
        poses_world = {
            "pregrasp": matrix_with_pose(
                grasp_pos + np.array([0.0, 0.0, pregrasp_height - grasp_clearance], dtype=float),
                rot,
            ),
            "grasp": matrix_with_pose(grasp_pos, rot),
            "lift": matrix_with_pose(
                grasp_pos + np.array([0.0, 0.0, lift_height], dtype=float),
                rot,
            ),
        }
    return {
        "grasp_strategy": orientation_mode,
        "object_bbox": bbox,
        "poses": {
            name: pose_pair_from_world(base_world, pose)
            for name, pose in poses_world.items()
        },
    }


def make_manual_object_pick_target_poses(
    env: Any,
    obs: dict[str, Any],
    *,
    object_name: str,
    grasp_pose_file: str | Path | None,
    grasp_pose_payload: dict[str, Any] | None = None,
    pregrasp_distance: float | None,
    lift_height: float,
    object_pos_world: np.ndarray | None = None,
    object_pose_world: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build pick targets from a simulator-calibrated EEF pose in object frame."""
    path = None if grasp_pose_file is None else Path(grasp_pose_file)
    payload = dict(grasp_pose_payload) if grasp_pose_payload is not None else json.loads(path.read_text())
    eef_in_object = _manual_eef_in_object(payload)

    base_world = matrix_from_base_obs(obs)
    object_world = object_pose_matrix_with_override(
        env,
        object_name,
        object_pos_world=object_pos_world,
        object_pose_world=object_pose_world,
    )
    grasp_world = object_world @ eef_in_object

    pregrasp_offset_tcp, pregrasp_source = _manual_pregrasp_offset_tcp(payload, pregrasp_distance)
    pregrasp_tcp = np.eye(4)
    pregrasp_tcp[:3, 3] = pregrasp_offset_tcp
    pregrasp_world = grasp_world @ pregrasp_tcp
    lift_world = np.array(grasp_world, copy=True)
    lift_world[:3, 3] = grasp_world[:3, 3] + np.array([0.0, 0.0, float(lift_height)])

    body_id = env.obj_body_id[object_name]
    obj = env.objects[object_name]
    pos = object_world[:3, 3]
    quat_xyzw = np.array(env.sim.data.body_xquat[body_id], dtype=float)[[1, 2, 3, 0]]
    bbox = object_bbox(env, object_name, obj, pos, quat_xyzw)
    poses_world = {
        "pregrasp": pregrasp_world,
        "grasp": grasp_world,
        "lift": lift_world,
    }
    score_value = payload.get("score")
    if score_value is None:
        score_value = (payload.get("selected") or {}).get("score")
    try:
        score = None if score_value is None else float(score_value)
    except Exception:
        score = None
    return {
        "grasp_strategy": "manual-grasp-file",
        "manual_grasp_pose_file": str(path) if path is not None else payload.get("source_file"),
        "manual_grasp_calibrated_object": payload.get("object_name"),
        "manual_grasp_asset": payload.get("asset"),
        "manual_grasp_schema_version": payload.get("schema_version"),
        "manual_grasp_score": score,
        "manual_eef_site_in_object": pose_from_matrix(eef_in_object),
        "manual_pregrasp_distance": float(np.linalg.norm(pregrasp_offset_tcp)),
        "manual_pregrasp_offset_tcp": pregrasp_offset_tcp.astype(float).tolist(),
        "manual_pregrasp_source": pregrasp_source,
        "object_bbox": bbox,
        "poses": {
            name: pose_pair_from_world(base_world, pose)
            for name, pose in poses_world.items()
        },
    }


def make_receptacle_place_target_poses(
    env: Any,
    obs: dict[str, Any],
    *,
    object_name: str,
    receptacle_name: str,
    place_height: float,
    retreat_height: float,
    release_offset_x: float = 0.0,
    release_offset_y: float = 0.0,
    retreat_x: float = -0.10,
    retreat_y: float = -0.18,
    receptacle_pos_world: np.ndarray | None = None,
    current_eef_world: np.ndarray | None = None,
    post_release_clearance_distance: float = 0.0,
    post_release_clearance_z_offset: float = 0.0,
) -> dict[str, Any]:
    """Build local place-stage TCP targets from the current base pose."""
    del post_release_clearance_distance, post_release_clearance_z_offset
    base_world = matrix_from_base_obs(obs)
    current_eef = (
        np.array(current_eef_world, dtype=float)
        if current_eef_world is not None
        else matrix_from_site_pose(env, env.robots[0])
    )
    rot = current_eef[:3, :3]
    receptacle_pos = (
        np.array(receptacle_pos_world, dtype=float)
        if receptacle_pos_world is not None
        else object_pos(env, receptacle_name)
    )
    release_xy = np.array([release_offset_x, release_offset_y], dtype=float)
    target_obj_pos = receptacle_pos + np.array(
        [release_xy[0], release_xy[1], place_height],
        dtype=float,
    )
    poses_world = {
        "place": matrix_with_pose(target_obj_pos, rot),
        "retreat": matrix_with_pose(
            receptacle_pos + np.array([retreat_x, retreat_y, retreat_height], dtype=float),
            rot,
        ),
    }
    return {
        "object_name": object_name,
        "receptacle_name": receptacle_name,
        "target_object_pos_world": target_obj_pos.astype(float).tolist(),
        "poses": {
            name: pose_pair_from_world(base_world, pose)
            for name, pose in poses_world.items()
        },
    }


def make_world_place_target_poses(
    env: Any,
    obs: dict[str, Any],
    *,
    object_name: str,
    target_pos_world: np.ndarray,
    place_height: float,
    retreat_height: float,
    release_offset_x: float = 0.0,
    release_offset_y: float = 0.0,
    retreat_x: float = -0.10,
    retreat_y: float = -0.18,
    current_eef_world: np.ndarray | None = None,
    current_object_pos_world: np.ndarray | None = None,
    current_object_pose_world: np.ndarray | None = None,
    current_reference_pos_world: np.ndarray | None = None,
    placement_reference: str | None = None,
    target_eef_rot_world: np.ndarray | None = None,
    placement_object_upright: bool = False,
    placement_yaw_offset_deg: float | None = None,
    post_release_clearance_distance: float = 0.0,
    post_release_clearance_z_offset: float = 0.0,
    post_release_clearance_direction_world: Any | None = None,
) -> dict[str, Any]:
    """Build local place-stage TCP targets for a world-frame xyz anchor."""
    base_world = matrix_from_base_obs(obs)
    current_eef = (
        np.array(current_eef_world, dtype=float)
        if current_eef_world is not None
        else matrix_from_site_pose(env, env.robots[0])
    )
    reference = normalize_placement_reference(placement_reference)
    object_pose = (
        None
        if current_object_pose_world is None
        else np.array(current_object_pose_world, dtype=float)
    )
    target_rot = (
        current_eef[:3, :3]
        if target_eef_rot_world is None
        else np.array(target_eef_rot_world, dtype=float)
    )
    if bool(placement_object_upright):
        if object_pose is None:
            raise ValueError(
                "placement_object_upright requires current_object_pose_world"
            )
        eef_to_object_rot = current_eef[:3, :3].T @ object_pose[:3, :3]
        desired_object_rot = upright_object_rotation(object_pose[:3, :3])
        target_rot = desired_object_rot @ eef_to_object_rot.T
    if placement_yaw_offset_deg is not None:
        yaw_rot = rot_z(np.deg2rad(float(placement_yaw_offset_deg)))
        if object_pose is not None:
            eef_to_object_rot = current_eef[:3, :3].T @ object_pose[:3, :3]
            target_object_rot = target_rot @ eef_to_object_rot
            target_rot = target_object_rot @ yaw_rot @ eef_to_object_rot.T
        else:
            target_rot = yaw_rot @ target_rot
    current_obj_pos = object_reference_pos(
        env,
        object_name,
        placement_reference="object_origin",
        object_pos_world=current_object_pos_world,
        object_pose_world=object_pose,
    )
    current_ref_pos = (
        np.array(current_reference_pos_world, dtype=float).reshape(3)
        if current_reference_pos_world is not None
        else object_reference_pos(
            env,
            object_name,
            placement_reference=reference,
            object_pos_world=current_object_pos_world,
            object_pose_world=object_pose,
        )
    )
    obj_from_eef_world = current_obj_pos - current_eef[:3, 3]
    ref_from_eef_world = current_ref_pos - current_eef[:3, 3]
    obj_from_eef_local = current_eef[:3, :3].T @ obj_from_eef_world
    target_pos_world = np.array(target_pos_world, dtype=float)
    release_xy = np.array([release_offset_x, release_offset_y], dtype=float)
    target_ref_pos = target_pos_world + np.array(
        [release_xy[0], release_xy[1], place_height],
        dtype=float,
    )
    place_world, ref_from_eef_local = eef_pose_for_object_reference_target(
        current_eef_world=current_eef,
        current_reference_pos_world=current_ref_pos,
        target_reference_pos_world=target_ref_pos,
        target_eef_rot_world=target_rot,
    )
    place_eef_pos = place_world[:3, 3]
    target_obj_pos = place_eef_pos + target_rot @ obj_from_eef_local
    poses_world = {"place": place_world}
    clearance_distance = float(post_release_clearance_distance)
    clearance_z_offset = float(post_release_clearance_z_offset)
    clearance_direction = None
    clearance_eef_pos = None
    if clearance_distance > 1e-9 or abs(clearance_z_offset) > 1e-9:
        if post_release_clearance_direction_world is not None:
            direction = np.asarray(
                post_release_clearance_direction_world, dtype=float
            ).reshape(-1)
            if direction.size != 3 or not np.all(np.isfinite(direction)):
                raise ValueError(
                    "post_release_clearance_direction_world must be a finite xyz vector"
                )
            direction = direction.copy()
            direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                raise ValueError(
                    "post_release_clearance_direction_world must have nonzero xy"
                )
        else:
            direction = np.array(place_eef_pos - target_ref_pos, dtype=float)
            direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                direction = np.array(place_eef_pos - target_obj_pos, dtype=float)
                direction[2] = 0.0
                norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                direction = np.array(current_eef[:3, 3] - current_ref_pos, dtype=float)
                direction[2] = 0.0
                norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                direction = np.array([0.0, -1.0, 0.0], dtype=float)
                norm = 1.0
        clearance_direction = direction / norm
        clearance_eef_pos = (
            place_eef_pos
            + clearance_direction * clearance_distance
            + np.array([0.0, 0.0, clearance_z_offset], dtype=float)
        )
        poses_world["clearance"] = matrix_with_pose(clearance_eef_pos, target_rot)

    retreat_start = place_eef_pos if clearance_eef_pos is None else clearance_eef_pos
    retreat_eef_pos = retreat_start + np.array(
        [retreat_x, retreat_y, retreat_height - place_height],
        dtype=float,
    )
    poses_world["retreat"] = matrix_with_pose(retreat_eef_pos, target_rot)
    return {
        "object_name": object_name,
        "placement_reference": reference,
        "placement_object_upright": bool(placement_object_upright),
        "placement_yaw_offset_deg": (
            None if placement_yaw_offset_deg is None else float(placement_yaw_offset_deg)
        ),
        "target_pos_world": target_pos_world.astype(float).tolist(),
        "target_reference_pos_world": target_ref_pos.astype(float).tolist(),
        "target_object_pos_world": target_obj_pos.astype(float).tolist(),
        "current_object_pos_world": current_obj_pos.astype(float).tolist(),
        "current_reference_pos_world": current_ref_pos.astype(float).tolist(),
        "current_eef_pos_world": current_eef[:3, 3].astype(float).tolist(),
        "object_from_eef_world": obj_from_eef_world.astype(float).tolist(),
        "reference_from_eef_world": ref_from_eef_world.astype(float).tolist(),
        "reference_from_eef_local": ref_from_eef_local.astype(float).tolist(),
        "post_release_clearance_distance": clearance_distance,
        "post_release_clearance_z_offset": clearance_z_offset,
        "post_release_clearance_direction_world": (
            None
            if clearance_direction is None
            else clearance_direction.astype(float).tolist()
        ),
        "poses": {
            name: pose_pair_from_world(base_world, pose)
            for name, pose in poses_world.items()
        },
    }


def make_world_place_target_pose_candidates(
    env: Any,
    obs: dict[str, Any],
    *,
    object_name: str,
    target_pos_world: np.ndarray,
    place_height: float,
    retreat_height: float,
    release_offset_x: float = 0.0,
    release_offset_y: float = 0.0,
    retreat_x: float = -0.10,
    retreat_y: float = -0.18,
    current_eef_world: np.ndarray | None = None,
    current_object_pos_world: np.ndarray | None = None,
    current_object_pose_world: np.ndarray | None = None,
    current_reference_pos_world: np.ndarray | None = None,
    placement_reference: str | None = None,
    target_eef_rot_world: np.ndarray | None = None,
    placement_object_upright: bool = False,
    placement_yaw_offset_deg: Any | None = None,
    placement_yaw_offsets_deg: Any | None = None,
    placement_approach_angles_deg: Any | None = None,
    placement_entry_standoff: float | None = None,
    placement_exit_standoff: float | None = None,
    placement_entry_z_offset: float = 0.0,
    placement_exit_z_offset: float | None = None,
    placement_approach_frame_world: np.ndarray | None = None,
    post_release_clearance_distance: float = 0.0,
    post_release_clearance_z_offset: float = 0.0,
    post_release_clearance_direction_world: Any | None = None,
) -> list[dict[str, Any]]:
    yaw_offsets = parse_placement_yaw_offsets_deg(
        placement_yaw_offset_deg=placement_yaw_offset_deg,
        placement_yaw_offsets_deg=placement_yaw_offsets_deg,
    )
    if yaw_offsets is None:
        payloads = [
            make_world_place_target_poses(
                env,
                obs,
                object_name=object_name,
                target_pos_world=target_pos_world,
                place_height=place_height,
                retreat_height=retreat_height,
                release_offset_x=release_offset_x,
                release_offset_y=release_offset_y,
                retreat_x=retreat_x,
                retreat_y=retreat_y,
                current_eef_world=current_eef_world,
                current_object_pos_world=current_object_pos_world,
                current_object_pose_world=current_object_pose_world,
                current_reference_pos_world=current_reference_pos_world,
                placement_reference=placement_reference,
                target_eef_rot_world=target_eef_rot_world,
                placement_object_upright=placement_object_upright,
                post_release_clearance_distance=post_release_clearance_distance,
                post_release_clearance_z_offset=post_release_clearance_z_offset,
                post_release_clearance_direction_world=(
                    post_release_clearance_direction_world
                ),
            )
        ]
    else:
        payloads = [
            make_world_place_target_poses(
                env,
                obs,
                object_name=object_name,
                target_pos_world=target_pos_world,
                place_height=place_height,
                retreat_height=retreat_height,
                release_offset_x=release_offset_x,
                release_offset_y=release_offset_y,
                retreat_x=retreat_x,
                retreat_y=retreat_y,
                current_eef_world=current_eef_world,
                current_object_pos_world=current_object_pos_world,
                current_object_pose_world=current_object_pose_world,
                current_reference_pos_world=current_reference_pos_world,
                placement_reference=placement_reference,
                target_eef_rot_world=target_eef_rot_world,
                placement_object_upright=placement_object_upright,
                placement_yaw_offset_deg=float(yaw_offset),
                post_release_clearance_distance=post_release_clearance_distance,
                post_release_clearance_z_offset=post_release_clearance_z_offset,
                post_release_clearance_direction_world=(
                    post_release_clearance_direction_world
                ),
            )
            for yaw_offset in yaw_offsets
        ]

    return make_place_entry_exit_ring_pose_candidates(
        payloads,
        obs,
        placement_approach_angles_deg=placement_approach_angles_deg,
        placement_entry_standoff=placement_entry_standoff,
        placement_exit_standoff=placement_exit_standoff,
        placement_entry_z_offset=placement_entry_z_offset,
        placement_exit_z_offset=placement_exit_z_offset,
        placement_approach_frame_world=placement_approach_frame_world,
    )


def make_place_entry_exit_ring_pose_candidates(
    payloads: dict[str, Any] | list[dict[str, Any]],
    obs: dict[str, Any],
    *,
    placement_approach_angles_deg: Any | None,
    placement_entry_standoff: float | None = None,
    placement_exit_standoff: float | None = None,
    placement_entry_z_offset: float = 0.0,
    placement_exit_z_offset: float | None = None,
    placement_approach_frame_world: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Expand place payloads with radial entry/exit poses around their place pose.

    Angles are measured in the tabletop plane. By default 0 deg is +world x and
    90 deg is +world y; when ``placement_approach_frame_world`` is provided,
    the frame's projected +x/+y axes define 0/90 deg.
    """
    payload_list = [payloads] if isinstance(payloads, dict) else list(payloads)
    approach_angles = parse_placement_approach_angles_deg(
        placement_approach_angles_deg
    )
    if approach_angles is None:
        return payload_list

    entry_standoff = (
        0.12 if placement_entry_standoff is None else float(placement_entry_standoff)
    )
    exit_standoff = (
        entry_standoff
        if placement_exit_standoff is None
        else float(placement_exit_standoff)
    )
    entry_z_offset = float(placement_entry_z_offset)
    exit_z_offset = (
        entry_z_offset
        if placement_exit_z_offset is None
        else float(placement_exit_z_offset)
    )

    return [
        _with_entry_exit_poses(
            payload,
            obs,
            approach_angle_deg=float(approach_angle),
            entry_standoff=entry_standoff,
            exit_standoff=exit_standoff,
            entry_z_offset=entry_z_offset,
            exit_z_offset=exit_z_offset,
            approach_frame_world=placement_approach_frame_world,
        )
        for payload in payload_list
        for approach_angle in approach_angles
    ]


def _placement_approach_direction_world(
    approach_angle_deg: float,
    approach_frame_world: np.ndarray | None,
) -> np.ndarray:
    direction = np.array(
        [
            np.cos(np.deg2rad(float(approach_angle_deg))),
            np.sin(np.deg2rad(float(approach_angle_deg))),
            0.0,
        ],
        dtype=float,
    )
    if approach_frame_world is not None:
        frame = np.array(approach_frame_world, dtype=float)
        if frame.shape == (4, 4):
            frame_rot = frame[:3, :3]
        elif frame.shape == (3, 3):
            frame_rot = frame
        else:
            raise ValueError(
                "placement_approach_frame_world must be a 3x3 rotation or 4x4 pose"
            )
        direction = frame_rot @ direction

    direction = np.array([direction[0], direction[1], 0.0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        raise ValueError("placement approach direction is near zero in the tabletop plane")
    return direction / norm


def _with_entry_exit_poses(
    payload: dict[str, Any],
    obs: dict[str, Any],
    *,
    approach_angle_deg: float,
    entry_standoff: float,
    exit_standoff: float | None,
    entry_z_offset: float,
    exit_z_offset: float | None,
    approach_frame_world: np.ndarray | None = None,
) -> dict[str, Any]:
    base_world = matrix_from_base_obs(obs)
    place_world = matrix_from_json_pose(payload["poses"]["place"]["world"])
    direction = _placement_approach_direction_world(
        approach_angle_deg,
        approach_frame_world,
    )
    exit_distance = entry_standoff if exit_standoff is None else float(exit_standoff)
    exit_z = entry_z_offset if exit_z_offset is None else float(exit_z_offset)
    entry_pos = (
        place_world[:3, 3]
        + direction * float(entry_standoff)
        + np.array([0.0, 0.0, float(entry_z_offset)], dtype=float)
    )
    exit_pos = (
        place_world[:3, 3]
        + direction * exit_distance
        + np.array([0.0, 0.0, exit_z], dtype=float)
    )
    entry_world = matrix_with_pose(entry_pos, place_world[:3, :3])
    retreat_world = matrix_with_pose(exit_pos, place_world[:3, :3])
    poses = dict(payload["poses"])
    poses["entry"] = pose_pair_from_world(base_world, entry_world)
    poses["retreat"] = pose_pair_from_world(base_world, retreat_world)
    return {
        **{key: value for key, value in payload.items() if key != "poses"},
        "placement_approach_angle_deg": float(approach_angle_deg),
        "placement_approach_direction_world": direction.astype(float).tolist(),
        "placement_approach_frame": (
            "world" if approach_frame_world is None else "custom"
        ),
        "placement_entry_standoff": float(entry_standoff),
        "placement_exit_standoff": float(exit_distance),
        "placement_entry_z_offset": float(entry_z_offset),
        "placement_exit_z_offset": float(exit_z),
        "poses": poses,
    }


__all__ = [
    "green_closing_pose_to_panda_tcp",
    "green_closing_rot_to_panda_tcp",
    "eef_pose_for_object_reference_target",
    "grasp_rotation",
    "make_manual_object_pick_target_poses",
    "make_object_pick_target_poses",
    "make_place_entry_exit_ring_pose_candidates",
    "make_receptacle_pick_place_target_poses",
    "make_receptacle_place_target_poses",
    "make_world_place_target_pose_candidates",
    "make_world_place_target_poses",
    "matrix_with_pose",
    "normalize_placement_reference",
    "object_bbox",
    "object_pos",
    "object_pose_matrix",
    "object_pose_matrix_with_override",
    "object_reference_pos",
    "parse_placement_approach_angles_deg",
    "parse_placement_yaw_offsets_deg",
    "pose_pair_from_world",
    "rot_z",
    "rotate_world_z",
    "side_rotation_from_approach",
    "topdown_rotation_from_gripper_y",
    "upright_object_rotation",
]
