"""Pose and joint-alignment helpers for arm planning.

This is the pure Phase 7 subset of the previous MPlib model-alignment helpers.
It must stay importable without RoboCasa, robosuite, or MPlib.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def canonicalize_quat_xyzw_sign(quat: np.ndarray) -> np.ndarray:
    """Choose one deterministic sign for an XYZW quaternion.

    ``q`` and ``-q`` encode the same rotation, but a policy that consumes the
    four quaternion components as ordinary state features does not know about
    that equivalence.  Using the largest-magnitude component avoids relying on
    ``w`` near 180-degree rotations, where ``w`` is close to zero and its sign
    is numerically unstable.

    This intentionally changes only the sign.  It does not renormalize the
    quaternion, so existing observation scaling is otherwise preserved.
    """
    value = np.asarray(quat).reshape(-1)
    if value.size < 4:
        raise ValueError(f"expected at least 4 quaternion values, got shape {np.shape(quat)}")
    result = np.array(value[:4], copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"quaternion contains non-finite values: {result.tolist()}")
    dominant_index = int(np.argmax(np.abs(result)))
    if result[dominant_index] < 0:
        result = -result
    return result


def align_quat_xyzw_sign(
    quat: np.ndarray,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize an XYZW quaternion and select its episode-continuous sign.

    The first value uses the deterministic dominant-component convention. Each
    later value is placed in the same quaternion hemisphere as ``reference``.
    This removes representation-only ``q`` / ``-q`` jumps while preserving the
    represented rotation.
    """
    value = np.asarray(quat, dtype=float).reshape(-1)
    if value.size != 4 or not np.all(np.isfinite(value)):
        raise ValueError(f"expected one finite XYZW quaternion, got {value.tolist()}")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("quaternion norm must be nonzero")
    result = value / norm
    if reference is None:
        return canonicalize_quat_xyzw_sign(result)

    previous = np.asarray(reference, dtype=float).reshape(-1)
    if previous.size != 4 or not np.all(np.isfinite(previous)):
        raise ValueError(
            f"expected one finite reference XYZW quaternion, got {previous.tolist()}"
        )
    previous_norm = float(np.linalg.norm(previous))
    if previous_norm < 1e-12:
        raise ValueError("reference quaternion norm must be nonzero")
    if float(np.dot(result, previous / previous_norm)) < 0.0:
        result = -result
    return result


def align_quat_xyzw_sequence(quaternions: np.ndarray) -> np.ndarray:
    """Apply :func:`align_quat_xyzw_sign` to one ordered quaternion sequence."""
    values = np.asarray(quaternions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"expected quaternion sequence shape [T, 4], got {values.shape}")
    result = np.empty_like(values, dtype=float)
    reference: np.ndarray | None = None
    for index, value in enumerate(values):
        result[index] = align_quat_xyzw_sign(value, reference)
        reference = result[index]
    return result


def quat_xyzw_to_mat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.array(quat, dtype=float).reshape(-1)[:4]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=float,
    )


def mat_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    rot = np.array(rot, dtype=float)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=float)
    if quat[3] < 0:
        quat = -quat
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def matrix_from_pos_quat_xyzw(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = quat_xyzw_to_mat(np.array(quat_xyzw, dtype=float))
    mat[:3, 3] = np.array(pos, dtype=float)
    return mat


def matrix_from_pos_quat_wxyz(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.array(quat_wxyz, dtype=float).reshape(-1)[:4]
    return matrix_from_pos_quat_xyzw(pos, quat_wxyz[[1, 2, 3, 0]])


def matrix_from_base_obs(obs: dict[str, Any]) -> np.ndarray:
    return matrix_from_pos_quat_xyzw(
        np.array(obs["robot0_base_pos"], dtype=float),
        np.array(obs["robot0_base_quat"], dtype=float),
    )


def matrix_from_site_pose(env: Any, robot: Any) -> np.ndarray:
    site_id = robot.eef_site_id["right"]
    mat = np.eye(4)
    mat[:3, 3] = np.array(env.sim.data.site_xpos[site_id], dtype=float)
    mat[:3, :3] = np.array(env.sim.data.site_xmat[site_id].reshape(3, 3), dtype=float)
    return mat


def pose_from_matrix(mat: np.ndarray) -> dict[str, Any]:
    mat = np.array(mat, dtype=float)
    return {
        "p": mat[:3, 3].astype(float).tolist(),
        "q_xyzw": mat_to_quat_xyzw(mat[:3, :3]).astype(float).tolist(),
        "matrix": mat.astype(float).tolist(),
    }


def matrix_from_json_pose(pose: dict[str, Any]) -> np.ndarray:
    if "matrix" in pose:
        return np.array(pose["matrix"], dtype=float)
    return matrix_from_pos_quat_xyzw(
        np.array(pose["p"], dtype=float),
        np.array(pose["q_xyzw"], dtype=float),
    )


def pose7_from_matrix_wxyz(mat: np.ndarray) -> list[float]:
    mat = np.array(mat, dtype=float)
    quat_xyzw = mat_to_quat_xyzw(mat[:3, :3])
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return np.concatenate([mat[:3, 3], quat_wxyz]).astype(float).tolist()


def rotation_error_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_delta = np.array(rot_a, dtype=float).T @ np.array(rot_b, dtype=float)
    cos_angle = float(np.clip((np.trace(rot_delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def pose_error(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[float, float]:
    pose_a = np.array(pose_a, dtype=float)
    pose_b = np.array(pose_b, dtype=float)
    return (
        float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3])),
        rotation_error_deg(pose_a[:3, :3], pose_b[:3, :3]),
    )


def normalize_joint_name(name: str) -> str:
    name = str(name).lower()
    name = re.sub(r"^robot\d+_", "", name)
    name = re.sub(r"^gripper\d+_", "", name)
    name = re.sub(r"^(left|right)_", "", name)
    name = re.sub(r"^panda_", "", name)
    name = name.replace("-", "_")
    return re.sub(r"[^a-z0-9]", "", name)


def build_joint_mapping(
    mplib_joint_names: list[str],
    robocasa_arm_names: list[str],
    robocasa_gripper_names: list[str],
) -> dict[str, Any]:
    robocasa_names = list(robocasa_arm_names) + list(robocasa_gripper_names)
    norm_to_robo: dict[str, list[str]] = {}
    for name in robocasa_names:
        norm_to_robo.setdefault(normalize_joint_name(name), []).append(name)

    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for mplib_name in mplib_joint_names:
        norm = normalize_joint_name(mplib_name)
        matches = norm_to_robo.get(norm, [])
        entry: dict[str, Any] = {
            "mplib_joint_name": mplib_name,
            "mplib_normalized": norm,
            "robocasa_matches": matches,
        }
        if len(matches) == 1:
            entry["robocasa_joint_name"] = matches[0]
            entry["status"] = "ok"
            if re.match(r"panda_finger_joint[12]$", mplib_name):
                entry["value_transform"] = "absolute_opening_distance"
        elif not matches:
            entry["status"] = "missing"
            missing.append(mplib_name)
        else:
            entry["status"] = "ambiguous"
            ambiguous.append(mplib_name)
        entries.append(entry)

    move_group_entries = [
        entry
        for entry in entries
        if re.match(r"panda_joint[1-7]$", str(entry["mplib_joint_name"]))
    ]
    move_group_ok = len(move_group_entries) == 7 and all(
        entry["status"] == "ok" for entry in move_group_entries
    )
    return {
        "normalization_rules": [
            "lowercase",
            "remove leading robot<digits>_",
            "remove leading gripper<digits>_",
            "remove leading left_ or right_",
            "remove leading panda_",
            "drop non-alphanumeric separators",
        ],
        "entries": entries,
        "missing": missing,
        "ambiguous": ambiguous,
        "move_group_ok": move_group_ok,
    }


def trajectory_for_robocasa(
    mplib_positions: np.ndarray,
    move_group_joint_names: list[str],
    joint_mapping: dict[str, Any],
    robocasa_arm_joint_names: list[str],
) -> list[list[float]]:
    mapping_by_robo = {
        entry.get("robocasa_joint_name"): entry
        for entry in joint_mapping["entries"]
        if entry.get("status") == "ok"
    }
    move_group_index = {name: idx for idx, name in enumerate(move_group_joint_names)}
    trajectory = []
    for point in np.array(mplib_positions, dtype=float):
        arm_qpos = []
        for robo_name in robocasa_arm_joint_names:
            entry = mapping_by_robo[robo_name]
            arm_qpos.append(float(point[move_group_index[entry["mplib_joint_name"]]]))
        trajectory.append(arm_qpos)
    return trajectory


__all__ = [
    "align_quat_xyzw_sequence",
    "align_quat_xyzw_sign",
    "build_joint_mapping",
    "canonicalize_quat_xyzw_sign",
    "mat_to_quat_xyzw",
    "matrix_from_base_obs",
    "matrix_from_json_pose",
    "matrix_from_pos_quat_wxyz",
    "matrix_from_pos_quat_xyzw",
    "matrix_from_site_pose",
    "normalize_joint_name",
    "pose7_from_matrix_wxyz",
    "pose_error",
    "pose_from_matrix",
    "quat_xyzw_to_mat",
    "rotation_error_deg",
    "trajectory_for_robocasa",
]
