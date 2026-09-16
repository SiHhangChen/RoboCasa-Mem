"""Runtime geometry for contact-driven TS04 drawer motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DrawerMotionFrame:
    open_direction_world: np.ndarray
    side_direction_world: np.ndarray
    hook_rotation_world: np.ndarray


def resolve_drawer_motion_frame(
    *,
    fixture_yaw: float,
    current_eef_rotation_world: np.ndarray,
    hook_orientation_blend: float,
) -> DrawerMotionFrame:
    """Resolve opening direction and a reachable side-hook orientation."""

    rotation = rotation_z(float(fixture_yaw))
    open_direction = rotation @ np.asarray([0.0, -1.0, 0.0], dtype=float)
    open_direction /= max(float(np.linalg.norm(open_direction)), 1e-12)
    side_direction = rotation @ np.asarray([1.0, 0.0, 0.0], dtype=float)
    ideal = _side_pull_rotation(-open_direction)
    current = np.asarray(current_eef_rotation_world, dtype=float).reshape(3, 3)
    hook = _blend_rotation(
        current,
        ideal,
        float(np.clip(hook_orientation_blend, 0.0, 1.0)),
    )
    return DrawerMotionFrame(open_direction, side_direction, hook)


def drawer_pull_target_positions(
    handle_world: np.ndarray,
    frame: DrawerMotionFrame,
    *,
    prehook_distance: float,
    hook_offset: float,
    pull_distance: float,
    handle_side_offset: float = 0.0,
    handle_vertical_offset: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return the calibrated prehook, hook, and pull target positions."""

    handle = np.asarray(handle_world, dtype=float).reshape(3)
    handle = (
        handle
        + frame.side_direction_world * float(handle_side_offset)
        + np.asarray([0.0, 0.0, float(handle_vertical_offset)], dtype=float)
    )
    hook = handle + frame.open_direction_world * float(hook_offset)
    return {
        "prehook": handle + frame.open_direction_world * float(prehook_distance),
        "hook": hook,
        "pull": hook + frame.open_direction_world * float(pull_distance),
    }


def vary_drawer_grasp_targets(
    targets: dict[str, np.ndarray],
    frame: DrawerMotionFrame,
    *,
    handle_side_offset: float = 0.0,
    handle_vertical_offset: float = 0.0,
    prehook_rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, np.ndarray]:
    """Shift the whole approach along the handle; rotate only the prehook."""

    angles = np.asarray(prehook_rotation_offset_deg, dtype=float)
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ValueError("TS04 prehook rotation offset must contain three finite angles")
    offsets = np.asarray([handle_side_offset, handle_vertical_offset], dtype=float)
    if not np.all(np.isfinite(offsets)):
        raise ValueError("TS04 handle offsets must be finite")
    translation = (
        frame.side_direction_world * float(handle_side_offset)
        + np.asarray([0.0, 0.0, float(handle_vertical_offset)], dtype=float)
    )
    result = {name: value.copy() for name, value in targets.items()}
    for target in result.values():
        target[:3, 3] += translation
    rotation_offset = np.eye(3)
    for axis, angle in zip(np.eye(3), np.radians(angles)):
        rotation_offset = rotation_offset @ _rodrigues(axis, float(angle))
    result["prehook"][:3, :3] = targets["prehook"][:3, :3] @ rotation_offset
    return result


def pose_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    value[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return value


def rotation_z(theta: float) -> np.ndarray:
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _side_pull_rotation(approach_direction_world: np.ndarray) -> np.ndarray:
    z_axis = np.asarray(approach_direction_world, dtype=float).copy()
    z_axis[2] = 0.0
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    x_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _blend_rotation(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    relative = np.asarray(start, dtype=float).T @ np.asarray(end, dtype=float)
    angle = float(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    axis = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or angle < 1e-12:
        return np.asarray(start, dtype=float).copy()
    axis /= norm
    return np.asarray(start, dtype=float) @ _rodrigues(axis, angle * fraction)


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    one_minus = 1.0 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=float,
    )


__all__ = [
    "DrawerMotionFrame",
    "drawer_pull_target_positions",
    "pose_matrix",
    "resolve_drawer_motion_frame",
    "rotation_z",
    "vary_drawer_grasp_targets",
]
