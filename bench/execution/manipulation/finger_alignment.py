"""Measure opposing box finger-pad faces in the live simulator."""

from typing import Any

import numpy as np


def measure_finger_pad_center(env: Any) -> dict[str, Any]:
    gripper = env.robots[0].gripper["right"]
    model, data = env.sim.model, env.sim.data
    ids = []
    for side in ("left_fingerpad", "right_fingerpad"):
        names = gripper.important_geoms[side]
        if len(names) != 1:
            raise ValueError("finger alignment requires one box pad per finger")
        ids.append(int(model.geom_name2id(names[0])))
    centers = np.asarray([data.geom_xpos[index] for index in ids], dtype=float)
    separation = centers[1] - centers[0]
    distance = float(np.linalg.norm(separation))
    if distance < 1e-8:
        raise ValueError("finger-pad centers coincide")
    axis = separation / distance
    faces = []
    for index, center, direction in zip(ids, centers, (axis, -axis), strict=True):
        if int(model.geom_type[index]) != 6:
            raise ValueError("finger alignment requires box collision pads")
        rotation = np.asarray(data.geom_xmat[index]).reshape(3, 3)
        local_direction = rotation.T @ direction
        normal_axis = int(np.argmax(np.abs(local_direction)))
        if abs(local_direction[normal_axis]) < 0.98:
            raise ValueError("finger-pad faces are not opposing")
        normal = rotation[:, normal_axis] * np.sign(local_direction[normal_axis])
        faces.append(center + normal * model.geom_size[index, normal_axis])
    midpoint = np.mean(faces, axis=0)
    site_id = int(env.robots[0].eef_site_id["right"])
    tcp_position = np.asarray(data.site_xpos[site_id])
    tcp_rotation = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
    return {
        "pad_face_centers_world": [point.tolist() for point in faces],
        "pad_midpoint_world": midpoint.tolist(),
        "tcp_position_world": tcp_position.tolist(),
        "pad_midpoint_in_tcp": (tcp_rotation.T @ (midpoint - tcp_position)).tolist(),
        "closing_axis_world": axis.tolist(),
        "opening_m": float(np.dot(faces[1] - faces[0], axis)),
    }
