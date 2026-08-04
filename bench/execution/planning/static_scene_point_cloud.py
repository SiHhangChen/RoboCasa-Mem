"""Static obstacle point-cloud construction for MPLib planning."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.planning.model_alignment import matrix_from_base_obs


def parse_keyword_csv(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    return [str(item).strip().lower() for item in raw_items if str(item).strip()]


def parse_vector3(
    value: str | list[float] | tuple[float, ...] | np.ndarray | None,
) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "null"}:
            return None
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    if len(raw_items) != 3:
        raise ValueError(f"expected 3 comma-separated values, got {value!r}")
    arr = np.array([float(item) for item in raw_items], dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"workspace vector contains non-finite values: {value!r}")
    return arr


def _safe_array(value: Any, *, length: int | None = None) -> np.ndarray | None:
    try:
        arr = np.array(value, dtype=float)
    except Exception:
        return None
    if length is not None and arr.size != length:
        return None
    arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _fixture_bbox_points(env: Any, fixture: Any) -> tuple[np.ndarray | None, str]:
    try:
        points = np.array(fixture.get_bbox_points(), dtype=float)
        if points.shape == (8, 3) and np.all(np.isfinite(points)):
            return points, "fixture.get_bbox_points"
    except Exception:
        pass

    pos = _safe_array(getattr(fixture, "pos", None))
    size = _safe_array(getattr(fixture, "size", None))
    if pos is None or size is None or size.size < 3:
        root_body = getattr(fixture, "root_body", None) or getattr(fixture, "name", None)
        if root_body is not None:
            try:
                pos = np.array(env.sim.data.get_body_xpos(root_body), dtype=float).reshape(-1)
            except Exception:
                pos = None
    if pos is None or size is None or size.size < 3:
        return None, "unavailable"
    if pos.size == 2:
        pos = np.array([pos[0], pos[1], 0.0], dtype=float)
    half = 0.5 * size[:3]
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
    return pos[:3] + offsets * half, "fixture.pos_size_aabb"


def _axis_values(lo: float, hi: float, spacing: float) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    length = max(hi - lo, 0.0)
    count = max(2, int(np.ceil(length / spacing)) + 1)
    return np.linspace(lo, hi, count, dtype=float)


def sample_aabb_surface(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    spacing: float,
    inflate: float,
    faces: str | list[str] | tuple[str, ...] = "all",
) -> np.ndarray:
    """Sample only the six surfaces of an inflated axis-aligned box."""
    requested_faces = set(parse_keyword_csv(faces))
    if not requested_faces or "all" in requested_faces:
        requested_faces = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
    expanded_faces = set()
    for face in requested_faces:
        if face == "bottom":
            expanded_faces.add("zmin")
        elif face == "top":
            expanded_faces.add("zmax")
        elif face == "sides":
            expanded_faces.update({"xmin", "xmax", "ymin", "ymax"})
        else:
            expanded_faces.add(face)
    requested_faces = expanded_faces

    spacing = max(float(spacing), 1e-4)
    inflate = max(float(inflate), 0.0)
    lo = np.array(bbox_min, dtype=float) - inflate
    hi = np.array(bbox_max, dtype=float) + inflate
    xs = _axis_values(lo[0], hi[0], spacing)
    ys = _axis_values(lo[1], hi[1], spacing)
    zs = _axis_values(lo[2], hi[2], spacing)

    clouds = []
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    if "xmin" in requested_faces:
        clouds.append(np.column_stack([np.full(yy.size, lo[0]), yy.ravel(), zz.ravel()]))
    if "xmax" in requested_faces:
        clouds.append(np.column_stack([np.full(yy.size, hi[0]), yy.ravel(), zz.ravel()]))
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    if "ymin" in requested_faces:
        clouds.append(np.column_stack([xx.ravel(), np.full(xx.size, lo[1]), zz.ravel()]))
    if "ymax" in requested_faces:
        clouds.append(np.column_stack([xx.ravel(), np.full(xx.size, hi[1]), zz.ravel()]))
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    if "zmin" in requested_faces:
        clouds.append(np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, lo[2])]))
    if "zmax" in requested_faces:
        clouds.append(np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, hi[2])]))

    if not clouds:
        return np.zeros((0, 3), dtype=float)

    points = np.concatenate(clouds, axis=0)
    return np.unique(np.round(points, decimals=5), axis=0)


def _fixture_matches(name: str, fixture: Any, include: list[str], exclude: list[str]) -> bool:
    class_name = fixture.__class__.__name__.lower()
    haystack = f"{name.lower()} {class_name}"
    if any(keyword in haystack for keyword in exclude):
        return False
    if not include or "*" in include or "all" in include:
        return True
    return any(keyword in haystack for keyword in include)


def _world_to_base_points(points_world: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
    base_world = matrix_from_base_obs(obs)
    world_to_base = np.linalg.inv(base_world)
    points_h = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=float)],
        axis=1,
    )
    return (world_to_base @ points_h.T).T[:, :3]


def _deterministic_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    max_points = int(max_points)
    if max_points <= 0 or len(points) <= max_points:
        return points
    indexes = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indexes]


def _aabb_intersects(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    roi_min: np.ndarray | None,
    roi_max: np.ndarray | None,
    *,
    inflate: float = 0.0,
) -> bool:
    if roi_min is None or roi_max is None:
        return True
    margin = max(float(inflate), 0.0)
    return bool(np.all((bbox_max + margin) >= roi_min) and np.all((bbox_min - margin) <= roi_max))


def _mask_points_in_workspace(
    points_base: np.ndarray,
    roi_min: np.ndarray | None,
    roi_max: np.ndarray | None,
) -> np.ndarray:
    if roi_min is None or roi_max is None:
        return np.ones(len(points_base), dtype=bool)
    return np.all((points_base >= roi_min) & (points_base <= roi_max), axis=1)


def build_static_scene_point_cloud(
    env: Any,
    obs: dict[str, Any],
    *,
    enabled: bool,
    fixture_keywords: str | list[str] | tuple[str, ...] = "cabinet",
    exclude_keywords: str | list[str] | tuple[str, ...] = "floor,wall",
    spacing: float = 0.04,
    inflate: float = 0.05,
    point_radius: float = 0.01,
    faces: str | list[str] | tuple[str, ...] = "bottom",
    min_z: float | None = 1.05,
    max_points: int = 4000,
    workspace_min_base: str | list[float] | tuple[float, ...] | np.ndarray | None = "-0.2,-0.9,0.2",
    workspace_max_base: str | list[float] | tuple[float, ...] | np.ndarray | None = "1.2,0.9,2.1",
) -> dict[str, Any]:
    """Return a JSON-serializable collision scene payload for MPLib."""
    include = parse_keyword_csv(fixture_keywords)
    exclude = parse_keyword_csv(exclude_keywords)
    workspace_min = parse_vector3(workspace_min_base)
    workspace_max = parse_vector3(workspace_max_base)
    if workspace_min is not None and workspace_max is not None:
        lo = np.minimum(workspace_min, workspace_max)
        hi = np.maximum(workspace_min, workspace_max)
        workspace_min, workspace_max = lo, hi
    scene: dict[str, Any] = {
        "enabled": bool(enabled),
        "frame": "robot_base",
        "source": "robocasa_fixture_aabb_surface",
        "fixture_keywords": include,
        "exclude_keywords": exclude,
        "spacing": float(spacing),
        "inflate": float(inflate),
        "point_radius": float(point_radius),
        "faces": parse_keyword_csv(faces),
        "min_z": None if min_z is None else float(min_z),
        "min_z_frame": "world",
        "max_points": int(max_points),
        "workspace_min_base": (
            None if workspace_min is None else workspace_min.astype(float).tolist()
        ),
        "workspace_max_base": (
            None if workspace_max is None else workspace_max.astype(float).tolist()
        ),
        "obstacles": [],
        "point_count": 0,
        "points": [],
    }
    if not enabled:
        return scene

    all_points_world: list[np.ndarray] = []
    fixtures = getattr(env, "fixtures", {}) or {}
    for fixture_name, fixture in fixtures.items():
        obstacle = _fixture_point_cloud_obstacle(
            env,
            obs,
            str(fixture_name),
            fixture,
            include=include,
            exclude=exclude,
            min_z=min_z,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            spacing=float(spacing),
            inflate=float(inflate),
            faces=faces,
        )
        if obstacle is None:
            continue
        scene["obstacles"].append(obstacle["summary"])
        if len(obstacle["points_world"]) > 0:
            all_points_world.append(obstacle["points_world"])

    if not all_points_world:
        scene["enabled"] = False
        scene["reason"] = "no_matching_obstacle_points"
        return scene

    points_world = np.concatenate(all_points_world, axis=0)
    points_world = _deterministic_downsample(points_world, int(max_points))
    points_base = _world_to_base_points(points_world, obs)
    scene["point_count"] = int(len(points_base))
    scene["points"] = points_base.astype(float).tolist()
    return scene


def _fixture_point_cloud_obstacle(
    env: Any,
    obs: dict[str, Any],
    fixture_name: str,
    fixture: Any,
    *,
    include: list[str],
    exclude: list[str],
    min_z: float | None,
    workspace_min: np.ndarray | None,
    workspace_max: np.ndarray | None,
    spacing: float,
    inflate: float,
    faces: str | list[str] | tuple[str, ...],
) -> dict[str, Any] | None:
    if not _fixture_matches(fixture_name, fixture, include, exclude):
        return None
    bbox_points, bbox_source = _fixture_bbox_points(env, fixture)
    if bbox_points is None:
        return {
            "summary": {
                "name": fixture_name,
                "class": fixture.__class__.__name__,
                "bbox_source": bbox_source,
                "skipped": True,
                "reason": "bbox_unavailable",
            },
            "points_world": np.zeros((0, 3), dtype=float),
        }
    bbox_min = np.min(bbox_points, axis=0)
    bbox_max = np.max(bbox_points, axis=0)
    if min_z is not None and float(bbox_max[2]) < float(min_z):
        return None
    bbox_points_base = _world_to_base_points(bbox_points, obs)
    bbox_min_base = np.min(bbox_points_base, axis=0)
    bbox_max_base = np.max(bbox_points_base, axis=0)
    base_summary = {
        "name": fixture_name,
        "class": fixture.__class__.__name__,
        "bbox_source": bbox_source,
        "bbox_min_world": bbox_min.astype(float).tolist(),
        "bbox_max_world": bbox_max.astype(float).tolist(),
        "bbox_min_base": bbox_min_base.astype(float).tolist(),
        "bbox_max_base": bbox_max_base.astype(float).tolist(),
    }
    if not _aabb_intersects(
        bbox_min_base,
        bbox_max_base,
        workspace_min,
        workspace_max,
        inflate=float(inflate),
    ):
        return {
            "summary": {**base_summary, "skipped": True, "reason": "outside_workspace_base"},
            "points_world": np.zeros((0, 3), dtype=float),
        }
    points_world = sample_aabb_surface(
        bbox_min,
        bbox_max,
        spacing=float(spacing),
        inflate=float(inflate),
        faces=faces,
    )
    points_base = _world_to_base_points(points_world, obs)
    workspace_mask = _mask_points_in_workspace(points_base, workspace_min, workspace_max)
    points_world = points_world[workspace_mask]
    if len(points_world) == 0:
        return {
            "summary": {
                **base_summary,
                "skipped": True,
                "reason": "no_surface_points_inside_workspace_base",
            },
            "points_world": np.zeros((0, 3), dtype=float),
        }
    return {
        "summary": {
            **base_summary,
            "workspace_point_count": int(len(points_world)),
            "point_count": int(len(points_world)),
        },
        "points_world": points_world,
    }


__all__ = [
    "build_static_scene_point_cloud",
    "parse_keyword_csv",
    "parse_vector3",
    "sample_aabb_surface",
]
