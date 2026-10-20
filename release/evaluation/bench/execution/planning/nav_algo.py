"""Pure 2D navigation algorithms for station-level mobile base planning.

This module is intentionally self-contained: only numpy and the standard
library are allowed here. Environment-specific fixture extraction stays
outside this file and should pass prebuilt obstacle polygons in.
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_BASE_FOOTPRINT_HALF_SIZE = (0.35, 0.25)
DEFAULT_BASE_FOOTPRINT_CENTER_OFFSET = (-0.20, 0.0)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def convex_hull(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Compute the convex hull of 2D points."""
    pts = sorted({(float(p[0]), float(p[1])) for p in np.array(points, dtype=float)})
    if len(pts) <= 1:
        return np.array(pts, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


# Backward-compatible private name used by older diagnostics.
_convex_hull = convex_hull


def polygon_area(poly: np.ndarray | Sequence[Sequence[float]]) -> float:
    poly = np.array(poly, dtype=float)
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_polygon(
    point: np.ndarray | Sequence[float],
    poly: np.ndarray | Sequence[Sequence[float]],
) -> bool:
    poly = np.array(poly, dtype=float)
    x = float(point[0])
    y = float(point[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            denom = yj - yi
            if abs(denom) < 1e-12:
                j = i
                continue
            x_at_y = (xj - xi) * (y - yi) / denom + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def project_point_to_segment(
    point: np.ndarray | Sequence[float],
    a: np.ndarray | Sequence[float],
    b: np.ndarray | Sequence[float],
) -> np.ndarray:
    point = np.array(point, dtype=float)[:2]
    a = np.array(a, dtype=float)[:2]
    b = np.array(b, dtype=float)[:2]
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return np.array(a, dtype=float)
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return a + t * ab


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(point - project_point_to_segment(point, a, b)))


def point_polygon_distance(
    point: np.ndarray | Sequence[float],
    poly: np.ndarray | Sequence[Sequence[float]],
) -> float:
    point = np.array(point, dtype=float)[:2]
    poly = np.array(poly, dtype=float)
    if point_in_polygon(point, poly):
        return 0.0
    dists = [
        _point_segment_distance(point, poly[i], poly[(i + 1) % len(poly)])
        for i in range(len(poly))
    ]
    return float(min(dists)) if dists else float("inf")


def _cross2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, *, eps: float = 1e-9) -> bool:
    return (
        min(float(a[0]), float(b[0])) - eps
        <= float(p[0])
        <= max(float(a[0]), float(b[0])) + eps
        and min(float(a[1]), float(b[1])) - eps
        <= float(p[1])
        <= max(float(a[1]), float(b[1])) + eps
        and abs(_cross2d(a, b, p)) <= eps
    )


def segments_intersect(
    a0: np.ndarray | Sequence[float],
    a1: np.ndarray | Sequence[float],
    b0: np.ndarray | Sequence[float],
    b1: np.ndarray | Sequence[float],
) -> bool:
    a0 = np.array(a0, dtype=float)[:2]
    a1 = np.array(a1, dtype=float)[:2]
    b0 = np.array(b0, dtype=float)[:2]
    b1 = np.array(b1, dtype=float)[:2]
    c1 = _cross2d(a0, a1, b0)
    c2 = _cross2d(a0, a1, b1)
    c3 = _cross2d(b0, b1, a0)
    c4 = _cross2d(b0, b1, a1)
    eps = 1e-9
    if (c1 > eps and c2 < -eps or c1 < -eps and c2 > eps) and (
        c3 > eps and c4 < -eps or c3 < -eps and c4 > eps
    ):
        return True
    return (
        _on_segment(a0, a1, b0)
        or _on_segment(a0, a1, b1)
        or _on_segment(b0, b1, a0)
        or _on_segment(b0, b1, a1)
    )


def polygons_intersect(
    poly_a: np.ndarray | Sequence[Sequence[float]],
    poly_b: np.ndarray | Sequence[Sequence[float]],
) -> bool:
    poly_a = np.array(poly_a, dtype=float)
    poly_b = np.array(poly_b, dtype=float)
    if len(poly_a) < 3 or len(poly_b) < 3:
        return False
    for point in poly_a:
        if point_in_polygon(point, poly_b):
            return True
    for point in poly_b:
        if point_in_polygon(point, poly_a):
            return True
    for i in range(len(poly_a)):
        a0 = poly_a[i]
        a1 = poly_a[(i + 1) % len(poly_a)]
        for j in range(len(poly_b)):
            b0 = poly_b[j]
            b1 = poly_b[(j + 1) % len(poly_b)]
            if segments_intersect(a0, a1, b0, b1):
                return True
    return False


def polygon_polygon_distance(
    poly_a: np.ndarray | Sequence[Sequence[float]],
    poly_b: np.ndarray | Sequence[Sequence[float]],
) -> float:
    poly_a = np.array(poly_a, dtype=float)
    poly_b = np.array(poly_b, dtype=float)
    if len(poly_a) < 3 or len(poly_b) < 3:
        return float("inf")
    if polygons_intersect(poly_a, poly_b):
        return 0.0
    distances = [point_polygon_distance(point, poly_b) for point in poly_a]
    distances.extend(point_polygon_distance(point, poly_a) for point in poly_b)
    return float(min(distances)) if distances else float("inf")


def base_footprint_polygon(
    xy: np.ndarray | Sequence[float],
    yaw: float,
    *,
    half_size: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_HALF_SIZE,
    center_offset: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_CENTER_OFFSET,
    safety_margin: float = 0.0,
) -> np.ndarray:
    """Return the 2D Omron base rectangle in world coordinates."""
    xy = np.array(xy, dtype=float)[:2]
    half_x = float(half_size[0]) + float(safety_margin)
    half_y = float(half_size[1]) + float(safety_margin)
    offset = np.array(center_offset, dtype=float)[:2]
    local = np.array(
        [
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y],
        ],
        dtype=float,
    )
    c = float(math.cos(yaw))
    s = float(math.sin(yaw))
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return xy + (offset + local) @ rot.T


def base_footprint_collision_summary(
    xy: np.ndarray | Sequence[float],
    yaw: float,
    obstacles: list[dict[str, Any]],
    *,
    half_size: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_HALF_SIZE,
    center_offset: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_CENTER_OFFSET,
    safety_margin: float = 0.0,
) -> dict[str, Any]:
    footprint = base_footprint_polygon(
        xy,
        yaw,
        half_size=half_size,
        center_offset=center_offset,
        safety_margin=safety_margin,
    )
    nearest: dict[str, Any] | None = None
    collision = False
    for obstacle in obstacles:
        if not obstacle.get("include", False) or "polygon" not in obstacle:
            continue
        poly = np.array(obstacle["polygon"], dtype=float)
        distance = polygon_polygon_distance(footprint, poly)
        item = {
            "name": obstacle.get("name"),
            "class": obstacle.get("class"),
            "distance": float(distance),
        }
        if distance <= 1e-9:
            collision = True
            item["collision"] = True
        if nearest is None or distance < float(nearest["distance"]):
            nearest = item
    return {
        "pass": not collision,
        "collision": bool(collision),
        "xy": np.array(xy, dtype=float)[:2].astype(float).tolist(),
        "yaw": float(yaw),
        "footprint_polygon": footprint.astype(float).tolist(),
        "half_size": [float(half_size[0]), float(half_size[1])],
        "center_offset": [float(center_offset[0]), float(center_offset[1])],
        "safety_margin": float(safety_margin),
        "nearest_obstacle": nearest,
    }


def base_motion_sweep_collision_summary(
    start_xy: np.ndarray | Sequence[float],
    start_yaw: float,
    goal_xy: np.ndarray | Sequence[float],
    goal_yaw: float,
    obstacles: list[dict[str, Any]],
    *,
    half_size: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_HALF_SIZE,
    center_offset: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_CENTER_OFFSET,
    safety_margin: float = 0.0,
    translation_step: float = 0.04,
    yaw_step: float = 0.08,
) -> dict[str, Any]:
    start_xy = np.array(start_xy, dtype=float)[:2]
    goal_xy = np.array(goal_xy, dtype=float)[:2]
    dist = float(np.linalg.norm(goal_xy - start_xy))
    yaw_delta = abs(normalize_angle(float(goal_yaw) - float(start_yaw)))
    sample_count = int(
        max(
            1,
            math.ceil(dist / max(float(translation_step), 1e-6)),
            math.ceil(yaw_delta / max(float(yaw_step), 1e-6)),
        )
    )
    worst: dict[str, Any] | None = None
    first_collision: dict[str, Any] | None = None
    for index in range(sample_count + 1):
        t = float(index / max(sample_count, 1))
        xy = start_xy + (goal_xy - start_xy) * t
        yaw = normalize_angle(
            float(start_yaw) + normalize_angle(float(goal_yaw) - float(start_yaw)) * t
        )
        check = base_footprint_collision_summary(
            xy,
            yaw,
            obstacles,
            half_size=half_size,
            center_offset=center_offset,
            safety_margin=safety_margin,
        )
        check["sample_index"] = int(index)
        check["sample_t"] = t
        nearest = check.get("nearest_obstacle") or {}
        if worst is None or float(nearest.get("distance", float("inf"))) < float(
            (worst.get("nearest_obstacle") or {}).get("distance", float("inf"))
        ):
            worst = check
        if bool(check.get("collision", False)):
            first_collision = check
            break
    return {
        "pass": first_collision is None,
        "collision": first_collision is not None,
        "start_xy": start_xy.astype(float).tolist(),
        "start_yaw": float(start_yaw),
        "goal_xy": goal_xy.astype(float).tolist(),
        "goal_yaw": float(goal_yaw),
        "sample_count": int(sample_count + 1),
        "first_collision": first_collision,
        "nearest_sample": worst,
        "half_size": [float(half_size[0]), float(half_size[1])],
        "center_offset": [float(center_offset[0]), float(center_offset[1])],
        "safety_margin": float(safety_margin),
    }


def base_path_sweep_collision_summary(
    waypoints: Sequence[Sequence[float] | np.ndarray],
    obstacles: list[dict[str, Any]],
    *,
    start_yaw: float | None = None,
    terminal_yaw: float | None = None,
    heading_tolerance: float | None = None,
    half_size: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_HALF_SIZE,
    center_offset: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_CENTER_OFFSET,
    safety_margin: float = 0.0,
    translation_step: float = 0.04,
    yaw_step: float = 0.08,
) -> dict[str, Any]:
    """Check a waypoint path using the rectangular base footprint.

    A* itself searches the base center on an inflated grid. This check verifies
    the actual rectangular footprint swept along the sparse path segments.
    """
    points = [
        np.array(point, dtype=float).reshape(-1)[:2]
        for point in waypoints
    ]
    points = [point for point in points if point.size >= 2 and np.all(np.isfinite(point))]
    if len(points) < 2:
        return {
            "pass": True,
            "collision": False,
            "reason": "insufficient_waypoints",
            "waypoint_count": len(points),
            "segment_count": 0,
            "checked_segment_count": 0,
        }

    checked: list[dict[str, Any]] = []
    first_collision: dict[str, Any] | None = None
    nearest_segment: dict[str, Any] | None = None
    previous_yaw: float | None = None if start_yaw is None else normalize_angle(float(start_yaw))
    heading_tolerance_value = (
        None
        if heading_tolerance is None
        else max(0.0, float(heading_tolerance))
    )

    def append_check(kind: str, index: int, sweep: dict[str, Any]) -> None:
        nonlocal first_collision, nearest_segment
        nearest = sweep.get("nearest_sample") or {}
        nearest_obstacle = nearest.get("nearest_obstacle")
        item = {
            "kind": kind,
            "index": int(index),
            "pass": bool(sweep.get("pass", False)),
            "collision": bool(sweep.get("collision", False)),
            "start_xy": sweep.get("start_xy"),
            "goal_xy": sweep.get("goal_xy"),
            "start_yaw": sweep.get("start_yaw"),
            "goal_yaw": sweep.get("goal_yaw"),
            "sample_count": int(sweep.get("sample_count", 0)),
            "nearest_obstacle": nearest_obstacle,
        }
        checked.append(item)
        nearest_distance = float(
            (nearest_obstacle or {}).get("distance", float("inf"))
        )
        current_best_distance = float(
            ((nearest_segment or {}).get("nearest_obstacle") or {}).get(
                "distance",
                float("inf"),
            )
        )
        if nearest_segment is None or nearest_distance < current_best_distance:
            nearest_segment = item
        if first_collision is None and bool(sweep.get("collision", False)):
            first_collision = {
                **item,
                "sweep": sweep,
            }

    for index, (start_xy, goal_xy) in enumerate(zip(points, points[1:])):
        delta = goal_xy - start_xy
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            continue
        segment_yaw = float(math.atan2(float(delta[1]), float(delta[0])))
        translation_yaw = segment_yaw
        if previous_yaw is not None:
            rotation_goal_yaw = segment_yaw
            if heading_tolerance_value is not None:
                yaw_delta = normalize_angle(segment_yaw - previous_yaw)
                if abs(yaw_delta) <= heading_tolerance_value:
                    rotation_goal_yaw = previous_yaw
                else:
                    rotation_goal_yaw = normalize_angle(
                        segment_yaw - math.copysign(heading_tolerance_value, yaw_delta)
                    )
            rotation = base_motion_sweep_collision_summary(
                start_xy,
                previous_yaw,
                start_xy,
                rotation_goal_yaw,
                obstacles,
                half_size=half_size,
                center_offset=center_offset,
                safety_margin=safety_margin,
                translation_step=translation_step,
                yaw_step=yaw_step,
            )
            append_check("rotation", index, rotation)
            translation_yaw = normalize_angle(float(rotation_goal_yaw))
        translation = base_motion_sweep_collision_summary(
            start_xy,
            translation_yaw,
            goal_xy,
            translation_yaw,
            obstacles,
            half_size=half_size,
            center_offset=center_offset,
            safety_margin=safety_margin,
            translation_step=translation_step,
            yaw_step=yaw_step,
        )
        append_check("translation", index, translation)
        previous_yaw = translation_yaw

    if terminal_yaw is not None and previous_yaw is not None:
        terminal_rotation = base_motion_sweep_collision_summary(
            points[-1],
            previous_yaw,
            points[-1],
            float(terminal_yaw),
            obstacles,
            half_size=half_size,
            center_offset=center_offset,
            safety_margin=safety_margin,
            translation_step=translation_step,
            yaw_step=yaw_step,
        )
        append_check("terminal_rotation", len(points) - 1, terminal_rotation)

    return {
        "pass": first_collision is None,
        "collision": first_collision is not None,
        "reason": "collision" if first_collision is not None else "collision_free",
        "waypoint_count": len(points),
        "segment_count": max(0, len(points) - 1),
        "checked_segment_count": len(checked),
        "first_collision": first_collision,
        "nearest_segment": nearest_segment,
        "segments": checked,
        "heading_tolerance": (
            None
            if heading_tolerance_value is None
            else float(heading_tolerance_value)
        ),
        "half_size": [float(half_size[0]), float(half_size[1])],
        "center_offset": [float(center_offset[0]), float(center_offset[1])],
        "safety_margin": float(safety_margin),
    }


def _occupied_by_obstacles(
    xy: np.ndarray,
    obstacles: list[dict[str, Any]],
    inflation: float,
) -> bool:
    for obstacle in obstacles:
        if not obstacle.get("include", False):
            continue
        bbox_min = obstacle.get("bbox_min")
        bbox_max = obstacle.get("bbox_max")
        if bbox_min is not None and bbox_max is not None:
            lo = np.array(bbox_min, dtype=float)[:2] - float(inflation)
            hi = np.array(bbox_max, dtype=float)[:2] + float(inflation)
            if (
                float(xy[0]) < float(lo[0])
                or float(xy[0]) > float(hi[0])
                or float(xy[1]) < float(lo[1])
                or float(xy[1]) > float(hi[1])
            ):
                continue
        poly = np.array(obstacle["polygon"], dtype=float)
        if point_polygon_distance(xy, poly) <= inflation:
            return True
    return False


def nearest_obstacle_distance(
    xy: np.ndarray | Sequence[float],
    obstacles: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    xy = np.array(xy, dtype=float)[:2]
    for obstacle in obstacles:
        if not obstacle.get("include", False):
            continue
        poly = np.array(obstacle["polygon"], dtype=float)
        distance = point_polygon_distance(xy, poly)
        item = {
            "name": obstacle.get("name"),
            "class": obstacle.get("class"),
            "distance": float(distance),
        }
        if best is None or distance < float(best["distance"]):
            best = item
    return best


@dataclass
class OccupancyGrid:
    origin: np.ndarray
    resolution: float
    occupied: np.ndarray
    obstacles: list[dict[str, Any]]
    inflation: float

    @property
    def height(self) -> int:
        return int(self.occupied.shape[0])

    @property
    def width(self) -> int:
        return int(self.occupied.shape[1])

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def world_to_cell(self, xy: np.ndarray | Sequence[float]) -> tuple[int, int]:
        xy = np.array(xy, dtype=float)[:2]
        col = int(round((xy[0] - self.origin[0]) / self.resolution))
        row = int(round((xy[1] - self.origin[1]) / self.resolution))
        return row, col

    def cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        row, col = cell
        return np.array(
            [
                self.origin[0] + col * self.resolution,
                self.origin[1] + row * self.resolution,
            ],
            dtype=float,
        )

    def is_free_cell(self, cell: tuple[int, int]) -> bool:
        if not self.in_bounds(cell):
            return False
        return not bool(self.occupied[cell[0], cell[1]])

    def is_free_xy(self, xy: np.ndarray | Sequence[float]) -> bool:
        return self.is_free_cell(self.world_to_cell(xy))

    def summary(self) -> dict[str, Any]:
        occupied_count = int(np.count_nonzero(self.occupied))
        total = int(self.occupied.size)
        return {
            "origin": self.origin.astype(float).tolist(),
            "resolution": float(self.resolution),
            "width": self.width,
            "height": self.height,
            "bounds_min": self.origin.astype(float).tolist(),
            "bounds_max": (
                self.origin
                + np.array([self.width - 1, self.height - 1], dtype=float) * self.resolution
            ).astype(float).tolist(),
            "inflation": float(self.inflation),
            "occupied_cell_count": occupied_count,
            "free_cell_count": total - occupied_count,
            "occupied_fraction": float(occupied_count / max(total, 1)),
            "obstacles": self.obstacles,
        }


def _normalise_obstacle(obstacle: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(obstacle)
    if "polygon" in item:
        poly = np.array(item["polygon"], dtype=float)
        item["polygon"] = poly.astype(float).tolist()
        item.setdefault("bbox_min", np.min(poly, axis=0).astype(float).tolist())
        item.setdefault("bbox_max", np.max(poly, axis=0).astype(float).tolist())
    item["include"] = bool(item.get("include", "polygon" in item))
    return item


def build_occupancy_grid(
    obstacles: Sequence[Mapping[str, Any]],
    *,
    include_points: list[np.ndarray] | None = None,
    resolution: float = 0.04,
    inflation: float = 0.35,
    margin: float = 1.0,
) -> OccupancyGrid:
    """Build an occupancy grid from precomputed obstacle polygons.

    Each obstacle may include ``polygon`` and ``include`` fields. Environment
    adapters are responsible for converting fixtures into this shape.
    """
    normalised_obstacles = [_normalise_obstacle(obstacle) for obstacle in obstacles]
    bounds_points: list[np.ndarray] = []
    for obstacle in normalised_obstacles:
        if "polygon" in obstacle:
            bounds_points.append(np.array(obstacle["polygon"], dtype=float))
    if include_points:
        for point in include_points:
            arr = np.array(point, dtype=float).reshape(-1)
            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                bounds_points.append(arr[:2].reshape(1, 2))
    if not bounds_points:
        raise RuntimeError("cannot build occupancy grid without obstacle or include points")

    all_points = np.vstack(bounds_points)
    lo = np.min(all_points, axis=0) - float(margin)
    hi = np.max(all_points, axis=0) + float(margin)
    resolution = float(resolution)
    width = int(math.ceil((hi[0] - lo[0]) / resolution)) + 1
    height = int(math.ceil((hi[1] - lo[1]) / resolution)) + 1
    occupied = np.zeros((height, width), dtype=bool)
    for row in range(height):
        y = lo[1] + row * resolution
        for col in range(width):
            x = lo[0] + col * resolution
            occupied[row, col] = _occupied_by_obstacles(
                np.array([x, y], dtype=float),
                normalised_obstacles,
                float(inflation),
            )
    return OccupancyGrid(
        origin=lo.astype(float),
        resolution=resolution,
        occupied=occupied,
        obstacles=normalised_obstacles,
        inflation=float(inflation),
    )


build_occupancy_grid_from_obstacles = build_occupancy_grid


def _astar_neighbors(cell: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
    row, col = cell
    out: list[tuple[tuple[int, int], float]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            cost = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
            out.append(((row + dr, col + dc), cost))
    return out


def simplify_waypoints(
    points: list[list[float]],
    *,
    min_spacing: float = 0.20,
    angle_threshold: float = 0.15,  # ~8.6度，移除接近共线的点
) -> list[list[float]]:
    """简化路径点：移除冗余点并考虑转角。

    策略：
    1. 距离过近的点移除（min_spacing）
    2. 接近共线的点移除（angle_threshold）
    3. 始终保留起点和终点
    """
    if len(points) <= 2:
        return points

    points_arr = [np.array(p, dtype=float)[:2] for p in points]
    simplified = [points[0]]
    last_kept_idx = 0

    for i in range(1, len(points_arr) - 1):
        # 距离检查
        dist_to_last = float(np.linalg.norm(points_arr[i] - points_arr[last_kept_idx]))
        if dist_to_last < min_spacing:
            continue

        # 转角检查：如果当前点、上一个保留点、下一个点接近共线，跳过当前点
        vec1 = points_arr[i] - points_arr[last_kept_idx]
        vec2 = points_arr[i + 1] - points_arr[i]
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))

        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_angle = float(np.dot(vec1, vec2) / (norm1 * norm2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)

            # 转角太小（接近共线），跳过
            if angle < angle_threshold:
                continue

        simplified.append(points[i])
        last_kept_idx = i

    # 始终保留终点
    if simplified[-1] != points[-1]:
        simplified.append(points[-1])

    return simplified


def smooth_path(
    waypoints: list[list[float]],
    *,
    iterations: int = 3,
    weight_data: float = 0.5,
    weight_smooth: float = 0.3,
    tolerance: float = 0.001,
) -> list[list[float]]:
    """路径平滑：梯度下降法减少锯齿。

    参数：
    - iterations: 平滑迭代次数
    - weight_data: 数据项权重（靠近原始路径）
    - weight_smooth: 平滑项权重（减少曲率）
    - tolerance: 收敛阈值

    算法：每个内部点向"原始位置"和"相邻点中心"移动。
    """
    if len(waypoints) <= 2:
        return waypoints

    # 转换为numpy数组
    path = np.array([np.array(p, dtype=float)[:2] for p in waypoints], dtype=float)
    original = path.copy()
    n = len(path)

    for _ in range(iterations):
        change = 0.0
        for i in range(1, n - 1):  # 不动起点和终点
            # 数据项：拉向原始点
            data_force = weight_data * (original[i] - path[i])

            # 平滑项：拉向相邻点的中心
            smooth_force = weight_smooth * (path[i - 1] + path[i + 1] - 2.0 * path[i])

            # 更新
            delta = data_force + smooth_force
            path[i] += delta
            change += float(np.linalg.norm(delta))

        # 收敛判定
        if change < tolerance:
            break

    return [p.astype(float).tolist() for p in path]


def astar_path(
    grid: OccupancyGrid,
    start_xy: np.ndarray | Sequence[float],
    goal_xy: np.ndarray | Sequence[float],
    *,
    max_expanded: int | None = None,
    heuristic_weight: float = 1.0,  # >1 = 加速但可能次优；<1 = 更优但慢
) -> dict[str, Any]:
    start_time = time.perf_counter()
    start = grid.world_to_cell(start_xy)
    goal = grid.world_to_cell(goal_xy)
    result: dict[str, Any] = {
        "start_xy": np.array(start_xy, dtype=float)[:2].tolist(),
        "goal_xy": np.array(goal_xy, dtype=float)[:2].tolist(),
        "start_cell": [int(start[0]), int(start[1])],
        "goal_cell": [int(goal[0]), int(goal[1])],
        "pass": False,
    }
    if not grid.in_bounds(start):
        result["reason"] = "start_out_of_bounds"
        result["duration_sec"] = float(time.perf_counter() - start_time)
        return result
    if not grid.in_bounds(goal):
        result["reason"] = "goal_out_of_bounds"
        result["duration_sec"] = float(time.perf_counter() - start_time)
        return result
    if not grid.is_free_cell(start):
        result["reason"] = "start_occupied"
        result["duration_sec"] = float(time.perf_counter() - start_time)
        return result
    if not grid.is_free_cell(goal):
        result["reason"] = "goal_occupied"
        result["duration_sec"] = float(time.perf_counter() - start_time)
        return result

    def heuristic(cell: tuple[int, int]) -> float:
        """改进的启发式：考虑欧式距离 + 轻微转向惩罚。"""
        dx = cell[0] - goal[0]
        dy = cell[1] - goal[1]
        euclidean = math.hypot(dx, dy)
        # 对非轴向移动略微增加成本（鼓励大段直线）
        cross = abs(dx) * abs(dy)
        alignment_penalty = 0.001 * cross  # 很小的惩罚，不影响最优性
        return heuristic_weight * (euclidean + alignment_penalty)

    frontier: list[tuple[float, int, tuple[int, int]]] = [(heuristic(start), 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
    expanded = 0
    expansion_limit = None if max_expanded is None else max(1, int(max_expanded))
    hit_expansion_limit = False
    counter = 0  # 打破相同优先级的tie

    while frontier:
        _, _, current = heapq.heappop(frontier)

        # 延迟扩展：如果已经有更好路径到达此节点，跳过
        if cost_so_far.get(current, float('inf')) < cost_so_far.get(current, 0.0):
            continue

        expanded += 1
        if current == goal:
            break
        if expansion_limit is not None and expanded >= expansion_limit:
            hit_expansion_limit = True
            break
        for nxt, step_cost in _astar_neighbors(current):
            if not grid.is_free_cell(nxt):
                continue
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt)
                counter += 1
                heapq.heappush(frontier, (priority, counter, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        result.update(
            {
                "reason": "max_expanded" if hit_expansion_limit else "no_path",
                "expanded_cells": int(expanded),
                "max_expanded": None if expansion_limit is None else int(expansion_limit),
                "duration_sec": float(time.perf_counter() - start_time),
            }
        )
        return result

    cells = []
    current: tuple[int, int] | None = goal
    while current is not None:
        cells.append(current)
        current = came_from[current]
    cells.reverse()
    waypoints = [grid.cell_to_world(cell).astype(float).tolist() for cell in cells]

    # 路径后处理流程：简化 -> 平滑 -> 再简化
    sparse = simplify_waypoints(waypoints, min_spacing=0.15, angle_threshold=0.12)
    smoothed = smooth_path(sparse, iterations=3, weight_data=0.5, weight_smooth=0.3)
    final = simplify_waypoints(smoothed, min_spacing=0.12, angle_threshold=0.10)

    path_length = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        path_length += float(np.linalg.norm(np.array(b, dtype=float) - np.array(a, dtype=float)))

    smoothed_length = 0.0
    for a, b in zip(final, final[1:]):
        smoothed_length += float(np.linalg.norm(np.array(b, dtype=float) - np.array(a, dtype=float)))

    result.update(
        {
            "pass": True,
            "reason": "path_found",
            "expanded_cells": int(expanded),
            "raw_cell_count": len(cells),
            "raw_waypoint_count": len(waypoints),
            "sparse_waypoint_count": len(sparse),
            "smoothed_waypoint_count": len(smoothed),
            "final_waypoint_count": len(final),
            "path_length": float(path_length),
            "smoothed_path_length": float(smoothed_length),
            "waypoints": waypoints,
            "sparse_waypoints": final,  # 返回最终优化后的路径
            "duration_sec": float(time.perf_counter() - start_time),
        }
    )
    return result


def _heading_bins(count: int) -> list[float]:
    count = max(4, int(count))
    return [normalize_angle(2.0 * math.pi * index / count) for index in range(count)]


def _nearest_heading_index(yaw: float, heading_yaws: Sequence[float]) -> int:
    return int(
        min(
            range(len(heading_yaws)),
            key=lambda index: abs(normalize_angle(float(yaw) - float(heading_yaws[index]))),
        )
    )


def astar_heading_path(
    grid: OccupancyGrid,
    start_xy: np.ndarray | Sequence[float],
    start_yaw: float | None,
    goal_xy: np.ndarray | Sequence[float],
    *,
    terminal_yaw: float | None = None,
    half_size: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_HALF_SIZE,
    center_offset: tuple[float, float] = DEFAULT_BASE_FOOTPRINT_CENTER_OFFSET,
    safety_margin: float = 0.0,
    heading_count: int = 8,
    turn_cost: float = 0.20,
    max_expanded: int = 120000,
) -> dict[str, Any]:
    """A* over grid cell plus base heading with rectangular-footprint edge checks."""
    start_time = time.perf_counter()
    start = grid.world_to_cell(start_xy)
    goal = grid.world_to_cell(goal_xy)
    result: dict[str, Any] = {
        "planner": "footprint_heading_astar",
        "start_xy": np.array(start_xy, dtype=float)[:2].tolist(),
        "goal_xy": np.array(goal_xy, dtype=float)[:2].tolist(),
        "start_cell": [int(start[0]), int(start[1])],
        "goal_cell": [int(goal[0]), int(goal[1])],
        "pass": False,
    }
    if not grid.in_bounds(start):
        result["reason"] = "start_out_of_bounds"
        result["duration_sec"] = float(time.perf_counter() - start_time)
        return result
    if not grid.in_bounds(goal):
        result["reason"] = "goal_out_of_bounds"
        result["duration_sec"] = float(time.perf_counter() - start_time)
        return result
    if not grid.is_free_cell(start):
        result["reason"] = "start_occupied"
        result["duration_sec"] = float(time.perf_counter() - start_time)
        return result
    if not grid.is_free_cell(goal):
        result["reason"] = "goal_occupied"
        result["duration_sec"] = float(time.perf_counter() - start_time)
        return result

    heading_yaws = _heading_bins(int(heading_count))
    if start_yaw is None:
        start_states = [(start[0], start[1], index) for index in range(len(heading_yaws))]
    else:
        start_states = [
            (start[0], start[1], _nearest_heading_index(float(start_yaw), heading_yaws))
        ]
    terminal_heading_index = (
        None if terminal_yaw is None else _nearest_heading_index(float(terminal_yaw), heading_yaws)
    )
    move_specs: list[tuple[int, int, float, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            move_yaw = math.atan2(float(dr), float(dc))
            move_index = _nearest_heading_index(move_yaw, heading_yaws)
            move_cost = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
            move_specs.append((dr, dc, move_cost, move_index))

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    motion_cache: dict[tuple[tuple[int, int], int, int], bool] = {}
    terminal_cache: dict[int, bool] = {}

    def motion_ok(cell: tuple[int, int], heading_index: int, move_index: int) -> bool:
        key = (cell, int(heading_index), int(move_index))
        if key in motion_cache:
            return motion_cache[key]
        next_cell = (
            int(cell[0] + round(math.sin(heading_yaws[move_index]))),
            int(cell[1] + round(math.cos(heading_yaws[move_index]))),
        )
        current_xy = grid.cell_to_world(cell)
        next_xy = grid.cell_to_world(next_cell)
        current_yaw = heading_yaws[int(heading_index)]
        move_yaw = heading_yaws[int(move_index)]
        rotation = base_motion_sweep_collision_summary(
            current_xy,
            current_yaw,
            current_xy,
            move_yaw,
            grid.obstacles,
            half_size=half_size,
            center_offset=center_offset,
            safety_margin=safety_margin,
        )
        if not bool(rotation.get("pass", False)):
            motion_cache[key] = False
            return False
        translation = base_motion_sweep_collision_summary(
            current_xy,
            move_yaw,
            next_xy,
            move_yaw,
            grid.obstacles,
            half_size=half_size,
            center_offset=center_offset,
            safety_margin=safety_margin,
        )
        motion_cache[key] = bool(translation.get("pass", False))
        return motion_cache[key]

    def terminal_ok(heading_index: int) -> bool:
        if terminal_yaw is None:
            return True
        heading_index = int(heading_index)
        if heading_index in terminal_cache:
            return terminal_cache[heading_index]
        sweep = base_motion_sweep_collision_summary(
            grid.cell_to_world(goal),
            heading_yaws[heading_index],
            grid.cell_to_world(goal),
            float(terminal_yaw),
            grid.obstacles,
            half_size=half_size,
            center_offset=center_offset,
            safety_margin=safety_margin,
        )
        terminal_cache[heading_index] = bool(sweep.get("pass", False))
        return terminal_cache[heading_index]

    frontier: list[tuple[float, tuple[int, int, int]]] = []
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    cost_so_far: dict[tuple[int, int, int], float] = {}
    for state in start_states:
        came_from[state] = None
        cost_so_far[state] = 0.0
        heading_bias = 0.0
        if terminal_heading_index is not None:
            heading_bias = 0.05 * abs(
                normalize_angle(heading_yaws[state[2]] - heading_yaws[terminal_heading_index])
            )
        heapq.heappush(frontier, (heuristic(start) + heading_bias, state))

    expanded = 0
    goal_state: tuple[int, int, int] | None = None
    max_expanded = max(1, int(max_expanded))
    while frontier:
        _, current = heapq.heappop(frontier)
        expanded += 1
        current_cell = (int(current[0]), int(current[1]))
        current_heading = int(current[2])
        if current_cell == goal and terminal_ok(current_heading):
            goal_state = current
            break
        if expanded >= max_expanded:
            break
        for dr, dc, step_cost, move_index in move_specs:
            next_cell = (current_cell[0] + int(dr), current_cell[1] + int(dc))
            if not grid.is_free_cell(next_cell):
                continue
            if not motion_ok(current_cell, current_heading, int(move_index)):
                continue
            next_state = (next_cell[0], next_cell[1], int(move_index))
            turn_delta = abs(
                normalize_angle(heading_yaws[int(move_index)] - heading_yaws[current_heading])
            )
            new_cost = (
                cost_so_far[current]
                + float(step_cost)
                + float(turn_cost) * turn_delta / math.pi
            )
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + heuristic(next_cell)
                heapq.heappush(frontier, (priority, next_state))
                came_from[next_state] = current

    if goal_state is None:
        result.update(
            {
                "reason": "no_footprint_heading_path",
                "expanded_cells": int(expanded),
                "max_expanded": int(max_expanded),
                "motion_cache_size": int(len(motion_cache)),
                "terminal_cache_size": int(len(terminal_cache)),
                "duration_sec": float(time.perf_counter() - start_time),
            }
        )
        return result

    states = []
    current: tuple[int, int, int] | None = goal_state
    while current is not None:
        states.append(current)
        current = came_from[current]
    states.reverse()
    cells = [(int(state[0]), int(state[1])) for state in states]
    waypoints = [grid.cell_to_world(cell).astype(float).tolist() for cell in cells]
    headings = [float(heading_yaws[int(state[2])]) for state in states]
    path_length = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        path_length += float(np.linalg.norm(np.array(b, dtype=float) - np.array(a, dtype=float)))
    result.update(
        {
            "pass": True,
            "reason": "path_found",
            "expanded_cells": int(expanded),
            "raw_cell_count": len(cells),
            "raw_waypoint_count": len(waypoints),
            "sparse_waypoint_count": len(waypoints),
            "path_length": float(path_length),
            "waypoints": waypoints,
            "sparse_waypoints": waypoints,
            "heading_yaws": headings,
            "motion_cache_size": int(len(motion_cache)),
            "terminal_cache_size": int(len(terminal_cache)),
            "duration_sec": float(time.perf_counter() - start_time),
        }
    )
    return result


def validate_station_candidates(
    grid: OccupancyGrid,
    station_candidates: dict[str, list[dict[str, Any]]],
    *,
    start_xy: np.ndarray | Sequence[float],
) -> dict[str, list[dict[str, Any]]]:
    validation: dict[str, list[dict[str, Any]]] = {}
    for station_id, candidates in station_candidates.items():
        station_results = []
        for candidate in candidates:
            xy_value = candidate.get("xy")
            if xy_value is None:
                station_results.append(
                    {
                        "station_id": station_id,
                        "index": int(candidate.get("index", -1)),
                        "pass": False,
                        "footprint": {
                            "pass": False,
                            "reason": candidate.get("reason", "missing_xy"),
                        },
                        "path": {"pass": False, "reason": "skipped_missing_xy"},
                    }
                )
                continue
            xy = np.array(xy_value, dtype=float)
            footprint_pass = bool(grid.is_free_xy(xy))
            nearest = nearest_obstacle_distance(xy, grid.obstacles)
            path = (
                astar_path(grid, start_xy, xy)
                if footprint_pass
                else {
                    "pass": False,
                    "reason": "skipped_candidate_occupied",
                    "goal_xy": xy.astype(float).tolist(),
                }
            )
            station_results.append(
                {
                    "station_id": station_id,
                    "index": int(candidate.get("index", -1)),
                    "xy": xy.astype(float).tolist(),
                    "yaw": candidate.get("yaw"),
                    "footprint": {
                        "pass": footprint_pass,
                        "reason": "free" if footprint_pass else "occupied_or_out_of_bounds",
                        "nearest_obstacle": nearest,
                        "inflation": float(grid.inflation),
                    },
                    "path": path,
                    "pass": bool(footprint_pass and path.get("pass", False)),
                }
            )
        validation[station_id] = station_results
    return validation


def select_station_candidates(
    validation: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any] | None]:
    selected: dict[str, dict[str, Any] | None] = {}
    for station_id, results in validation.items():
        passing = [item for item in results if item.get("pass", False)]
        if not passing:
            selected[station_id] = None
            continue
        passing.sort(
            key=lambda item: (
                float(item.get("path", {}).get("path_length", float("inf"))),
                abs(float(item.get("yaw") or 0.0)),
                int(item.get("index", 10**6)),
            )
        )
        best = passing[0]
        path = best.get("path", {})
        selected[station_id] = {
            "station_id": station_id,
            "index": int(best.get("index", -1)),
            "xy": best.get("xy"),
            "yaw": best.get("yaw"),
            "path_length": path.get("path_length"),
            "sparse_waypoints": path.get("sparse_waypoints"),
            "sparse_waypoint_count": path.get("sparse_waypoint_count"),
            "selection_reason": "min_path_length_then_yaw_then_index",
        }
    return selected


__all__ = [
    "DEFAULT_BASE_FOOTPRINT_CENTER_OFFSET",
    "DEFAULT_BASE_FOOTPRINT_HALF_SIZE",
    "OccupancyGrid",
    "astar_heading_path",
    "astar_path",
    "base_footprint_collision_summary",
    "base_footprint_polygon",
    "base_motion_sweep_collision_summary",
    "base_path_sweep_collision_summary",
    "build_occupancy_grid",
    "build_occupancy_grid_from_obstacles",
    "convex_hull",
    "jsonable",
    "nearest_obstacle_distance",
    "normalize_angle",
    "point_in_polygon",
    "point_polygon_distance",
    "polygon_area",
    "polygon_polygon_distance",
    "polygons_intersect",
    "project_point_to_segment",
    "segments_intersect",
    "select_station_candidates",
    "simplify_waypoints",
    "smooth_path",
    "validate_station_candidates",
]
