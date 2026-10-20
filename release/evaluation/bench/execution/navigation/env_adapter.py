"""Environment bridge for station navigation.

This file is the env-facing half of Phase 6. It reads RoboCasa-like objects
and fixtures, then returns plain numpy/list data consumed by the pure planning
layer. It must not import task implementations or ``src`` modules.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.navigation.station_poses import yaw_to_quat_xyzw
from bench.execution.planning.nav_algo import (
    convex_hull,
    polygon_area,
    project_point_to_segment,
    select_station_candidates,
    validate_station_candidates,
)


def safe_name(obj: Any) -> str | None:
    if obj is None:
        return None
    for attr in ("name", "naming_prefix"):
        value = getattr(obj, attr, None)
        if value is not None:
            return str(value)
    return None


_safe_name = safe_name


def _safe_array(value: Any) -> np.ndarray | None:
    try:
        arr = np.array(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _fixture_ext_polygon(fixture: Any) -> np.ndarray | None:
    try:
        p0, px, py, _ = fixture.get_ext_sites(relative=False)
        p0 = np.array(p0, dtype=float)[:2]
        px = np.array(px, dtype=float)[:2]
        py = np.array(py, dtype=float)[:2]
        pxy = px + py - p0
        points = np.array([p0, px, pxy, py], dtype=float)
        if points.shape == (4, 2) and np.all(np.isfinite(points)):
            return points
    except Exception:
        pass
    return None


def fixture_footprint_polygon(fixture: Any) -> tuple[np.ndarray | None, str]:
    """Return a 2D fixture footprint polygon in world coordinates."""
    try:
        points = np.array(fixture.get_bbox_points(), dtype=float)
        if points.ndim == 2 and points.shape[1] >= 2 and np.all(np.isfinite(points)):
            hull = convex_hull(points[:, :2])
            if len(hull) >= 3:
                return hull, "fixture.get_bbox_points"
    except Exception:
        pass

    ext = _fixture_ext_polygon(fixture)
    if ext is not None:
        return ext, "fixture.get_ext_sites"

    pos = _safe_array(getattr(fixture, "pos", None))
    size = _safe_array(getattr(fixture, "size", None))
    if pos is None or size is None or pos.size < 2 or size.size < 2:
        return None, "unavailable"
    half = 0.5 * size[:2]
    xy = pos[:2]
    return np.array(
        [
            [xy[0] - half[0], xy[1] - half[1]],
            [xy[0] + half[0], xy[1] - half[1]],
            [xy[0] + half[0], xy[1] + half[1]],
            [xy[0] - half[0], xy[1] + half[1]],
        ],
        dtype=float,
    ), "fixture.pos_size_aabb"


def object_xy(env: Any, obj_name: str) -> np.ndarray | None:
    try:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]], dtype=float)[:2]
    except Exception:
        return None


def fixture_xy(fixture: Any) -> np.ndarray | None:
    pos = _safe_array(getattr(fixture, "pos", None))
    if pos is None or pos.size < 2:
        return None
    return pos[:2]


def callable_xy(fn: Any) -> np.ndarray | None:
    if fn is None:
        return None
    try:
        value = np.array(fn(), dtype=float).reshape(-1)
    except Exception:
        return None
    if value.size < 2 or not np.all(np.isfinite(value)):
        return None
    return value[:2]


def nominal_base_xy(env: Any, ref_fixture: Any, ref_object: str | None) -> np.ndarray | None:
    """Ask RoboCasa for its nominal base placement when available."""
    try:
        from robocasa.utils import env_utils as EnvUtils

        pos, _ = EnvUtils.compute_robot_base_placement_pose(
            env,
            ref_fixture=ref_fixture,
            ref_object=ref_object,
        )
        arr = np.array(pos, dtype=float).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr)):
            return arr[:2]
    except Exception:
        return None
    return None


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(point - project_point_to_segment(point, a, b)))


def best_operation_edge(
    poly: np.ndarray,
    anchor_xy: np.ndarray,
    nominal_xy: np.ndarray | None,
) -> dict[str, Any]:
    center = np.mean(poly, axis=0)
    area = polygon_area(poly)
    best: dict[str, Any] | None = None
    for index in range(len(poly)):
        a = poly[index]
        b = poly[(index + 1) % len(poly)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-6:
            continue
        tangent = edge / length
        if area >= 0:
            normal_a = np.array([tangent[1], -tangent[0]], dtype=float)
        else:
            normal_a = np.array([-tangent[1], tangent[0]], dtype=float)
        mid = 0.5 * (a + b)
        if np.dot(normal_a, mid - center) < 0:
            normal_a = -normal_a
        projection = project_point_to_segment(anchor_xy, a, b)
        nominal_edge_dist = 0.0
        if nominal_xy is not None:
            side_score = float(np.dot(normal_a, nominal_xy - projection))
            normal = normal_a if side_score >= 0 else -normal_a
            nominal_projection = project_point_to_segment(nominal_xy, a, b)
            nominal_edge_dist = _point_segment_distance(nominal_xy, a, b)
        else:
            normal = normal_a
            nominal_projection = projection
        anchor_edge_dist = _point_segment_distance(anchor_xy, a, b)
        score = anchor_edge_dist
        if nominal_xy is not None:
            score = 4.0 * nominal_edge_dist + 0.25 * anchor_edge_dist
        candidate = {
            "edge_index": index,
            "a": a,
            "b": b,
            "projection": projection,
            "nominal_projection": nominal_projection,
            "tangent": tangent,
            "normal": normal / max(float(np.linalg.norm(normal)), 1e-9),
            "nominal_edge_dist": nominal_edge_dist,
            "anchor_edge_dist": anchor_edge_dist,
            "score": score,
        }
        if best is None or float(candidate["score"]) < float(best["score"]):
            best = candidate
    if best is None:
        raise ValueError("fixture footprint has no usable edge")
    return best


def generate_station_candidates(
    env: Any,
    station_specs: Mapping[str, Mapping[str, Any]],
    *,
    tangential_offsets: tuple[float, ...] = (-0.16, -0.08, 0.0, 0.08, 0.16),
    stand_off: float = 0.50,
) -> dict[str, list[dict[str, Any]]]:
    """Generate fixture-edge station candidates from task-provided station specs."""
    specs = station_specs
    station_candidates: dict[str, list[dict[str, Any]]] = {}
    for station_id, spec in specs.items():
        fixture = spec.get("fixture")
        anchor_xy = spec.get("anchor_xy")
        station_offsets = _station_tangential_offsets(spec, tangential_offsets)
        candidates: list[dict[str, Any]] = []
        if fixture is None or anchor_xy is None:
            station_candidates[station_id] = [
                {
                    "station_id": station_id,
                    "index": index,
                    "pass": False,
                    "reason": "missing_fixture_or_anchor",
                }
                for index in range(len(station_offsets))
            ]
            continue
        anchor_xy = np.array(anchor_xy, dtype=float).reshape(2)
        poly, poly_source = fixture_footprint_polygon(fixture)
        if poly is None:
            station_candidates[station_id] = [
                {
                    "station_id": station_id,
                    "index": index,
                    "pass": False,
                    "reason": "fixture_polygon_unavailable",
                }
                for index in range(len(station_offsets))
            ]
            continue
        source = str(spec.get("source_object_or_phase") or "")
        ref_object = source if source in getattr(env, "objects", {}) else None
        configured_nominal_xy = _safe_array(spec.get("nominal_xy"))
        nominal_xy = (
            configured_nominal_xy[:2]
            if configured_nominal_xy is not None and configured_nominal_xy.size >= 2
            else nominal_base_xy(env, fixture, ref_object)
        )
        edge = best_operation_edge(poly, anchor_xy, nominal_xy)
        edge_clearance = float(spec.get("stand_off", stand_off))
        for index, offset in enumerate(station_offsets):
            xy = (
                edge["projection"]
                + edge["tangent"] * float(offset)
                + edge["normal"] * edge_clearance
            )
            yaw = math.atan2(float(anchor_xy[1] - xy[1]), float(anchor_xy[0] - xy[0]))
            candidates.append(
                {
                    "station_id": station_id,
                    "index": int(index),
                    "xy": xy.astype(float).tolist(),
                    "yaw": float(((yaw + np.pi) % (2 * np.pi)) - np.pi),
                    "anchor_xy": anchor_xy.astype(float).tolist(),
                    "source_fixture": spec.get("source_fixture") or _safe_name(fixture),
                    "source_object_or_phase": spec.get("source_object_or_phase"),
                    "fixture_polygon_source": poly_source,
                    "operation_edge_index": int(edge["edge_index"]),
                    "operation_edge_projection": np.array(edge["projection"], dtype=float).tolist(),
                    "operation_edge_nominal_projection": np.array(
                        edge["nominal_projection"],
                        dtype=float,
                    ).tolist(),
                    "operation_edge_tangent": np.array(edge["tangent"], dtype=float).tolist(),
                    "operation_edge_normal": np.array(edge["normal"], dtype=float).tolist(),
                    "operation_edge_anchor_dist": float(edge["anchor_edge_dist"]),
                    "operation_edge_nominal_dist": float(edge["nominal_edge_dist"]),
                    "tangential_offset": float(offset),
                    "stand_off": float(edge_clearance),
                    "nominal_base_xy": (
                        None if nominal_xy is None else nominal_xy.astype(float).tolist()
                    ),
                    "generation_reason": "fixture_edge_tangent_sample",
                }
            )
        station_candidates[station_id] = candidates
    return station_candidates


def _station_tangential_offsets(
    spec: Mapping[str, Any],
    default: Sequence[float],
) -> tuple[float, ...]:
    raw = spec.get("tangential_offsets")
    if raw is None:
        return tuple(float(value) for value in default)
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return tuple(float(value) for value in default)
    return values if values else tuple(float(value) for value in default)


def fixture_include_reason(name: str, fixture: Any) -> tuple[bool, str]:
    haystack = f"{name} {fixture.__class__.__name__}".lower()
    exclude = ("floor", "wall", "stool", "accessory", "plant", "window", "light")
    if any(token in haystack for token in exclude):
        return False, "excluded_by_keyword"
    include = (
        "counter",
        "island",
        "cabinet",
        "stove",
        "stovetop",
        "fridge",
        "dishwasher",
        "microwave",
        "sink",
        "oven",
        "coffee",
        "housing",
        "hood",
    )
    if any(token in haystack for token in include):
        return True, "included_static_fixture"
    return False, "not_floor_level_static_fixture"


def build_fixture_obstacles(env: Any) -> list[dict[str, Any]]:
    obstacles: list[dict[str, Any]] = []
    for name, fixture in (getattr(env, "fixtures", {}) or {}).items():
        include, reason = fixture_include_reason(str(name), fixture)
        poly, source = fixture_footprint_polygon(fixture)
        item: dict[str, Any] = {
            "name": str(name),
            "class": fixture.__class__.__name__,
            "include": bool(include and poly is not None),
            "reason": reason if poly is not None else "polygon_unavailable",
            "polygon_source": source,
        }
        if poly is not None:
            item["polygon"] = poly.astype(float).tolist()
            item["bbox_min"] = np.min(poly, axis=0).astype(float).tolist()
            item["bbox_max"] = np.max(poly, axis=0).astype(float).tolist()
        obstacles.append(item)
    return obstacles


def current_base_xy_from_obs(obs: Mapping[str, Any]) -> np.ndarray:
    value = obs["robot0_base_pos"]
    arr = np.array(value, dtype=float).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        raise ValueError("obs['robot0_base_pos'] does not contain a finite xy position")
    return arr[:2]


def station_specs_from_candidates(
    station_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, StationSpec]:
    """Convert generated station candidates into ``StationSpec`` records."""
    specs: dict[str, StationSpec] = {}
    for station_id, candidates in station_candidates.items():
        poses: list[Pose] = []
        for candidate in candidates:
            xy_value = candidate.get("xy")
            if xy_value is None:
                continue
            xy = np.array(xy_value, dtype=float).reshape(-1)
            if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
                continue
            yaw = float(candidate.get("yaw", 0.0))
            poses.append(
                Pose(
                    position=(float(xy[0]), float(xy[1]), 0.0),
                    quaternion=yaw_to_quat_xyzw(yaw),
                )
            )
        specs[station_id] = StationSpec(
            name=station_id,
            base_candidates=poses,
            metadata={"source": "generated_station_candidates"},
        )
    for alias, station_id in (aliases or {}).items():
        if station_id in specs:
            specs[str(alias)] = specs[station_id]
    return specs


def validate_env_station_candidates(
    env: Any,
    station_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    start_xy: Sequence[float] | np.ndarray,
    resolution: float = 0.04,
    inflation: float = 0.35,
    margin: float = 1.0,
) -> dict[str, list[dict[str, Any]]]:
    from bench.execution.planning.nav_algo import build_occupancy_grid

    include_points = [np.array(start_xy, dtype=float)[:2]]
    for candidates in station_candidates.values():
        for candidate in candidates:
            if candidate.get("xy") is not None:
                include_points.append(np.array(candidate["xy"], dtype=float)[:2])
    grid = build_occupancy_grid(
        build_fixture_obstacles(env),
        include_points=include_points,
        resolution=resolution,
        inflation=inflation,
        margin=margin,
    )
    return validate_station_candidates(grid, dict(station_candidates), start_xy=start_xy)


def station_navigation_inspection(
    env: Any,
    obs: Mapping[str, Any],
    station_specs: Mapping[str, Mapping[str, Any]],
    *,
    resolution: float = 0.04,
    inflation: float = 0.35,
    stand_off: float = 0.50,
    expected_station_ids: Sequence[str] | None = None,
    expected_candidate_count: int | None = None,
) -> dict[str, Any]:
    """Generic station navigation inspection report."""
    from bench.execution.planning.nav_algo import build_occupancy_grid

    start_xy = current_base_xy_from_obs(obs)
    candidates = generate_station_candidates(env, station_specs, stand_off=float(stand_off))
    include_points = [start_xy]
    for station in candidates.values():
        for candidate in station:
            if "xy" in candidate:
                include_points.append(np.array(candidate["xy"], dtype=float))
    grid = build_occupancy_grid(
        build_fixture_obstacles(env),
        include_points=include_points,
        resolution=float(resolution),
        inflation=float(inflation),
    )
    validation = validate_station_candidates(grid, candidates, start_xy=start_xy)
    selected = select_station_candidates(validation)
    pass_counts = {
        station_id: int(sum(1 for item in results if item.get("pass", False)))
        for station_id, results in validation.items()
    }
    expected_ids = set(str(item) for item in (expected_station_ids or ()))
    station_path_checks = {
        station_id: bool(count >= 1)
        for station_id, count in pass_counts.items()
    }
    checks = {
        "all_stations_have_candidates": all(len(items) > 0 for items in candidates.values()),
        "all_stations_have_path_candidate": all(station_path_checks.values()),
        "all_expected_stations_present": (
            True if not expected_ids else expected_ids.issubset(set(candidates))
        ),
    }
    if expected_ids:
        checks["all_expected_stations_have_path_candidate"] = all(
            pass_counts.get(station_id, 0) >= 1 for station_id in expected_ids
        )
    if expected_candidate_count is not None:
        checks["all_stations_have_expected_candidate_count"] = all(
            len(items) == int(expected_candidate_count) for items in candidates.values()
        )
    return {
        "start_xy": start_xy.astype(float).tolist(),
        "parameters": {
            "resolution": float(resolution),
            "inflation": float(inflation),
            "stand_off": float(stand_off),
        },
        "station_specs": {
            station_id: {
                "station_id": station_id,
                "source_fixture": spec.get("source_fixture"),
                "source_object_or_phase": spec.get("source_object_or_phase"),
                "anchor_xy": None
                if spec.get("anchor_xy") is None
                else np.array(spec["anchor_xy"], dtype=float).tolist(),
            }
            for station_id, spec in station_specs.items()
        },
        "grid": grid.summary(),
        "candidates": candidates,
        "validation": validation,
        "selected": selected,
        "pass_counts": pass_counts,
        "station_path_checks": station_path_checks,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


@dataclass
class RoboCasaNavigationAdapter:
    """Small stateful adapter for RoboCasa-like navigation data."""

    env: Any
    obs: dict[str, Any]
    station_specs_provider: Callable[[Any], Mapping[str, Mapping[str, Any]]] | None = None

    def current_base_xy(self) -> np.ndarray:
        return current_base_xy_from_obs(self.obs)

    def build_fixture_obstacles(self) -> list[dict[str, Any]]:
        return build_fixture_obstacles(self.env)

    def station_specs(self) -> dict[str, dict[str, Any]]:
        if self.station_specs_provider is None:
            raise RuntimeError(
                "station_specs_provider is required for station candidate generation"
            )
        return dict(self.station_specs_provider(self.env))

    def station_candidates(
        self,
        *,
        tangential_offsets: tuple[float, ...] = (-0.16, -0.08, 0.0, 0.08, 0.16),
        stand_off: float = 0.50,
    ) -> dict[str, list[dict[str, Any]]]:
        return generate_station_candidates(
            self.env,
            self.station_specs(),
            tangential_offsets=tangential_offsets,
            stand_off=stand_off,
        )

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.obs = obs

    def inspection(
        self,
        *,
        resolution: float = 0.04,
        inflation: float = 0.35,
        stand_off: float = 0.50,
    ) -> dict[str, Any]:
        return station_navigation_inspection(
            self.env,
            self.obs,
            self.station_specs(),
            resolution=resolution,
            inflation=inflation,
            stand_off=stand_off,
        )


__all__ = [
    "RoboCasaNavigationAdapter",
    "best_operation_edge",
    "build_fixture_obstacles",
    "callable_xy",
    "current_base_xy_from_obs",
    "fixture_footprint_polygon",
    "fixture_include_reason",
    "fixture_xy",
    "generate_station_candidates",
    "nominal_base_xy",
    "object_xy",
    "safe_name",
    "station_navigation_inspection",
    "station_specs_from_candidates",
    "validate_env_station_candidates",
]
