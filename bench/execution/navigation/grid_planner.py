"""Station-level path planning for mobile-base navigation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import sys
import time
from typing import Any

import numpy as np

from bench.execution.planning.config import NavigationConfig
from bench.execution.planning.nav_algo import (
    OccupancyGrid,
    astar_path,
    base_footprint_collision_summary,
    base_motion_sweep_collision_summary,
    base_path_sweep_collision_summary,
    build_occupancy_grid,
    point_polygon_distance,
    simplify_waypoints,
)


ObstacleProvider = Callable[[], Sequence[Mapping[str, Any]]]


@dataclass
class StationPathPlanner:
    """Plan station approach paths from precomputed obstacle polygons.

    This mirrors the source navigation semantics: A* searches over the base
    center on an inflated occupancy grid, then exact rectangular-base footprint
    checks gate the predock/final docking poses.
    """

    obstacles: Sequence[Mapping[str, Any]] | ObstacleProvider
    config: NavigationConfig = field(default_factory=NavigationConfig)
    extra_include_points: Sequence[Sequence[float] | np.ndarray] = ()

    def plan(
        self,
        *,
        start_xy: np.ndarray | Sequence[float],
        start_yaw: float | None = None,
        target_xy: np.ndarray | Sequence[float],
        target_yaw: float,
        station_name: str = "",
        include_points: Sequence[Sequence[float] | np.ndarray] = (),
    ) -> dict[str, Any]:
        plan_start = time.perf_counter()
        timing: dict[str, float] = {}

        def mark(name: str, start: float) -> None:
            timing[name] = float(time.perf_counter() - start)

        start_xy = np.array(start_xy, dtype=float)[:2]
        final_goal_xy = np.array(target_xy, dtype=float)[:2]
        target_yaw = float(target_yaw)

        dock_enabled = bool(self.config.station_nav_use_docking)
        dock_standoff = float(self.config.station_nav_dock_standoff)
        predock_xy = (
            self._predock_xy(final_goal_xy, target_yaw)
            if dock_enabled and dock_standoff > 1e-6
            else final_goal_xy
        )
        planning_goal_xy = predock_xy

        try:
            phase_start = time.perf_counter()
            grid = build_occupancy_grid(
                self._obstacles(),
                include_points=self._include_points(
                    start_xy,
                    final_goal_xy,
                    planning_goal_xy,
                    include_points,
                ),
                resolution=self.config.station_nav_grid_resolution,
                inflation=self.config.station_nav_grid_inflation,
                margin=self.config.station_nav_grid_margin,
            )
            mark("build_occupancy_grid", phase_start)

            phase_start = time.perf_counter()
            planned_start_xy, planned_goal_xy, endpoint_adjustment = self._snap_endpoints(
                grid,
                start_xy=start_xy,
                goal_xy=planning_goal_xy,
                final_goal_xy=final_goal_xy,
                predock_xy=predock_xy,
            )
            mark("snap_endpoints", phase_start)
            if not bool(endpoint_adjustment.get("pass", True)):
                return {
                    "pass": False,
                    "reason": endpoint_adjustment.get("reason", "endpoint_snap_failed"),
                    "station_name": station_name,
                    "endpoint_adjustment": endpoint_adjustment,
                    "start_xy": start_xy.astype(float).tolist(),
                    "goal_xy": planning_goal_xy.astype(float).tolist(),
                    "final_goal_xy": final_goal_xy.astype(float).tolist(),
                    "timing_sec": {**timing, "total": float(time.perf_counter() - plan_start)},
                }

            phase_start = time.perf_counter()
            path = astar_path(grid, planned_start_xy, planned_goal_xy)
            mark("center_astar", phase_start)
            path["station_name"] = station_name
            path["requested_start_xy"] = start_xy.astype(float).tolist()
            path["requested_goal_xy"] = planning_goal_xy.astype(float).tolist()
            path["requested_final_xy"] = final_goal_xy.astype(float).tolist()
            path["endpoint_adjustment"] = endpoint_adjustment
            path["timing_sec"] = dict(timing)

            phase_start = time.perf_counter()
            self._annotate_docking_and_footprint(
                path,
                grid,
                predock_xy=predock_xy,
                final_goal_xy=final_goal_xy,
                target_yaw=target_yaw,
                dock_enabled=dock_enabled,
                dock_standoff=dock_standoff,
            )
            mark("dock_footprint", phase_start)
            path["timing_sec"] = dict(timing)

            phase_start = time.perf_counter()
            self._annotate_path_footprint(
                path,
                grid,
                start_xy=start_xy,
                start_yaw=start_yaw,
                planning_goal_xy=planning_goal_xy,
                target_yaw=target_yaw,
            )
            mark("path_footprint", phase_start)
            path["timing_sec"] = dict(timing)

            phase_start = time.perf_counter()
            self._annotate_grid(path, grid)
            mark("annotate_grid", phase_start)
            path["timing_sec"] = dict(timing)
            if self._should_footprint_replan(path):
                initial_path_summary = self._replan_candidate_summary(path)
                phase_start = time.perf_counter()
                replan_path, replan_summary = self._try_via_point_replan(
                    path,
                    grid,
                    planned_start_xy=planned_start_xy,
                    planned_goal_xy=planned_goal_xy,
                    start_xy=start_xy,
                    start_yaw=start_yaw,
                    planning_goal_xy=planning_goal_xy,
                    final_goal_xy=final_goal_xy,
                    target_yaw=target_yaw,
                    station_name=station_name,
                    endpoint_adjustment=endpoint_adjustment,
                    predock_xy=predock_xy,
                    dock_enabled=dock_enabled,
                    dock_standoff=dock_standoff,
                    initial_path_summary=initial_path_summary,
                )
                mark("footprint_replan", phase_start)
                if bool(replan_summary["accepted"]):
                    path = replan_path
                path["footprint_replan"] = replan_summary
                path["timing_sec"] = dict(timing)
        except Exception as exc:
            return {
                "pass": False,
                "reason": "station_path_planning_exception",
                "error": str(exc),
                "station_name": station_name,
                "start_xy": start_xy.astype(float).tolist(),
                "goal_xy": planning_goal_xy.astype(float).tolist(),
                "final_goal_xy": final_goal_xy.astype(float).tolist(),
                "timing_sec": {**timing, "total": float(time.perf_counter() - plan_start)},
            }

        astar_pass = bool(path.get("pass", False))
        footprint = path.get("footprint", {})
        footprint_pass = bool(footprint.get("pass", True))
        require_footprint = bool(self.config.station_nav_require_footprint)
        path["astar_pass"] = astar_pass
        path["checks"] = {
            "astar_path": astar_pass,
            "rectangular_base_footprint": (not require_footprint) or footprint_pass,
        }
        path["footprint_required"] = require_footprint
        path["pass"] = bool(astar_pass and ((not require_footprint) or footprint_pass))
        if astar_pass and not path["pass"]:
            path["reason"] = "station_footprint_check_failed"
        if path["pass"]:
            self._finalize_sparse_waypoints(path, start_xy, planning_goal_xy, final_goal_xy)
        timing["total"] = float(time.perf_counter() - plan_start)
        path["timing_sec"] = dict(timing)
        path.pop("waypoints", None)
        return path

    def _should_footprint_replan(self, path: dict[str, Any]) -> bool:
        if not bool(self.config.station_nav_footprint_replan):
            return False
        if not bool(self.config.station_nav_require_footprint):
            return False
        if not bool(path.get("pass", False)):
            return False
        footprint = path.get("footprint") or {}
        path_sweep = footprint.get("path_sweep") or {}
        if bool(path_sweep.get("pass", True)):
            return False
        return bool(
            (footprint.get("final_pose") or {}).get("pass", False)
            and (footprint.get("predock_pose") or {}).get("pass", False)
            and (footprint.get("dock_sweep") or {}).get("pass", False)
        )

    def _replan_candidate_summary(self, path: dict[str, Any]) -> dict[str, Any]:
        footprint = path.get("footprint") or {}
        path_sweep = footprint.get("path_sweep") or {}
        first_collision = path_sweep.get("first_collision")
        return {
            "planner": path.get("planner", "center_astar"),
            "pass": bool(path.get("pass", False)),
            "reason": path.get("reason"),
            "path_sweep_pass": path_sweep.get("pass"),
            "first_collision": first_collision,
            "path_length": path.get("path_length"),
            "raw_waypoint_count": path.get("raw_waypoint_count"),
            "sparse_waypoint_count": path.get("sparse_waypoint_count"),
            "expanded_cells": path.get("expanded_cells"),
            "segments": path.get("segments"),
            "timing_sec": path.get("timing_sec"),
            "sparse_waypoints": path.get("sparse_waypoints"),
        }

    def _try_via_point_replan(
        self,
        initial_path: dict[str, Any],
        grid: OccupancyGrid,
        *,
        planned_start_xy: np.ndarray,
        planned_goal_xy: np.ndarray,
        start_xy: np.ndarray,
        start_yaw: float | None,
        planning_goal_xy: np.ndarray,
        final_goal_xy: np.ndarray,
        target_yaw: float,
        station_name: str,
        endpoint_adjustment: dict[str, Any],
        predock_xy: np.ndarray,
        dock_enabled: bool,
        dock_standoff: float,
        initial_path_summary: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        accepted_path: dict[str, Any] | None = None
        replan_start = time.perf_counter()
        candidate_start = time.perf_counter()
        candidates = self._via_point_candidates(initial_path, grid)
        candidate_duration = float(time.perf_counter() - candidate_start)
        max_candidates = max(1, int(self.config.station_nav_replan_max_candidates))
        max_duration = max(0.0, float(self.config.station_nav_replan_max_duration_sec))
        stopped_reason: str | None = None
        self._debug_log(
            "footprint replan candidates={} max_candidates={} candidate_duration={:.3f}s".format(
                len(candidates),
                max_candidates,
                candidate_duration,
            )
        )
        for candidate in candidates[:max_candidates]:
            if max_duration > 0.0 and time.perf_counter() - replan_start >= max_duration:
                stopped_reason = "max_duration"
                break
            attempt_start = time.perf_counter()
            via_xy = np.array(candidate["xy"], dtype=float)[:2]
            astar_start = time.perf_counter()
            path = self._astar_path_through_points(
                grid,
                [planned_start_xy, via_xy, planned_goal_xy],
                planner_name="via_point_astar",
                max_expanded_per_segment=int(
                    self.config.station_nav_replan_segment_max_expanded
                ),
                sparse_spacing=float(self.config.station_nav_replan_sparse_spacing),
            )
            astar_duration = float(time.perf_counter() - astar_start)
            path["station_name"] = station_name
            path["requested_start_xy"] = start_xy.astype(float).tolist()
            path["requested_goal_xy"] = planning_goal_xy.astype(float).tolist()
            path["requested_final_xy"] = final_goal_xy.astype(float).tolist()
            path["endpoint_adjustment"] = endpoint_adjustment
            path["via_point"] = candidate
            dock_start = time.perf_counter()
            self._annotate_docking_and_footprint(
                path,
                grid,
                predock_xy=predock_xy,
                final_goal_xy=final_goal_xy,
                target_yaw=target_yaw,
                dock_enabled=dock_enabled,
                dock_standoff=dock_standoff,
            )
            dock_duration = float(time.perf_counter() - dock_start)
            footprint_start = time.perf_counter()
            self._annotate_path_footprint(
                path,
                grid,
                start_xy=start_xy,
                start_yaw=start_yaw,
                planning_goal_xy=planning_goal_xy,
                target_yaw=target_yaw,
            )
            footprint_duration = float(time.perf_counter() - footprint_start)
            self._annotate_grid(path, grid)
            attempt = {
                **candidate,
                "duration_sec": float(time.perf_counter() - attempt_start),
                "astar_duration_sec": astar_duration,
                "dock_footprint_duration_sec": dock_duration,
                "path_footprint_duration_sec": footprint_duration,
                "path": self._replan_candidate_summary(path),
            }
            attempts.append(attempt)
            self._debug_log(
                "footprint replan attempt {} pass={} reason={} sweep={} duration={:.3f}s".format(
                    len(attempts),
                    bool(path.get("pass", False)),
                    path.get("reason"),
                    ((path.get("footprint") or {}).get("path_sweep") or {}).get("pass"),
                    float(attempt["duration_sec"]),
                )
            )
            if bool(path.get("pass", False)) and bool((path.get("footprint") or {}).get("pass", False)):
                accepted_path = path
                break
            if max_duration > 0.0 and time.perf_counter() - replan_start >= max_duration:
                stopped_reason = "max_duration"
                break

        summary = {
            "enabled": True,
            "strategy": "via_point_astar",
            "trigger": "path_sweep_failed",
            "accepted": accepted_path is not None,
            "initial_path": initial_path_summary,
            "candidate_count": len(candidates),
            "max_candidates": int(max_candidates),
            "max_duration_sec": float(max_duration),
            "duration_sec": float(time.perf_counter() - replan_start),
            "candidate_duration_sec": candidate_duration,
            "stopped_reason": stopped_reason,
            "segment_max_expanded": int(
                self.config.station_nav_replan_segment_max_expanded
            ),
            "sparse_spacing": float(self.config.station_nav_replan_sparse_spacing),
            "attempt_count": len(attempts),
            "attempts": attempts,
            "candidate": self._replan_candidate_summary(accepted_path or initial_path),
        }
        return accepted_path or initial_path, summary

    def _via_point_candidates(
        self,
        path: dict[str, Any],
        grid: OccupancyGrid,
    ) -> list[dict[str, Any]]:
        first_collision = (
            ((path.get("footprint") or {}).get("path_sweep") or {}).get("first_collision")
            or {}
        )
        collision_xy_value = first_collision.get("start_xy") or first_collision.get("goal_xy")
        if collision_xy_value is None:
            return []
        collision_xy = np.array(collision_xy_value, dtype=float)[:2]
        obstacle = self._collision_obstacle(first_collision, grid)
        if obstacle is None or obstacle.get("polygon") is None:
            return []
        poly = np.array(obstacle["polygon"], dtype=float)
        center = np.mean(poly, axis=0)
        outward = collision_xy - center
        norm = float(np.linalg.norm(outward))
        if norm < 1e-6:
            nearest = self._nearest_polygon_vertex(collision_xy, poly)
            outward = collision_xy - nearest
            norm = float(np.linalg.norm(outward))
        if norm < 1e-6:
            return []
        outward = outward / norm
        tangent = np.array([-outward[1], outward[0]], dtype=float)
        collision_start_yaw = first_collision.get("start_yaw")
        forward = None
        if collision_start_yaw is not None and np.isfinite(float(collision_start_yaw)):
            forward = np.array(
                [np.cos(float(collision_start_yaw)), np.sin(float(collision_start_yaw))],
                dtype=float,
            )
        distances = (0.20, 0.32, 0.44, 0.60, 0.76, 0.92)
        laterals = (0.0, -0.20, 0.20, -0.36, 0.36)
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        raw_candidates: list[dict[str, Any]] = []
        if forward is not None:
            for distance in distances:
                for lateral in (0.0, -0.16, 0.16):
                    raw_candidates.append(
                        {
                            "raw_xy": collision_xy
                            + forward * float(distance)
                            + outward * float(lateral),
                            "distance": float(distance),
                            "lateral": float(lateral),
                            "priority": 0,
                            "strategy": "delay_turn_along_current_heading",
                        }
                    )
        for distance in distances:
            for lateral in laterals:
                raw_candidates.append(
                    {
                        "raw_xy": collision_xy
                        + outward * float(distance)
                        + tangent * float(lateral),
                        "distance": float(distance),
                        "lateral": float(lateral),
                        "priority": 1,
                        "strategy": "move_away_from_collision_obstacle",
                    }
                )
        for raw_candidate in raw_candidates:
                raw_xy = np.array(raw_candidate["raw_xy"], dtype=float)[:2]
                snap = self._nearest_free_grid_xy(
                    grid,
                    raw_xy,
                    max_radius=max(0.10, 2.0 * float(grid.resolution)),
                )
                if not bool(snap.get("pass", False)):
                    continue
                xy = np.array(snap["xy"], dtype=float)[:2]
                cell = grid.world_to_cell(xy)
                if cell in seen:
                    continue
                seen.add(cell)
                footprint = base_footprint_collision_summary(
                    xy,
                    float(first_collision.get("start_yaw", 0.0)),
                    grid.obstacles,
                    **self.config.footprint_kwargs(),
                )
                if not bool(footprint.get("pass", False)):
                    continue
                candidates.append(
                    {
                        "xy": xy.astype(float).tolist(),
                        "cell": [int(cell[0]), int(cell[1])],
                        "raw_xy": raw_xy.astype(float).tolist(),
                        "snap": snap,
                        "source_collision_xy": collision_xy.astype(float).tolist(),
                        "source_obstacle": obstacle.get("name"),
                        "distance": float(raw_candidate["distance"]),
                        "lateral": float(raw_candidate["lateral"]),
                        "priority": int(raw_candidate["priority"]),
                        "strategy": str(raw_candidate["strategy"]),
                        "obstacle_center": center.astype(float).tolist(),
                        "outward": outward.astype(float).tolist(),
                        "nearest_obstacle_distance": point_polygon_distance(xy, poly),
                    }
                )
        candidates.sort(
            key=lambda item: (
                int(item.get("priority", 10)),
                float(item.get("distance", 0.0)) + 0.25 * abs(float(item.get("lateral", 0.0))),
                -float(item.get("nearest_obstacle_distance", 0.0)),
            )
        )
        return candidates

    def _collision_obstacle(
        self,
        first_collision: dict[str, Any],
        grid: OccupancyGrid,
    ) -> dict[str, Any] | None:
        nearest = first_collision.get("nearest_obstacle") or {}
        name = nearest.get("name")
        if name is None:
            nearest = (
                ((first_collision.get("sweep") or {}).get("first_collision") or {}).get(
                    "nearest_obstacle"
                )
                or {}
            )
            name = nearest.get("name")
        for obstacle in grid.obstacles:
            if obstacle.get("name") == name:
                return obstacle
        return None

    def _nearest_polygon_vertex(self, xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
        return poly[
            int(
                min(
                    range(len(poly)),
                    key=lambda index: float(np.linalg.norm(poly[index] - xy)),
                )
            )
        ]

    def _astar_path_through_points(
        self,
        grid: OccupancyGrid,
        points: Sequence[np.ndarray | Sequence[float]],
        *,
        planner_name: str,
        max_expanded_per_segment: int | None = None,
        sparse_spacing: float | None = None,
    ) -> dict[str, Any]:
        arrays = [np.array(point, dtype=float)[:2] for point in points]
        combined_waypoints: list[list[float]] = []
        segment_summaries: list[dict[str, Any]] = []
        total_length = 0.0
        expanded = 0
        for index, (start_xy, goal_xy) in enumerate(zip(arrays, arrays[1:])):
            segment = astar_path(
                grid,
                start_xy,
                goal_xy,
                max_expanded=max_expanded_per_segment,
            )
            segment_summaries.append(
                {
                    "index": int(index),
                    "pass": bool(segment.get("pass", False)),
                    "reason": segment.get("reason"),
                    "start_xy": segment.get("start_xy"),
                    "goal_xy": segment.get("goal_xy"),
                    "path_length": segment.get("path_length"),
                    "raw_waypoint_count": segment.get("raw_waypoint_count"),
                    "expanded_cells": segment.get("expanded_cells"),
                    "duration_sec": segment.get("duration_sec"),
                }
            )
            expanded += int(segment.get("expanded_cells", 0) or 0)
            if not bool(segment.get("pass", False)):
                return {
                    "planner": planner_name,
                    "pass": False,
                    "reason": f"via_segment_{index}_{segment.get('reason', 'failed')}",
                    "segments": segment_summaries,
                    "expanded_cells": int(expanded),
                    "start_xy": arrays[0].astype(float).tolist(),
                    "goal_xy": arrays[-1].astype(float).tolist(),
                }
            waypoints = [
                np.array(point, dtype=float)[:2].astype(float).tolist()
                for point in list(segment.get("waypoints") or segment.get("sparse_waypoints") or [])
            ]
            if combined_waypoints and waypoints:
                waypoints = waypoints[1:]
            combined_waypoints.extend(waypoints)
            total_length += float(segment.get("path_length", 0.0) or 0.0)
        sparse = simplify_waypoints(
            combined_waypoints,
            min_spacing=max(
                0.04,
                float(sparse_spacing if sparse_spacing is not None else 0.12),
            ),
        )
        return {
            "planner": planner_name,
            "pass": True,
            "reason": "path_found",
            "start_xy": arrays[0].astype(float).tolist(),
            "goal_xy": arrays[-1].astype(float).tolist(),
            "expanded_cells": int(expanded),
            "raw_cell_count": len(combined_waypoints),
            "raw_waypoint_count": len(combined_waypoints),
            "sparse_waypoint_count": len(sparse),
            "path_length": float(total_length),
            "waypoints": combined_waypoints,
            "sparse_waypoints": sparse,
            "segments": segment_summaries,
        }

    def _debug_log(self, message: str) -> None:
        if bool(self.config.station_nav_debug):
            print(f"[station-planner] {message}", file=sys.stderr, flush=True)

    def _obstacles(self) -> Sequence[Mapping[str, Any]]:
        return self.obstacles() if callable(self.obstacles) else self.obstacles

    def _include_points(
        self,
        start_xy: np.ndarray,
        final_goal_xy: np.ndarray,
        planning_goal_xy: np.ndarray,
        include_points: Sequence[Sequence[float] | np.ndarray],
    ) -> list[np.ndarray]:
        points = [
            np.array(start_xy, dtype=float)[:2],
            np.array(final_goal_xy, dtype=float)[:2],
            np.array(planning_goal_xy, dtype=float)[:2],
        ]
        for point in list(self.extra_include_points) + list(include_points):
            arr = np.array(point, dtype=float).reshape(-1)
            if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
                points.append(arr[:2])
        return points

    def _predock_xy(self, target_xy: np.ndarray, target_yaw: float) -> np.ndarray:
        standoff = float(self.config.station_nav_dock_standoff)
        forward = np.array([np.cos(float(target_yaw)), np.sin(float(target_yaw))], dtype=float)
        return np.array(target_xy, dtype=float)[:2] - forward * standoff

    def _snap_endpoints(
        self,
        grid: OccupancyGrid,
        *,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        final_goal_xy: np.ndarray,
        predock_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        endpoint_adjustment = {
            "enabled": bool(self.config.station_nav_allow_endpoint_snap),
            "snap_radius": float(self.config.station_nav_endpoint_snap_radius),
            "requested_start_xy": start_xy.astype(float).tolist(),
            "requested_goal_xy": goal_xy.astype(float).tolist(),
            "requested_final_xy": final_goal_xy.astype(float).tolist(),
            "requested_predock_xy": predock_xy.astype(float).tolist(),
            "pass": True,
        }
        if not endpoint_adjustment["enabled"]:
            return start_xy, goal_xy, endpoint_adjustment

        start_snap = self._nearest_free_grid_xy(
            grid,
            start_xy,
            max_radius=float(endpoint_adjustment["snap_radius"]),
        )
        goal_snap = self._nearest_free_grid_xy(
            grid,
            goal_xy,
            max_radius=float(endpoint_adjustment["snap_radius"]),
        )
        endpoint_adjustment["start"] = start_snap
        endpoint_adjustment["goal"] = goal_snap
        if not bool(start_snap.get("pass", False)):
            endpoint_adjustment["pass"] = False
            endpoint_adjustment["reason"] = "start_endpoint_snap_failed"
            return start_xy, goal_xy, endpoint_adjustment
        if not bool(goal_snap.get("pass", False)):
            endpoint_adjustment["pass"] = False
            endpoint_adjustment["reason"] = "goal_endpoint_snap_failed"
            return start_xy, goal_xy, endpoint_adjustment
        return (
            np.array(start_snap["xy"], dtype=float),
            np.array(goal_snap["xy"], dtype=float),
            endpoint_adjustment,
        )

    def _nearest_free_grid_xy(
        self,
        grid: OccupancyGrid,
        xy: np.ndarray,
        *,
        max_radius: float,
    ) -> dict[str, Any]:
        query_xy = np.array(xy, dtype=float)[:2]
        query_cell = grid.world_to_cell(query_xy)
        if grid.in_bounds(query_cell) and grid.is_free_cell(query_cell):
            return {
                "pass": True,
                "snapped": False,
                "xy": query_xy.astype(float).tolist(),
                "cell": [int(query_cell[0]), int(query_cell[1])],
                "distance": 0.0,
            }

        max_cells = int(np.ceil(float(max_radius) / max(float(grid.resolution), 1e-9)))
        best: dict[str, Any] | None = None
        for dr in range(-max_cells, max_cells + 1):
            for dc in range(-max_cells, max_cells + 1):
                cell = (int(query_cell[0] + dr), int(query_cell[1] + dc))
                if not grid.is_free_cell(cell):
                    continue
                candidate_xy = grid.cell_to_world(cell)
                distance = float(np.linalg.norm(candidate_xy - query_xy))
                if distance > float(max_radius):
                    continue
                item = {
                    "pass": True,
                    "snapped": True,
                    "xy": candidate_xy.astype(float).tolist(),
                    "cell": [int(cell[0]), int(cell[1])],
                    "distance": distance,
                    "query_cell": [int(query_cell[0]), int(query_cell[1])],
                }
                if best is None or distance < float(best["distance"]):
                    best = item
        if best is not None:
            return best
        return {
            "pass": False,
            "snapped": False,
            "xy": query_xy.astype(float).tolist(),
            "query_cell": [int(query_cell[0]), int(query_cell[1])],
            "max_radius": float(max_radius),
            "reason": "no_free_cell_within_radius",
        }

    def _annotate_docking_and_footprint(
        self,
        path: dict[str, Any],
        grid: OccupancyGrid,
        *,
        predock_xy: np.ndarray,
        final_goal_xy: np.ndarray,
        target_yaw: float,
        dock_enabled: bool,
        dock_standoff: float,
    ) -> None:
        footprint_kwargs = self.config.footprint_kwargs()
        final_footprint = base_footprint_collision_summary(
            final_goal_xy,
            float(target_yaw),
            grid.obstacles,
            **footprint_kwargs,
        )
        predock_footprint = base_footprint_collision_summary(
            predock_xy,
            float(target_yaw),
            grid.obstacles,
            **footprint_kwargs,
        )
        dock_sweep = base_motion_sweep_collision_summary(
            predock_xy,
            float(target_yaw),
            final_goal_xy,
            float(target_yaw),
            grid.obstacles,
            **footprint_kwargs,
        )
        path["docking"] = {
            "enabled": bool(dock_enabled and dock_standoff > 1e-6),
            "standoff": float(dock_standoff),
            "predock_xy": predock_xy.astype(float).tolist(),
            "final_xy": final_goal_xy.astype(float).tolist(),
            "target_yaw": float(target_yaw),
            "planning_goal_xy": predock_xy.astype(float).tolist(),
        }
        path["footprint"] = {
            "enabled": True,
            "final_pose": final_footprint,
            "predock_pose": predock_footprint,
            "dock_sweep": dock_sweep,
            "pass": bool(
                final_footprint.get("pass", False)
                and predock_footprint.get("pass", False)
                and dock_sweep.get("pass", False)
            ),
        }

    def _annotate_grid(self, path: dict[str, Any], grid: OccupancyGrid) -> None:
        grid_summary = grid.summary()
        path["grid"] = {
            key: grid_summary[key]
            for key in (
                "origin",
                "resolution",
                "width",
                "height",
                "bounds_min",
                "bounds_max",
                "inflation",
                "occupied_cell_count",
                "free_cell_count",
                "occupied_fraction",
            )
            if key in grid_summary
        }

    def _annotate_path_footprint(
        self,
        path: dict[str, Any],
        grid: OccupancyGrid,
        *,
        start_xy: np.ndarray,
        start_yaw: float | None,
        planning_goal_xy: np.ndarray,
        target_yaw: float,
    ) -> None:
        footprint = path.setdefault("footprint", {})
        if not bool(path.get("pass", False)):
            footprint["path_sweep"] = {
                "pass": False,
                "collision": False,
                "reason": "skipped_astar_failed",
            }
            footprint["pass"] = bool(footprint.get("pass", False))
            return
        waypoints = self._path_sweep_waypoints(path, start_xy, planning_goal_xy)
        path_sweep = base_path_sweep_collision_summary(
            waypoints,
            grid.obstacles,
            start_yaw=start_yaw,
            terminal_yaw=float(target_yaw),
            heading_tolerance=float(self.config.base_heading_tolerance),
            **self.config.footprint_kwargs(),
        )
        footprint["path_sweep"] = path_sweep
        footprint["pass"] = bool(
            footprint.get("pass", False) and path_sweep.get("pass", False)
        )

    def _path_sweep_waypoints(
        self,
        path: dict[str, Any],
        start_xy: np.ndarray,
        planning_goal_xy: np.ndarray,
    ) -> list[list[float]]:
        sparse = [
            np.array(point, dtype=float)[:2].astype(float).tolist()
            for point in list(path.get("sparse_waypoints", []))
        ]
        if not sparse:
            sparse = [start_xy.astype(float).tolist()]

        checked = list(sparse)
        if float(np.linalg.norm(np.array(checked[0], dtype=float) - start_xy)) > 1e-6:
            checked.insert(0, start_xy.astype(float).tolist())
        if float(np.linalg.norm(np.array(checked[-1], dtype=float) - planning_goal_xy)) > 1e-6:
            checked.append(planning_goal_xy.astype(float).tolist())
        return checked

    def _finalize_sparse_waypoints(
        self,
        path: dict[str, Any],
        start_xy: np.ndarray,
        planning_goal_xy: np.ndarray,
        final_goal_xy: np.ndarray,
    ) -> None:
        sparse = [
            np.array(point, dtype=float)[:2].astype(float).tolist()
            for point in list(path.get("sparse_waypoints", []))
        ]
        if not sparse:
            sparse = [start_xy.astype(float).tolist()]
        if float(np.linalg.norm(np.array(sparse[-1], dtype=float) - planning_goal_xy)) > 1e-6:
            sparse.append(planning_goal_xy.astype(float).tolist())
        path["sparse_waypoints"] = sparse
        path["terminal_target_xy"] = final_goal_xy.astype(float).tolist()
        path["sparse_waypoint_count"] = len(sparse)
