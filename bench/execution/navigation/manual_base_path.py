"""Manual mobile-base path execution for task-owned waypoint routes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import (
    base_yaw,
    controller_translation_yaw,
    qpos_for_controller,
    step_base_and_hold_arm,
    world_xy_to_controller_action,
    world_vec_to_base,
)
from bench.execution.navigation.velocity_profile import (
    AxialVelocityProfile,
    BaseVelocityLimits,
    PlanarVelocityProfile,
    resolve_control_dt,
)
from bench.execution.planning.nav_algo import normalize_angle


@dataclass(frozen=True)
class ManualBasePathSample:
    index: int
    xy: np.ndarray
    yaw: float
    name: str | None = None


def _clip_action_norm(values: np.ndarray, *, max_norm: float = 1.0) -> np.ndarray:
    values = np.array(values, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm > float(max_norm) > 0.0:
        return values / norm * float(max_norm)
    return values


def _load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _sample_from_payload(item: dict[str, Any], fallback_index: int) -> ManualBasePathSample:
    index = int(item.get("index", item.get("sample_index", fallback_index)))
    xy = np.array(item.get("xy", item.get("base_xy", item.get("base_pos", [])[:2])), dtype=float)[:2]
    if xy.shape[0] != 2 or not np.all(np.isfinite(xy)):
        raise ValueError(f"manual base path sample {index} has invalid xy")
    yaw = float(item.get("yaw", item.get("base_yaw")))
    if not np.isfinite(yaw):
        raise ValueError(f"manual base path sample {index} has invalid yaw")
    name = item.get("name")
    return ManualBasePathSample(index=index, xy=xy, yaw=yaw, name=None if name is None else str(name))


def _normalize_key(value: Any) -> str:
    return str(value).strip()


class ManualBasePathExecutor:
    """Execute named navigation goals by traversing a calibrated base path."""

    def __init__(
        self,
        env: Any,
        *,
        path_file: str | Path,
        start_index: int = 1,
        xy_tolerance: float = 0.07,
        yaw_tolerance: float = 0.08,
        segment_yaw_threshold: float = 0.25,
        max_rotate_steps: int = 350,
        max_translate_steps: int = 900,
        xy_gain: float = 3.0,
        yaw_gain: float = 2.0,
        nearest_yaw_weight: float = 0.20,
        rotate_xy_tolerance: float = 0.10,
        intermediate_xy_tolerance: float | None = None,
        intermediate_yaw_tolerance: float | None = None,
        velocity_limits: BaseVelocityLimits | None = None,
    ) -> None:
        self.env = env
        self.path_file = Path(path_file)
        payload = _load_payload(self.path_file)
        raw_samples = list(payload.get("samples", []))
        if len(raw_samples) < 2:
            raise ValueError(f"manual base path {self.path_file} needs at least two samples")
        samples = [_sample_from_payload(item, i) for i, item in enumerate(raw_samples)]
        samples.sort(key=lambda item: item.index)
        expected = list(range(len(samples)))
        actual = [item.index for item in samples]
        if actual != expected:
            raise ValueError(f"manual base path indexes must be contiguous {expected}; got {actual}")

        self.samples = samples
        self.station_targets = {
            _normalize_key(key): value for key, value in dict(payload.get("station_targets", {})).items()
        }
        self.default_start_index = int(payload.get("default_start_index", start_index))
        self.route_mode = _normalize_key(payload.get("route_mode", "bidirectional")).lower()
        self.current_index: int | None = None
        self.xy_tolerance = float(xy_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        self.segment_yaw_threshold = float(segment_yaw_threshold)
        self.max_rotate_steps = int(max_rotate_steps)
        self.max_translate_steps = int(max_translate_steps)
        self.xy_gain = float(xy_gain)
        self.yaw_gain = float(yaw_gain)
        self.nearest_yaw_weight = float(nearest_yaw_weight)
        self.rotate_xy_tolerance = max(0.0, float(rotate_xy_tolerance))
        self.intermediate_xy_tolerance = (
            None if intermediate_xy_tolerance is None else float(intermediate_xy_tolerance)
        )
        self.intermediate_yaw_tolerance = (
            None if intermediate_yaw_tolerance is None else float(intermediate_yaw_tolerance)
        )
        for name, value in (
            ("intermediate_xy_tolerance", self.intermediate_xy_tolerance),
            ("intermediate_yaw_tolerance", self.intermediate_yaw_tolerance),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"manual base {name} must be positive and finite")
        self.velocity_limits = (
            velocity_limits if velocity_limits is not None and velocity_limits.enabled else None
        )
        self._control_dt = resolve_control_dt(env)

    def target_index_for(
        self,
        station_name: str,
        *,
        purpose: str | None = None,
        holding: str | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        keys = []
        station = _normalize_key(station_name)
        if purpose:
            keys.append(f"{station}:{_normalize_key(purpose)}")
        if holding:
            keys.append(f"{station}:holding:{_normalize_key(holding)}")
        keys.append(station)
        for key in keys:
            if key not in self.station_targets:
                continue
            value = self.station_targets[key]
            explicit_routes = None
            if isinstance(value, dict):
                raw_indexes = value.get("indexes")
                if raw_indexes is not None:
                    indexes = [int(item) for item in raw_indexes]
                    if not indexes:
                        raise ValueError("manual base station target indexes must be non-empty")
                    current = self.current_index
                    if current is None:
                        index = indexes[0]
                    else:
                        index = min(indexes, key=lambda item: abs(int(item) - int(current)))
                else:
                    index = int(value.get("index", value.get("target_index")))
                explicit_routes = value.get("routes")
            elif isinstance(value, list):
                indexes = [int(item) for item in value]
                current = self.current_index
                if current is None:
                    index = indexes[0]
                else:
                    index = min(indexes, key=lambda item: abs(int(item) - int(current)))
            else:
                index = int(value)
            if 0 <= index < len(self.samples):
                resolution = {"key": key, "value": value}
                if isinstance(explicit_routes, dict):
                    resolution["routes"] = explicit_routes
                return index, resolution
            return None, {"key": key, "value": value, "reason": "target_index_out_of_range"}
        return None, {"keys": keys, "reason": "manual_base_path_target_unavailable"}

    def nearest_index(self, obs: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        yaw = float(base_yaw(obs))
        scored = []
        for sample in self.samples:
            xy_dist = float(np.linalg.norm(xy - sample.xy))
            yaw_dist = abs(normalize_angle(yaw - sample.yaw))
            cost = xy_dist + self.nearest_yaw_weight * yaw_dist
            scored.append(
                {
                    "index": int(sample.index),
                    "xy_dist": xy_dist,
                    "yaw_dist": yaw_dist,
                    "cost": cost,
                }
            )
        best = min(scored, key=lambda item: float(item["cost"]))
        return int(best["index"]), {"method": "nearest_xy_yaw", "best": best, "candidates": scored}

    def move_to(
        self,
        obs: dict[str, Any],
        station_name: str,
        *,
        holding: str | None = None,
        purpose: str | None = None,
        gripper_action: float | None = None,
        force_terminal_settle: bool = False,
        terminal_xy_tolerance: float | None = None,
        terminal_yaw_tolerance: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        resolved_terminal_xy_tolerance = (
            self.xy_tolerance
            if terminal_xy_tolerance is None
            else float(terminal_xy_tolerance)
        )
        resolved_terminal_yaw_tolerance = (
            self.yaw_tolerance
            if terminal_yaw_tolerance is None
            else float(terminal_yaw_tolerance)
        )
        if (
            not np.isfinite(resolved_terminal_xy_tolerance)
            or resolved_terminal_xy_tolerance <= 0.0
            or not np.isfinite(resolved_terminal_yaw_tolerance)
            or resolved_terminal_yaw_tolerance <= 0.0
        ):
            raise ValueError("manual path terminal tolerances must be positive and finite")
        target_index, target_resolution = self.target_index_for(
            station_name,
            purpose=purpose,
            holding=holding,
        )
        if target_index is None:
            return obs, {
                "primitive": "manual_base_path_navigate",
                "pass": False,
                "station_name": str(station_name),
                "purpose": purpose,
                "holding": holding,
                "target_resolution": target_resolution,
                "reason": target_resolution.get("reason", "manual_base_path_target_unavailable"),
            }

        start_index = self.current_index
        nearest_summary = None
        if start_index is None:
            start_index, nearest_summary = self.nearest_index(obs)
        if start_index is None or not (0 <= int(start_index) < len(self.samples)):
            start_index = max(0, min(len(self.samples) - 1, int(self.default_start_index)))
        start_index = int(start_index)

        route = self._explicit_route(
            start_index,
            int(target_index),
            target_resolution=target_resolution,
        )
        if route is None:
            route = self._route_indices(start_index, int(target_index))
        right = self.env.robots[0].composite_controller.part_controllers["right"]
        arm_qpos = qpos_for_controller(self.env, right)
        resolved_gripper_action = (
            float(gripper_action)
            if gripper_action is not None
            else (1.0 if holding else -1.0)
        )
        initial_held_pos = self._held_object_pos(holding)
        segments: list[dict[str, Any]] = []
        current_obs = obs
        for source_index, target_index_item in zip(route, route[1:]):
            source = self.samples[int(source_index)]
            target = self.samples[int(target_index_item)]
            is_terminal = int(target_index_item) == int(target_index)
            current_obs, segment = self._move_segment(
                current_obs,
                source=source,
                target=target,
                arm_qpos=arm_qpos,
                gripper_action=resolved_gripper_action,
                xy_tolerance=(
                    resolved_terminal_xy_tolerance
                    if is_terminal
                    else getattr(self, "intermediate_xy_tolerance", None)
                ),
                yaw_tolerance=(
                    resolved_terminal_yaw_tolerance
                    if is_terminal
                    else getattr(self, "intermediate_yaw_tolerance", None)
                ),
            )
            segment["terminal_segment"] = bool(is_terminal)
            segments.append(segment)
            if not bool(segment.get("pass", False)):
                break

        if bool(force_terminal_settle) and segments and not all(
            bool(item.get("pass", False)) for item in segments
        ):
            failed_segment = segments[-1]
            failed_final_dist = float(failed_segment.get("final_dist", float("inf")))
            if failed_final_dist <= max(self.xy_tolerance, 0.12):
                current_obs, terminal_settle = self._settle_at_sample_with_tolerances(
                    current_obs,
                    target=self.samples[int(target_index)],
                    arm_qpos=arm_qpos,
                    gripper_action=resolved_gripper_action,
                    xy_tolerance=resolved_terminal_xy_tolerance,
                    yaw_tolerance=resolved_terminal_yaw_tolerance,
                )
                terminal_settle["forced_after_terminal_segment_miss"] = True
                segments.append(terminal_settle)
                if bool(terminal_settle.get("pass", False)):
                    failed_segment["accepted_after_forced_terminal_settle"] = True
                    failed_segment["pass"] = True

        if not segments or all(bool(item.get("pass", False)) for item in segments):
            current_obs, settle = self._settle_at_sample_with_tolerances(
                current_obs,
                target=self.samples[int(target_index)],
                arm_qpos=arm_qpos,
                gripper_action=resolved_gripper_action,
                xy_tolerance=resolved_terminal_xy_tolerance,
                yaw_tolerance=resolved_terminal_yaw_tolerance,
            )
            segments.append(settle)

        final_sample = self.samples[int(target_index)]
        final_xy = np.array(current_obs["robot0_base_pos"][:2], dtype=float)
        final_yaw = float(base_yaw(current_obs))
        final_dist = float(np.linalg.norm(final_sample.xy - final_xy))
        final_yaw_error = abs(normalize_angle(final_sample.yaw - final_yaw))
        held_motion = self._held_motion(holding, initial_held_pos)
        all_segments = bool(segments) and all(bool(item.get("pass", False)) for item in segments)
        checks = {
            "segments": all_segments,
            "final_xy": final_dist <= resolved_terminal_xy_tolerance,
            "final_yaw": final_yaw_error <= resolved_terminal_yaw_tolerance,
            "holding_motion": not held_motion.get("enabled", False) or bool(held_motion.get("drop_ok", False)),
        }
        passed = bool(all(checks.values()))
        if passed:
            self.current_index = int(target_index)
        else:
            self.current_index, _ = self.nearest_index(current_obs)

        return current_obs, {
            "primitive": "manual_base_path_navigate",
            "strategy": "manual_index_path",
            "pass": passed,
            "station_name": str(station_name),
            "purpose": purpose,
            "holding": holding,
            "force_terminal_settle": bool(force_terminal_settle),
            "path_file": str(self.path_file),
            "start_index": int(start_index),
            "target_index": int(target_index),
            "route_indices": [int(item) for item in route],
            "route_mode": self.route_mode,
            "target_resolution": target_resolution,
            "nearest_start": nearest_summary,
            "segments": segments,
            "final_xy": final_xy.astype(float).tolist(),
            "final_yaw": final_yaw,
            "target_xy": final_sample.xy.astype(float).tolist(),
            "target_yaw": float(final_sample.yaw),
            "final_dist": final_dist,
            "final_yaw_error": final_yaw_error,
            "path_xy_tolerance": float(self.xy_tolerance),
            "path_yaw_tolerance": float(self.yaw_tolerance),
            "intermediate_xy_tolerance": getattr(
                self, "intermediate_xy_tolerance", None
            ),
            "intermediate_yaw_tolerance": getattr(
                self, "intermediate_yaw_tolerance", None
            ),
            "terminal_xy_tolerance": float(resolved_terminal_xy_tolerance),
            "terminal_yaw_tolerance": float(resolved_terminal_yaw_tolerance),
            "holding_motion": held_motion,
            "checks": checks,
            "reason": None if passed else "manual_base_path_failed",
        }

    def _settle_at_sample_with_tolerances(
        self,
        obs: dict[str, Any],
        *,
        target: ManualBasePathSample,
        arm_qpos: np.ndarray,
        gripper_action: float,
        xy_tolerance: float,
        yaw_tolerance: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply strict precision only while settling the terminal sample."""

        original_xy_tolerance = self.xy_tolerance
        original_yaw_tolerance = self.yaw_tolerance
        self.xy_tolerance = float(xy_tolerance)
        self.yaw_tolerance = float(yaw_tolerance)
        try:
            settled_obs, result = self._settle_at_sample(
                obs,
                target=target,
                arm_qpos=arm_qpos,
                gripper_action=gripper_action,
            )
        finally:
            self.xy_tolerance = original_xy_tolerance
            self.yaw_tolerance = original_yaw_tolerance
        result["terminal_xy_tolerance"] = float(xy_tolerance)
        result["terminal_yaw_tolerance"] = float(yaw_tolerance)
        return settled_obs, result

    def _explicit_route(
        self,
        start_index: int,
        target_index: int,
        *,
        target_resolution: dict[str, Any],
    ) -> list[int] | None:
        routes = target_resolution.get("routes")
        if not isinstance(routes, dict):
            return None
        raw = routes.get(str(int(start_index)), routes.get(int(start_index)))
        if raw is None:
            return None
        route = [int(item) for item in raw]
        if not route or route[0] != int(start_index) or route[-1] != int(target_index):
            raise ValueError(
                "manual base explicit route must start at the current index "
                "and end at the resolved target index"
            )
        if any(index < 0 or index >= len(self.samples) for index in route):
            raise ValueError("manual base explicit route contains an out-of-range index")
        return route

    def _route_indices(self, start_index: int, target_index: int) -> list[int]:
        if int(start_index) == int(target_index):
            return [int(start_index)]
        if self.route_mode in {"spatial_graph", "direct_graph", "station_graph"}:
            # Samples are spatial station vertices rather than a chronological
            # demonstration timeline. A navigation request traverses the
            # direct calibrated edge between the current and target stations;
            # target-specific explicit routes can still override this above.
            return [int(start_index), int(target_index)]
        if self.route_mode in {"directed_loop", "loop", "forward_loop"}:
            if int(target_index) < int(start_index):
                return list(range(int(start_index), len(self.samples))) + list(
                    range(0, int(target_index) + 1)
                )
            return list(range(int(start_index), int(target_index) + 1))
        step = 1 if int(target_index) > int(start_index) else -1
        return list(range(int(start_index), int(target_index) + step, step))

    def _move_segment(
        self,
        obs: dict[str, Any],
        *,
        source: ManualBasePathSample,
        target: ManualBasePathSample,
        arm_qpos: np.ndarray,
        gripper_action: float,
        xy_tolerance: float | None = None,
        yaw_tolerance: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Execute one edge using terminal or pass-through tolerances."""

        resolved_xy_tolerance = (
            self.xy_tolerance if xy_tolerance is None else float(xy_tolerance)
        )
        resolved_yaw_tolerance = (
            self.yaw_tolerance if yaw_tolerance is None else float(yaw_tolerance)
        )
        if (
            not np.isfinite(resolved_xy_tolerance)
            or resolved_xy_tolerance <= 0.0
            or not np.isfinite(resolved_yaw_tolerance)
            or resolved_yaw_tolerance <= 0.0
        ):
            raise ValueError("manual base segment tolerances must be positive and finite")
        original_xy_tolerance = self.xy_tolerance
        original_yaw_tolerance = self.yaw_tolerance
        self.xy_tolerance = resolved_xy_tolerance
        self.yaw_tolerance = resolved_yaw_tolerance
        try:
            return self._move_segment_with_tolerances(
                obs,
                source=source,
                target=target,
                arm_qpos=arm_qpos,
                gripper_action=gripper_action,
            )
        finally:
            self.xy_tolerance = original_xy_tolerance
            self.yaw_tolerance = original_yaw_tolerance

    def _move_segment_with_tolerances(
        self,
        obs: dict[str, Any],
        *,
        source: ManualBasePathSample,
        target: ManualBasePathSample,
        arm_qpos: np.ndarray,
        gripper_action: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dyaw = normalize_angle(target.yaw - source.yaw)
        dxy = float(np.linalg.norm(target.xy - source.xy))
        if abs(dyaw) > self.segment_yaw_threshold:
            if dxy > max(self.xy_tolerance, 0.15):
                # 判定是否应该"先转后走"
                # 扩大near_u_turn的判定：大角度转向（>60°）都先转后走
                large_rotation = abs(dyaw) > 1.05  # > 60度
                near_u_turn = abs(abs(dyaw) - np.pi) <= 0.20  # 接近180度

                if near_u_turn or large_rotation:
                    # 先转后走：避免错误朝向平移导致的轨迹偏差
                    obs, rotate = self._rotate_to_sample(
                        obs,
                        target=target,
                        arm_qpos=arm_qpos,
                        gripper_action=gripper_action,
                    )
                    if not bool(rotate.get("pass", False)):
                        segment = dict(rotate)
                        segment["primitive"] = "rotate_then_translate"
                        segment["pre_rotate"] = rotate
                        segment["post_translate"] = {
                            "primitive": "translate",
                            "pass": False,
                            "skipped": True,
                            "reason": "pre_rotate_failed",
                            "steps": 0,
                        }
                    else:
                        obs, post_translate = self._translate_to_sample(
                            obs,
                            target=target,
                            arm_qpos=arm_qpos,
                            gripper_action=gripper_action,
                            pass_tolerance=self.xy_tolerance,
                            best_reached_tolerance=max(self.xy_tolerance, 0.03),
                        )
                        final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
                        final_yaw = float(base_yaw(obs))
                        final_dist = float(np.linalg.norm(target.xy - final_xy))
                        final_yaw_error = abs(normalize_angle(target.yaw - final_yaw))
                        segment = {
                            "primitive": "rotate_then_translate",
                            "pre_rotate": rotate,
                            "post_translate": post_translate,
                            "target_xy": target.xy.astype(float).tolist(),
                            "target_yaw": float(target.yaw),
                            "final_xy": final_xy.astype(float).tolist(),
                            "final_yaw": final_yaw,
                            "final_dist": final_dist,
                            "final_yaw_error": final_yaw_error,
                            "pass": bool(
                                rotate.get("pass", False)
                                and post_translate.get("pass", False)
                                and final_yaw_error <= self.yaw_tolerance
                            ),
                        }
                else:
                    obs, translate = self._translate_to_sample(
                        obs,
                        target=target,
                        arm_qpos=arm_qpos,
                        gripper_action=gripper_action,
                    )
                    if bool(translate.get("pass", False)):
                        obs, rotate = self._rotate_to_sample(
                            obs,
                            target=target,
                            arm_qpos=arm_qpos,
                            gripper_action=gripper_action,
                        )
                        if bool(rotate.get("pass", False)):
                            segment = rotate
                            segment["primitive"] = "translate_then_rotate"
                            segment["pre_translate"] = translate
                        else:
                            rotate_yaw_ok = (
                                float(rotate.get("final_yaw_error", float("inf")))
                                <= self.yaw_tolerance
                            )
                            if rotate_yaw_ok:
                                obs, post_translate = self._translate_to_sample(
                                    obs,
                                    target=target,
                                    arm_qpos=arm_qpos,
                                    gripper_action=gripper_action,
                                    pass_tolerance=self.xy_tolerance,
                                    best_reached_tolerance=max(self.xy_tolerance, 0.03),
                                )
                                final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
                                final_yaw = float(base_yaw(obs))
                                final_dist = float(np.linalg.norm(target.xy - final_xy))
                                final_yaw_error = abs(
                                    normalize_angle(target.yaw - final_yaw)
                                )
                                segment = {
                                    "primitive": "translate_rotate_translate",
                                    "pre_translate": translate,
                                    "pre_rotate": rotate,
                                    "post_translate": post_translate,
                                    "target_xy": target.xy.astype(float).tolist(),
                                    "target_yaw": float(target.yaw),
                                    "final_xy": final_xy.astype(float).tolist(),
                                    "final_yaw": final_yaw,
                                    "final_dist": final_dist,
                                    "final_yaw_error": final_yaw_error,
                                    "pass": bool(
                                        post_translate.get("pass", False)
                                        and final_yaw_error <= self.yaw_tolerance
                                    ),
                                }
                            else:
                                obs, segment = self._settle_at_sample(
                                    obs,
                                    target=target,
                                    arm_qpos=arm_qpos,
                                    gripper_action=gripper_action,
                                )
                                segment["primitive"] = "translate_rotate_settle"
                                segment["pre_translate"] = translate
                                segment["pre_rotate"] = rotate
                    else:
                        segment = translate
            else:
                obs, segment = self._rotate_to_sample(
                    obs,
                    target=target,
                    arm_qpos=arm_qpos,
                    gripper_action=gripper_action,
                )
        else:
            obs, segment = self._translate_to_sample(
                obs,
                target=target,
                arm_qpos=arm_qpos,
                gripper_action=gripper_action,
            )
        segment["source_index"] = int(source.index)
        segment["target_index"] = int(target.index)
        segment["sample_dyaw"] = float(dyaw)
        segment["sample_dxy"] = float(dxy)
        return obs, segment

    def _settle_at_sample(
        self,
        obs: dict[str, Any],
        *,
        target: ManualBasePathSample,
        arm_qpos: np.ndarray,
        gripper_action: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        yaw = float(base_yaw(obs))
        dist = float(np.linalg.norm(target.xy - xy))
        yaw_error = abs(normalize_angle(target.yaw - yaw))
        if dist <= self.xy_tolerance and yaw_error <= self.yaw_tolerance:
            return obs, {
                "primitive": "settle_current_sample",
                "source_index": int(target.index),
                "target_index": int(target.index),
                "steps": 0,
                "final_dist": dist,
                "final_yaw_error": yaw_error,
                "checks": {
                    "final_xy": True,
                    "final_yaw": True,
                },
                "pass": True,
            }

        # Primitive success remains decoupled: translation validates XY and
        # rotation validates yaw.  Settle may compose both primitives, then
        # performs the full-pose check only after all requested corrections.
        obs, pre_translate = self._translate_to_sample(
            obs,
            target=target,
            arm_qpos=arm_qpos,
            gripper_action=gripper_action,
            pass_tolerance=self.xy_tolerance,
            best_reached_tolerance=max(self.xy_tolerance, 0.03),
        )
        if bool(pre_translate.get("pass", False)):
            obs, rotate = self._rotate_to_sample(
                obs,
                target=target,
                arm_qpos=arm_qpos,
                gripper_action=gripper_action,
            )
        else:
            rotate = {
                "primitive": "rotate",
                "pass": False,
                "skipped": True,
                "reason": "pre_translate_failed",
                "steps": 0,
            }

        if bool(pre_translate.get("pass", False)) and bool(rotate.get("pass", False)):
            post_rotate_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
            post_rotate_dist = float(np.linalg.norm(target.xy - post_rotate_xy))
            if post_rotate_dist > self.xy_tolerance:
                obs, post_translate = self._translate_to_sample(
                    obs,
                    target=target,
                    arm_qpos=arm_qpos,
                    gripper_action=gripper_action,
                    pass_tolerance=self.xy_tolerance,
                    best_reached_tolerance=max(self.xy_tolerance, 0.03),
                )
            else:
                post_translate = {
                    "primitive": "translate",
                    "pass": True,
                    "skipped": True,
                    "reason": "xy_already_within_tolerance_after_rotate",
                    "steps": 0,
                    "final_dist": post_rotate_dist,
                }
        else:
            post_translate = {
                "primitive": "translate",
                "pass": False,
                "skipped": True,
                "reason": (
                    "pre_translate_failed"
                    if not bool(pre_translate.get("pass", False))
                    else "rotate_failed"
                ),
                "steps": 0,
            }

        final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        final_yaw = float(base_yaw(obs))
        final_dist = float(np.linalg.norm(target.xy - final_xy))
        final_yaw_error = abs(normalize_angle(target.yaw - final_yaw))
        checks = {
            "pre_translate": bool(pre_translate.get("pass", False)),
            "rotate": bool(rotate.get("pass", False)),
            "post_translate": bool(post_translate.get("pass", False)),
            "final_xy": final_dist <= self.xy_tolerance,
            "final_yaw": final_yaw_error <= self.yaw_tolerance,
        }
        return obs, {
            "primitive": "settle_pose",
            "source_index": int(target.index),
            "target_index": int(target.index),
            "target_xy": target.xy.astype(float).tolist(),
            "target_yaw": float(target.yaw),
            "pre_translate": pre_translate,
            "rotate": rotate,
            "post_translate": post_translate,
            "steps": int(pre_translate.get("steps", 0))
            + int(rotate.get("steps", 0))
            + int(post_translate.get("steps", 0)),
            "final_xy": final_xy.astype(float).tolist(),
            "final_yaw": final_yaw,
            "final_dist": final_dist,
            "final_yaw_error": final_yaw_error,
            "checks": checks,
            "pass": bool(all(checks.values())),
        }

    def _rotate_to_sample(
        self,
        obs: dict[str, Any],
        *,
        target: ManualBasePathSample,
        arm_qpos: np.ndarray,
        gripper_action: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        start_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        start_yaw = float(base_yaw(obs))
        steps = 0
        yaw_profile = (
            AxialVelocityProfile(self.velocity_limits.yaw_params(), self._control_dt)
            if self.velocity_limits is not None
            else None
        )
        xy_profile = (
            PlanarVelocityProfile(self.velocity_limits.xy_params(), self._control_dt)
            if self.velocity_limits is not None
            else None
        )

        # XY位置保持参数：旋转时抵抗漂移
        xy_hold_threshold = 0.08  # XY偏离 > 8cm时纠正
        xy_hold_gain = 0.4        # 纠正强度（适中）

        for _ in range(self.max_rotate_steps):
            current_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
            current_yaw = float(base_yaw(obs))
            yaw_error = normalize_angle(target.yaw - current_yaw)

            if abs(yaw_error) <= self.yaw_tolerance:
                break

            action = self._zero_base_action()

            # Yaw控制（主要）
            if action.size >= 3:
                if yaw_profile is not None:
                    action[2] = yaw_profile.step_action_signed(
                        yaw_error, tolerance=self.yaw_tolerance
                    )
                else:
                    action[2] = float(np.clip(yaw_error * self.yaw_gain, -1.0, 1.0))

            # Keep the base at the pose where rotation started. The waypoint
            # XY belongs to the following translation phase and must not leak
            # into the rotation command.
            xy_drift = current_xy - start_xy
            drift_dist = float(np.linalg.norm(xy_drift))
            if drift_dist > xy_hold_threshold and action.size >= 2:
                err_control = world_xy_to_controller_action(self.env, -xy_drift)
                if xy_profile is not None:
                    hold_action = xy_profile.step_action(err_control, tolerance=self.xy_tolerance)
                    action[:2] = hold_action[:2] * xy_hold_gain
                else:
                    action[:2] = _clip_action_norm(-xy_drift * self.xy_gain * xy_hold_gain, max_norm=0.3)[:2]

            obs = step_base_and_hold_arm(
                self.env,
                arm_qpos=arm_qpos,
                base_action=action,
                base_mode=1.0,
                gripper_action=gripper_action,
            )
            steps += 1
        final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        final_yaw = float(base_yaw(obs))
        final_dist = float(np.linalg.norm(target.xy - final_xy))
        final_yaw_error = abs(normalize_angle(target.yaw - final_yaw))
        xy_drift = float(np.linalg.norm(final_xy - start_xy))

        # Keep the historical XY diagnostic, but do not couple it to rotation
        # success. Omron's offset yaw pivot can move the reported base XY by
        # roughly 0.42 m during a near-pi turn; a following translation stage
        # owns the position correction.
        rotate_xy_tolerance = max(
            self.xy_tolerance,
            float(getattr(self, "rotate_xy_tolerance", 0.10)),
        )
        # The mobile base can translate toward the next waypoint while making
        # a large in-place turn.  Treat that motion as acceptable when it has
        # already brought the base inside the dedicated rotation-position
        # tolerance; rejecting it solely because start-relative drift is large
        # aborts an otherwise completed rotate-then-translate segment.
        xy_motion_ok = bool(
            xy_drift <= rotate_xy_tolerance
            or final_dist <= rotate_xy_tolerance
        )
        yaw_ok = final_yaw_error <= self.yaw_tolerance

        return obs, {
            "primitive": "rotate",
            "steps": int(steps),
            "target_xy": target.xy.astype(float).tolist(),
            "target_yaw": float(target.yaw),
            "start_xy": start_xy.astype(float).tolist(),
            "start_yaw": start_yaw,
            "final_xy": final_xy.astype(float).tolist(),
            "final_yaw": final_yaw,
            "final_dist": final_dist,
            "final_yaw_error": final_yaw_error,
            "xy_drift": xy_drift,
            "xy_motion_ok": xy_motion_ok,
            "checks": {
                "final_yaw": yaw_ok,
            },
            "pass": bool(yaw_ok),
        }

    def _translate_to_sample(
        self,
        obs: dict[str, Any],
        *,
        target: ManualBasePathSample,
        arm_qpos: np.ndarray,
        gripper_action: float,
        pass_tolerance: float | None = None,
        best_reached_tolerance: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        start_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        start_yaw = float(base_yaw(obs))
        start_controller_yaw = controller_translation_yaw(self.env)
        resolved_pass_tolerance = (
            max(self.xy_tolerance, 0.12)
            if pass_tolerance is None
            else float(pass_tolerance)
        )
        resolved_best_reached_tolerance = (
            max(self.xy_tolerance, 0.12)
            if best_reached_tolerance is None
            else float(best_reached_tolerance)
        )
        steps = 0
        min_dist = float("inf")
        no_progress_steps = 0
        stop_reason = "max_steps"
        xy_profile = (
            PlanarVelocityProfile(self.velocity_limits.xy_params(), self._control_dt)
            if self.velocity_limits is not None
            else None
        )
        yaw_profile = (
            AxialVelocityProfile(self.velocity_limits.yaw_params(), self._control_dt)
            if self.velocity_limits is not None
            else None
        )
        # 动态卡住阈值：速度越慢，容忍时间越短
        stuck_threshold_base = 150  # 240->150
        stuck_best_reached = 80     # 120->80

        # Yaw纠正参数：平移时保持目标朝向
        yaw_correction_threshold = 0.15  # > 8.6度时开始纠正
        yaw_correction_gain = 0.3        # 纠正强度（保守，避免过度反应）

        for _ in range(self.max_translate_steps):
            current_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
            current_yaw = float(base_yaw(obs))
            err_xy = target.xy - current_xy
            dist = float(np.linalg.norm(err_xy))

            if dist < min_dist - 1e-4:
                min_dist = dist
                no_progress_steps = 0
            else:
                no_progress_steps += 1

            if dist <= self.xy_tolerance:
                stop_reason = "tolerance"
                break
            if (
                no_progress_steps >= stuck_best_reached
                and min_dist <= resolved_best_reached_tolerance
            ):
                stop_reason = "best_reached"
                break
            if no_progress_steps >= stuck_threshold_base:
                stop_reason = "stuck_no_progress"
                break

            action = self._zero_base_action()

            # XY控制
            err_control = world_xy_to_controller_action(self.env, err_xy)
            if xy_profile is not None:
                command_xy = xy_profile.step_action(err_control, tolerance=self.xy_tolerance)
            else:
                command_xy = _clip_action_norm(err_control * self.xy_gain, max_norm=1.0)
            action[: min(2, action.size)] = command_xy[: min(2, action.size)]

            # Yaw纠正：平移时保持目标朝向，减少累积漂移
            yaw_error = normalize_angle(target.yaw - current_yaw)
            if abs(yaw_error) > yaw_correction_threshold and action.size >= 3:
                if yaw_profile is not None:
                    yaw_action = yaw_profile.step_action_signed(yaw_error, tolerance=self.yaw_tolerance)
                    # 平移时yaw纠正要保守，不能太激进
                    action[2] = yaw_action * yaw_correction_gain
                else:
                    action[2] = float(np.clip(yaw_error * self.yaw_gain * yaw_correction_gain, -0.3, 0.3))

            obs = step_base_and_hold_arm(
                self.env,
                arm_qpos=arm_qpos,
                base_action=action,
                base_mode=1.0,
                gripper_action=gripper_action,
            )
            steps += 1
        final_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        final_yaw = float(base_yaw(obs))
        final_dist = float(np.linalg.norm(target.xy - final_xy))
        final_controller_yaw = controller_translation_yaw(self.env)
        xy_ok = final_dist <= resolved_pass_tolerance
        return obs, {
            "primitive": self._translation_name(obs, target.xy),
            "local_primitive": self._local_translation_name(obs, target.xy),
            "command_mode": "velocity_profile_xy" if xy_profile is not None else "closed_loop_xy",
            "steps": int(steps),
            "stop_reason": stop_reason,
            "target_xy": target.xy.astype(float).tolist(),
            "start_xy": start_xy.astype(float).tolist(),
            "start_yaw": start_yaw,
            "start_controller_translation_yaw": start_controller_yaw,
            "final_xy": final_xy.astype(float).tolist(),
            "final_yaw": final_yaw,
            "final_controller_translation_yaw": final_controller_yaw,
            "final_dist": final_dist,
            "min_dist": float(min_dist),
            "yaw_drift": abs(normalize_angle(final_yaw - start_yaw)),
            "pass_tolerance": float(resolved_pass_tolerance),
            "best_reached_tolerance": float(resolved_best_reached_tolerance),
            "checks": {
                "final_xy": xy_ok,
            },
            "pass": bool(xy_ok),
        }

    def _translation_name(self, obs: dict[str, Any], target_xy: np.ndarray) -> str:
        current_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        err = np.array(target_xy, dtype=float)[:2] - current_xy
        axis = int(np.argmax(np.abs(err)))
        if axis == 0:
            return "world_x_plus" if err[0] >= 0.0 else "world_x_minus"
        return "world_y_plus" if err[1] >= 0.0 else "world_y_minus"

    def _local_translation_name(self, obs: dict[str, Any], target_xy: np.ndarray) -> str:
        current_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
        err = np.array(target_xy, dtype=float)[:2] - current_xy
        err_control = world_vec_to_base(obs, np.array([err[0], err[1], 0.0], dtype=float))[:2]
        axis = int(np.argmax(np.abs(err_control)))
        if axis == 0:
            return "forward" if err_control[0] >= 0.0 else "back"
        return "left" if err_control[1] >= 0.0 else "right"

    def _zero_base_action(self) -> np.ndarray:
        start, end = self.env.robots[0].composite_controller._action_split_indexes["base"]
        return np.zeros(int(end - start), dtype=float)

    def _held_object_pos(self, object_name: str | None) -> np.ndarray | None:
        if object_name is None:
            return None
        try:
            return np.array(self.env.sim.data.body_xpos[self.env.obj_body_id[object_name]], dtype=float)
        except Exception:
            return None

    def _held_motion(self, object_name: str | None, initial_pos: np.ndarray | None) -> dict[str, Any]:
        if object_name is None:
            return {"enabled": False}
        final_pos = self._held_object_pos(object_name)
        if initial_pos is None or final_pos is None:
            return {
                "enabled": True,
                "object_name": object_name,
                "reason": "object_position_unavailable",
                "drop_ok": False,
            }
        delta = final_pos - initial_pos
        return {
            "enabled": True,
            "object_name": object_name,
            "initial_pos": initial_pos.astype(float).tolist(),
            "final_pos": final_pos.astype(float).tolist(),
            "delta": delta.astype(float).tolist(),
            "drop_ok": bool(final_pos[2] >= initial_pos[2] - 0.04),
        }


__all__ = ["ManualBasePathExecutor", "ManualBasePathSample"]
