"""Planning configuration — typed parameter surfaces.

Pure dataclasses with defaults.  No argparse dependency.
In the new architecture these are populated from Hydra YAML or Pydantic.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


def parse_float_values(value: Any, *, default: list[float]) -> list[float]:
    """Parse a comma/semicolon-separated float string into a list."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = list(value)
    values = [float(item) for item in raw_items if str(item).strip()]
    return values or list(default)


def _mapping_value(
    sources: tuple[Mapping[str, Any] | None, ...],
    names: str | tuple[str, ...],
    default: Any,
) -> Any:
    lookup_names = (names,) if isinstance(names, str) else names
    for source in sources:
        if source is None:
            continue
        for name in lookup_names:
            keys = (name, name.replace("_", "-"))
            for key in keys:
                if key not in source:
                    continue
                value = source[key]
                if value is not None:
                    return value
    return default


def _bool_mapping_value(
    sources: tuple[Mapping[str, Any] | None, ...],
    names: str | tuple[str, ...],
    default: bool,
) -> bool:
    value = _mapping_value(sources, names, default)
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _path_mapping_value(
    sources: tuple[Mapping[str, Any] | None, ...],
    names: str | tuple[str, ...],
    default: Path | None,
) -> Path | None:
    value = _mapping_value(sources, names, default)
    if value is None or value == "":
        return None
    return Path(value)


def _optional_float_mapping_value(
    sources: tuple[Mapping[str, Any] | None, ...],
    name: str,
    default: float | None,
) -> float | None:
    value = _mapping_value(sources, name, default)
    if value is None or value == "":
        return None
    return float(value)


def navigation_config_from_mapping(
    options: Mapping[str, Any] | None = None,
    kwargs: Mapping[str, Any] | None = None,
    *,
    base: "NavigationConfig | None" = None,
) -> "NavigationConfig":
    """Build ``NavigationConfig`` from task options and recorder kwargs.

    ``options`` is checked first so task-specific overrides can refine the
    common recorder CLI without each task re-implementing this parameter list.
    """

    cfg = base or NavigationConfig()
    sources = (options, kwargs)
    station_nav_dock_standoff = float(
        _mapping_value(
            sources,
            "station_nav_dock_standoff",
            _mapping_value(
                sources,
                "base_terminal_dock_standoff",
                cfg.station_nav_dock_standoff,
            ),
        )
    )
    base_terminal_heading_tolerance = float(
        _mapping_value(
            sources,
            "base_terminal_heading_tolerance",
            _mapping_value(sources, "base_yaw_tolerance", cfg.base_yaw_tolerance),
        )
    )
    base_terminal_dock_standoff = float(
        _mapping_value(
            sources,
            "base_terminal_dock_standoff",
            station_nav_dock_standoff,
        )
    )
    base_path_fixture_keywords = str(
        _mapping_value(
            sources,
            "base_path_fixture_keywords",
            _mapping_value(
                sources,
                "static_scene_fixture_keywords",
                cfg.base_path_fixture_keywords,
            ),
        )
    )
    base_path_exclude_keywords = str(
        _mapping_value(
            sources,
            "base_path_exclude_keywords",
            _mapping_value(
                sources,
                "static_scene_exclude_keywords",
                cfg.base_path_exclude_keywords,
            ),
        )
    )
    return replace(
        cfg,
        base_target_x=float(_mapping_value(sources, "base_target_x", cfg.base_target_x)),
        base_target_y=float(_mapping_value(sources, "base_target_y", cfg.base_target_y)),
        base_tolerance=float(_mapping_value(sources, "base_tolerance", cfg.base_tolerance)),
        base_waypoint_tolerance=float(
            _mapping_value(sources, "base_waypoint_tolerance", cfg.base_waypoint_tolerance)
        ),
        base_yaw_tolerance=float(
            _mapping_value(sources, "base_yaw_tolerance", cfg.base_yaw_tolerance)
        ),
        base_max_steps=int(_mapping_value(sources, "base_max_steps", cfg.base_max_steps)),
        base_action_frame=str(
            _mapping_value(sources, "base_action_frame", cfg.base_action_frame)
        ),
        base_translation_action_sign=float(
            _mapping_value(
                sources,
                "base_translation_action_sign",
                cfg.base_translation_action_sign,
            )
        ),
        kinematic_base_navigation=_bool_mapping_value(
            sources,
            "kinematic_base_navigation",
            cfg.kinematic_base_navigation,
        ),
        manual_base_path_enabled=_bool_mapping_value(
            sources,
            "manual_base_path_enabled",
            cfg.manual_base_path_enabled,
        ),
        manual_base_path_file=_path_mapping_value(
            sources,
            "manual_base_path_file",
            cfg.manual_base_path_file,
        ),
        manual_base_path_start_index=int(
            _mapping_value(
                sources,
                "manual_base_path_start_index",
                cfg.manual_base_path_start_index,
            )
        ),
        manual_base_path_xy_tolerance=float(
            _mapping_value(
                sources,
                "manual_base_path_xy_tolerance",
                cfg.manual_base_path_xy_tolerance,
            )
        ),
        manual_base_path_yaw_tolerance=float(
            _mapping_value(
                sources,
                "manual_base_path_yaw_tolerance",
                cfg.manual_base_path_yaw_tolerance,
            )
        ),
        manual_base_path_intermediate_xy_tolerance=_optional_float_mapping_value(
            sources,
            "manual_base_path_intermediate_xy_tolerance",
            cfg.manual_base_path_intermediate_xy_tolerance,
        ),
        manual_base_path_intermediate_yaw_tolerance=_optional_float_mapping_value(
            sources,
            "manual_base_path_intermediate_yaw_tolerance",
            cfg.manual_base_path_intermediate_yaw_tolerance,
        ),
        manual_base_path_segment_yaw_threshold=float(
            _mapping_value(
                sources,
                "manual_base_path_segment_yaw_threshold",
                cfg.manual_base_path_segment_yaw_threshold,
            )
        ),
        manual_base_path_max_rotate_steps=int(
            _mapping_value(
                sources,
                "manual_base_path_max_rotate_steps",
                cfg.manual_base_path_max_rotate_steps,
            )
        ),
        manual_base_path_max_translate_steps=int(
            _mapping_value(
                sources,
                "manual_base_path_max_translate_steps",
                cfg.manual_base_path_max_translate_steps,
            )
        ),
        manual_base_path_xy_gain=float(
            _mapping_value(
                sources,
                "manual_base_path_xy_gain",
                cfg.manual_base_path_xy_gain,
            )
        ),
        manual_base_path_yaw_gain=float(
            _mapping_value(
                sources,
                "manual_base_path_yaw_gain",
                cfg.manual_base_path_yaw_gain,
            )
        ),
        manual_base_path_nearest_yaw_weight=float(
            _mapping_value(
                sources,
                "manual_base_path_nearest_yaw_weight",
                cfg.manual_base_path_nearest_yaw_weight,
            )
        ),
        manual_base_path_rotate_xy_tolerance=float(
            _mapping_value(
                sources,
                "manual_base_path_rotate_xy_tolerance",
                cfg.manual_base_path_rotate_xy_tolerance,
            )
        ),
        base_velocity_profile_enabled=_bool_mapping_value(
            sources,
            "base_velocity_profile_enabled",
            cfg.base_velocity_profile_enabled,
        ),
        base_profile_v_max=float(
            _mapping_value(sources, "base_profile_v_max", cfg.base_profile_v_max)
        ),
        base_profile_a_max=float(
            _mapping_value(sources, "base_profile_a_max", cfg.base_profile_a_max)
        ),
        base_profile_a_dec=float(
            _mapping_value(sources, "base_profile_a_dec", cfg.base_profile_a_dec)
        ),
        base_profile_v_min=float(
            _mapping_value(sources, "base_profile_v_min", cfg.base_profile_v_min)
        ),
        base_profile_w_max=float(
            _mapping_value(sources, "base_profile_w_max", cfg.base_profile_w_max)
        ),
        base_profile_alpha_max=float(
            _mapping_value(sources, "base_profile_alpha_max", cfg.base_profile_alpha_max)
        ),
        base_profile_alpha_dec=float(
            _mapping_value(sources, "base_profile_alpha_dec", cfg.base_profile_alpha_dec)
        ),
        base_profile_w_min=float(
            _mapping_value(sources, "base_profile_w_min", cfg.base_profile_w_min)
        ),
        base_profile_dock_v_max=float(
            _mapping_value(
                sources,
                "base_profile_dock_v_max",
                _mapping_value(
                    sources,
                    "base_terminal_translation_max_action",
                    cfg.base_profile_dock_v_max,
                ),
            )
        ),
        base_profile_dock_v_min=float(
            _mapping_value(
                sources,
                "base_profile_dock_v_min",
                _mapping_value(
                    sources,
                    "base_terminal_translation_min_action",
                    cfg.base_profile_dock_v_min,
                ),
            )
        ),
        base_lookahead_distance=float(
            _mapping_value(sources, "base_lookahead_distance", cfg.base_lookahead_distance)
        ),
        base_heading_tolerance=float(
            _mapping_value(sources, "base_heading_tolerance", cfg.base_heading_tolerance)
        ),
        base_terminal_yaw_distance=float(
            _mapping_value(
                sources,
                "base_terminal_yaw_distance",
                cfg.base_terminal_yaw_distance,
            )
        ),
        base_terminal_heading_tolerance=base_terminal_heading_tolerance,
        base_terminal_translation_gain_scale=float(
            _mapping_value(
                sources,
                "base_terminal_translation_gain_scale",
                cfg.base_terminal_translation_gain_scale,
            )
        ),
        base_terminal_translation_min_action=float(
            _mapping_value(
                sources,
                "base_terminal_translation_min_action",
                cfg.base_terminal_translation_min_action,
            )
        ),
        base_terminal_translation_max_action=float(
            _mapping_value(
                sources,
                "base_terminal_translation_max_action",
                cfg.base_terminal_translation_max_action,
            )
        ),
        base_single_axis_tracking=_bool_mapping_value(
            sources,
            "base_single_axis_tracking",
            cfg.base_single_axis_tracking,
        ),
        base_terminal_dock_enabled=_bool_mapping_value(
            sources,
            "base_terminal_dock_enabled",
            cfg.base_terminal_dock_enabled,
        ),
        base_terminal_dock_standoff=base_terminal_dock_standoff,
        base_footprint_half_x=float(
            _mapping_value(sources, "base_footprint_half_x", cfg.base_footprint_half_x)
        ),
        base_footprint_half_y=float(
            _mapping_value(sources, "base_footprint_half_y", cfg.base_footprint_half_y)
        ),
        base_footprint_center_offset_x=float(
            _mapping_value(
                sources,
                "base_footprint_center_offset_x",
                cfg.base_footprint_center_offset_x,
            )
        ),
        base_footprint_center_offset_y=float(
            _mapping_value(
                sources,
                "base_footprint_center_offset_y",
                cfg.base_footprint_center_offset_y,
            )
        ),
        base_footprint_safety_margin=float(
            _mapping_value(
                sources,
                "base_footprint_safety_margin",
                cfg.base_footprint_safety_margin,
            )
        ),
        base_stuck_window=int(
            _mapping_value(sources, "base_stuck_window", cfg.base_stuck_window)
        ),
        base_stuck_progress=float(
            _mapping_value(sources, "base_stuck_progress", cfg.base_stuck_progress)
        ),
        base_abort_on_stuck=_bool_mapping_value(
            sources,
            ("base_abort_on_stuck", "abort_on_base_stuck"),
            cfg.base_abort_on_stuck,
        ),
        base_snap_to_target_after_success=_bool_mapping_value(
            sources,
            "base_snap_to_target_after_success",
            cfg.base_snap_to_target_after_success,
        ),
        base_trace_stride=int(
            _mapping_value(sources, "base_trace_stride", cfg.base_trace_stride)
        ),
        base_candidate_mode=str(
            _mapping_value(sources, "base_candidate_mode", cfg.base_candidate_mode)
        ),
        base_candidate_offsets=str(
            _mapping_value(sources, "base_candidate_offsets", cfg.base_candidate_offsets)
            or ""
        ),
        base_candidate_step=float(
            _mapping_value(sources, "base_candidate_step", cfg.base_candidate_step)
        ),
        base_candidate_range_x_values=str(
            _mapping_value(
                sources,
                "base_candidate_range_x_values",
                cfg.base_candidate_range_x_values,
            )
        ),
        base_candidate_range_y_values=str(
            _mapping_value(
                sources,
                "base_candidate_range_y_values",
                cfg.base_candidate_range_y_values,
            )
        ),
        base_candidate_yaw_mode=str(
            _mapping_value(
                sources,
                "base_candidate_yaw_mode",
                cfg.base_candidate_yaw_mode,
            )
        ),
        base_candidate_yaw_offsets_deg=str(
            _mapping_value(
                sources,
                "base_candidate_yaw_offsets_deg",
                cfg.base_candidate_yaw_offsets_deg,
            )
        ),
        base_candidate_max_count=int(
            _mapping_value(
                sources,
                "base_candidate_max_count",
                cfg.base_candidate_max_count,
            )
        ),
        base_candidate_workspace_margin=float(
            _mapping_value(
                sources,
                "base_candidate_workspace_margin",
                cfg.base_candidate_workspace_margin,
            )
        ),
        base_candidate_score_static_scene=_bool_mapping_value(
            sources,
            "base_candidate_score_static_scene",
            cfg.base_candidate_score_static_scene,
        ),
        base_retry_max_attempts=int(
            _mapping_value(sources, "base_retry_max_attempts", cfg.base_retry_max_attempts)
        ),
        base_candidate_dry_run_entry_stage=_bool_mapping_value(
            sources,
            "base_candidate_dry_run_entry_stage",
            cfg.base_candidate_dry_run_entry_stage,
        ),
        validate_base_path_collisions=_bool_mapping_value(
            sources,
            "validate_base_path_collisions",
            cfg.validate_base_path_collisions,
        ),
        base_path_fixture_keywords=base_path_fixture_keywords,
        base_path_exclude_keywords=base_path_exclude_keywords,
        abort_on_base_path_collision=_bool_mapping_value(
            sources,
            "abort_on_base_path_collision",
            cfg.abort_on_base_path_collision,
        ),
        base_holding_drop_tolerance=float(
            _mapping_value(
                sources,
                "base_holding_drop_tolerance",
                cfg.base_holding_drop_tolerance,
            )
        ),
        station_base_pose_file=_path_mapping_value(
            sources,
            "station_base_pose_file",
            cfg.station_base_pose_file,
        ),
        use_station_base_poses=_bool_mapping_value(
            sources,
            "use_station_base_poses",
            cfg.use_station_base_poses,
        ),
        station_pose_selection_policy=str(
            _mapping_value(
                sources,
                "station_pose_selection_policy",
                cfg.station_pose_selection_policy,
            )
        ),
        station_pose_selection_seed_offset=int(
            _mapping_value(
                sources,
                "station_pose_selection_seed_offset",
                cfg.station_pose_selection_seed_offset,
            )
        ),
        use_station_astar_paths=_bool_mapping_value(
            sources,
            "use_station_astar_paths",
            cfg.use_station_astar_paths,
        ),
        station_nav_grid_resolution=float(
            _mapping_value(
                sources,
                "station_nav_grid_resolution",
                cfg.station_nav_grid_resolution,
            )
        ),
        station_nav_grid_inflation=float(
            _mapping_value(
                sources,
                "station_nav_grid_inflation",
                cfg.station_nav_grid_inflation,
            )
        ),
        station_nav_grid_margin=float(
            _mapping_value(sources, "station_nav_grid_margin", cfg.station_nav_grid_margin)
        ),
        station_nav_use_docking=_bool_mapping_value(
            sources,
            "station_nav_use_docking",
            cfg.station_nav_use_docking,
        ),
        station_nav_dock_standoff=station_nav_dock_standoff,
        station_nav_require_footprint=_bool_mapping_value(
            sources,
            "station_nav_require_footprint",
            cfg.station_nav_require_footprint,
        ),
        station_nav_allow_endpoint_snap=_bool_mapping_value(
            sources,
            "station_nav_allow_endpoint_snap",
            cfg.station_nav_allow_endpoint_snap,
        ),
        station_nav_endpoint_snap_radius=float(
            _mapping_value(
                sources,
                "station_nav_endpoint_snap_radius",
                cfg.station_nav_endpoint_snap_radius,
            )
        ),
        station_nav_footprint_replan=_bool_mapping_value(
            sources,
            "station_nav_footprint_replan",
            cfg.station_nav_footprint_replan,
        ),
        station_nav_replan_max_candidates=int(
            _mapping_value(
                sources,
                "station_nav_replan_max_candidates",
                cfg.station_nav_replan_max_candidates,
            )
        ),
        station_nav_replan_segment_max_expanded=int(
            _mapping_value(
                sources,
                "station_nav_replan_segment_max_expanded",
                cfg.station_nav_replan_segment_max_expanded,
            )
        ),
        station_nav_replan_max_duration_sec=float(
            _mapping_value(
                sources,
                "station_nav_replan_max_duration_sec",
                cfg.station_nav_replan_max_duration_sec,
            )
        ),
        station_nav_replan_sparse_spacing=float(
            _mapping_value(
                sources,
                "station_nav_replan_sparse_spacing",
                cfg.station_nav_replan_sparse_spacing,
            )
        ),
        station_nav_debug=_bool_mapping_value(
            sources,
            "station_nav_debug",
            cfg.station_nav_debug,
        ),
        station_nav_heading_bins=int(
            _mapping_value(
                sources,
                "station_nav_heading_bins",
                cfg.station_nav_heading_bins,
            )
        ),
        station_nav_heading_turn_cost=float(
            _mapping_value(
                sources,
                "station_nav_heading_turn_cost",
                cfg.station_nav_heading_turn_cost,
            )
        ),
        station_nav_heading_max_expanded=int(
            _mapping_value(
                sources,
                "station_nav_heading_max_expanded",
                cfg.station_nav_heading_max_expanded,
            )
        ),
    )


# Navigation


@dataclass(frozen=True)
class NavigationConfig:
    """Parameters for base movement and station navigation."""

    # --- base motion ---
    base_target_x: float = 0.20
    base_target_y: float = 0.00
    base_tolerance: float = 0.06
    base_waypoint_tolerance: float = 0.10
    base_yaw_tolerance: float = 0.20
    base_max_steps: int = 500
    base_action_frame: str = "base"
    base_translation_action_sign: float = -1.0
    kinematic_base_navigation: bool = False

    # --- manual base path ---
    manual_base_path_enabled: bool = True
    manual_base_path_file: Path | None = None
    manual_base_path_start_index: int = 1
    manual_base_path_xy_tolerance: float = 0.09      # 0.07->0.09，放宽容差减少末端震荡
    manual_base_path_yaw_tolerance: float = 0.10     # 0.08->0.10
    # Optional pass-through tolerances for non-terminal measured-path samples.
    # None preserves the historical per-sample behavior for other tasks.
    manual_base_path_intermediate_xy_tolerance: float | None = None
    manual_base_path_intermediate_yaw_tolerance: float | None = None
    manual_base_path_segment_yaw_threshold: float = 0.12  # 0.25->0.12，更早插入旋转段
    manual_base_path_max_rotate_steps: int = 350
    manual_base_path_max_translate_steps: int = 900
    manual_base_path_xy_gain: float = 1.2            # 3.0->1.2，P控制backup增益大幅降低
    manual_base_path_yaw_gain: float = 0.8           # 2.0->0.8
    manual_base_path_nearest_yaw_weight: float = 0.20
    manual_base_path_rotate_xy_tolerance: float = 0.10

    # --- velocity profile（梯形速度剖面；False 时回退 P 控制）---
    base_velocity_profile_enabled: bool = True
    base_profile_v_max: float = 0.65     # m/s（再降 7%，配合双轴运动）
    base_profile_a_max: float = 1.6      # m/s²（再降 11%，双轴时惯性叠加）
    base_profile_a_dec: float = 1.2      # m/s²（保持）
    base_profile_v_min: float = 0.22     # m/s 命令下限（略降）
    base_profile_w_max: float = 1.2      # rad/s（保持）
    base_profile_alpha_max: float = 3.0  # rad/s²（保持）
    base_profile_alpha_dec: float = 2.5  # rad/s²（保持）
    base_profile_w_min: float = 0.20     # rad/s 命令下限（保持）
    base_profile_dock_v_max: float = 0.50 # 停靠速度再降 9%（双轴更精确）
    base_profile_dock_v_min: float = 0.28

    # --- waypoint follower ---
    base_lookahead_distance: float = 0.35            # 0.25->0.35，增加前瞻距离
    base_heading_tolerance: float = 0.35
    base_terminal_yaw_distance: float = 0.30         # 0.25->0.30，提前进入末端区
    base_terminal_heading_tolerance: float = 0.20
    base_terminal_translation_gain_scale: float = 0.5
    base_terminal_translation_min_action: float = 0.30 # 0.35->0.30，末端速度略降
    base_terminal_translation_max_action: float = 0.55 # 0.65->0.55
    base_single_axis_tracking: bool = False          # True->False，消除漂移
    base_terminal_dock_enabled: bool = True
    base_terminal_dock_standoff: float = 0.35

    # --- footprint ---
    base_footprint_half_x: float = 0.35
    base_footprint_half_y: float = 0.25
    base_footprint_center_offset_x: float = -0.20
    base_footprint_center_offset_y: float = 0.0
    base_footprint_safety_margin: float = 0.0

    # --- stuck detection ---
    base_stuck_window: int = 60                      # 50->60，增加判定窗口
    base_stuck_progress: float = 0.015               # 0.02->0.015，更敏感的卡住检测
    base_abort_on_stuck: bool = True
    base_snap_to_target_after_success: bool = False
    base_trace_stride: int = 10

    # --- candidate selection ---
    base_candidate_mode: str = "single"
    base_candidate_offsets: str = ""
    base_candidate_step: float = 0.08
    base_candidate_range_x_values: str = "0.28,0.36,0.44"
    base_candidate_range_y_values: str = "-0.16,-0.08,0.00,0.08,0.16"
    base_candidate_yaw_mode: str = "current_and_anchor_sweep"
    base_candidate_yaw_offsets_deg: str = "-35,-20,0,20,35"
    base_candidate_max_count: int = 9
    base_candidate_workspace_margin: float = 0.08
    base_candidate_score_static_scene: bool = True
    base_retry_max_attempts: int = 1
    base_candidate_dry_run_entry_stage: bool = True

    # --- path validation ---
    validate_base_path_collisions: bool = False
    base_path_fixture_keywords: str = "cabinet"
    base_path_exclude_keywords: str = "floor,wall"
    abort_on_base_path_collision: bool = True
    base_holding_drop_tolerance: float = 0.04

    # --- station navigation ---
    station_base_pose_file: Path | None = None
    use_station_base_poses: bool = False
    station_pose_selection_policy: str = "episode_random"
    station_pose_selection_seed_offset: int = 314159
    use_station_astar_paths: bool = True
    station_nav_grid_resolution: float = 0.04
    station_nav_grid_inflation: float = 0.35
    station_nav_grid_margin: float = 1.0
    station_nav_use_docking: bool = True
    station_nav_dock_standoff: float = 0.35
    station_nav_require_footprint: bool = True
    station_nav_allow_endpoint_snap: bool = True
    station_nav_endpoint_snap_radius: float = 0.60
    station_nav_footprint_replan: bool = True
    station_nav_replan_max_candidates: int = 10
    station_nav_replan_segment_max_expanded: int = 8000
    station_nav_replan_max_duration_sec: float = 8.0
    station_nav_replan_sparse_spacing: float = 0.12
    station_nav_debug: bool = False
    station_nav_heading_bins: int = 8
    station_nav_heading_turn_cost: float = 0.20
    station_nav_heading_max_expanded: int = 120000

    # --- helpers ---

    def nominal_offset_xy(self) -> tuple[float, float]:
        return (self.base_target_x, self.base_target_y)

    def retry_max_attempts(self) -> int:
        return max(1, int(self.base_retry_max_attempts))

    def candidate_max_count(self) -> int:
        return max(1, int(self.base_candidate_max_count))

    def candidate_yaw_offsets_deg(self) -> list[float]:
        return parse_float_values(
            self.base_candidate_yaw_offsets_deg,
            default=[-35.0, -20.0, 0.0, 20.0, 35.0],
        )

    def waypoint_follower_kwargs(self) -> dict[str, Any]:
        return {
            "lookahead_distance": self.base_lookahead_distance,
            "heading_tolerance": self.base_heading_tolerance,
            "terminal_yaw_distance": self.base_terminal_yaw_distance,
            "terminal_heading_tolerance": self.base_terminal_heading_tolerance,
            "terminal_translation_gain_scale": self.base_terminal_translation_gain_scale,
            "terminal_translation_min_action": self.base_terminal_translation_min_action,
            "terminal_translation_max_action": self.base_terminal_translation_max_action,
            "translation_action_sign": self.base_translation_action_sign,
            "stuck_window": self.base_stuck_window,
            "stuck_progress": self.base_stuck_progress,
            "abort_on_stuck": self.base_abort_on_stuck,
            "trace_stride": self.base_trace_stride,
            "single_axis_tracking": self.base_single_axis_tracking,
            "terminal_dock_enabled": self.base_terminal_dock_enabled,
            "terminal_dock_standoff": self.base_terminal_dock_standoff,
        }

    def footprint_kwargs(self) -> dict[str, Any]:
        return {
            "half_size": (self.base_footprint_half_x, self.base_footprint_half_y),
            "center_offset": (
                self.base_footprint_center_offset_x,
                self.base_footprint_center_offset_y,
            ),
            "safety_margin": self.base_footprint_safety_margin,
        }

    def base_move_kwargs(self) -> dict[str, Any]:
        return {
            "tolerance": self.base_tolerance,
            "yaw_tolerance": self.base_yaw_tolerance,
            "max_steps": self.base_max_steps,
            "holding_drop_tolerance": self.base_holding_drop_tolerance,
            "kinematic": self.kinematic_base_navigation,
            "action_frame": self.base_action_frame,
        }


# Pick and place


@dataclass(frozen=True)
class MplibPlannerConfig:
    planning_time: float = 5.0
    time_step: float = 0.1
    qpos_step: float = 0.02
    rrt_range: float = 0.1
    direct_pos_threshold: float = 0.03
    direct_ori_threshold_deg: float = 5.0
    offset_pos_variation_threshold: float = 0.01
    offset_ori_variation_threshold_deg: float = 3.0

    def planner_args(self) -> dict[str, float]:
        return {
            "planning_time": self.planning_time,
            "time_step": self.time_step,
            "qpos_step": self.qpos_step,
            "rrt_range": self.rrt_range,
            "direct_pos_threshold": self.direct_pos_threshold,
            "direct_ori_threshold_deg": self.direct_ori_threshold_deg,
            "offset_pos_variation_threshold": self.offset_pos_variation_threshold,
            "offset_ori_variation_threshold_deg": self.offset_ori_variation_threshold_deg,
        }


@dataclass(frozen=True)
class PickPlaceConfig:
    orientation_mode: str = "object-y-topdown"
    mug_orientation_mode: str = "object-y-topdown"
    pregrasp_height: float = 0.20
    grasp_clearance: float = -0.05
    lift_height: float = 0.12
    mug_lift_height: float | None = None
    grasp_yaw_offset_deg: float = 0.0
    lift_delta_threshold: float = 0.01
    place_height: float = 0.16
    retreat_height: float = 0.30
    retreat_x: float = -0.10
    retreat_y: float = -0.18
    release_offset_x: float = 0.0
    release_offset_y: float = 0.0
    post_release_clearance_distance: float = 0.0
    post_release_clearance_z_offset: float = 0.0
    open_steps: int = 20
    close_steps: int = 90
    release_steps: int = 45
    release_settle_steps: int = 80

    def orientation_for_object(self, *, is_mug: bool) -> str:
        return self.mug_orientation_mode if is_mug else self.orientation_mode

    def lift_height_for_object(self, *, is_mug: bool) -> float:
        if is_mug and self.mug_lift_height is not None:
            return float(self.mug_lift_height)
        return float(self.lift_height)

    def place_target_kwargs(
        self,
        *,
        place_height: float | None = None,
        retreat_height: float | None = None,
        retreat_x: float | None = None,
        retreat_y: float | None = None,
        release_offset_x: float | None = None,
        release_offset_y: float | None = None,
        post_release_clearance_distance: float | None = None,
        post_release_clearance_z_offset: float | None = None,
    ) -> dict[str, float]:
        return {
            "place_height": self.place_height if place_height is None else float(place_height),
            "retreat_height": (
                self.retreat_height if retreat_height is None else float(retreat_height)
            ),
            "retreat_x": self.retreat_x if retreat_x is None else float(retreat_x),
            "retreat_y": self.retreat_y if retreat_y is None else float(retreat_y),
            "release_offset_x": (
                self.release_offset_x
                if release_offset_x is None
                else float(release_offset_x)
            ),
            "release_offset_y": (
                self.release_offset_y
                if release_offset_y is None
                else float(release_offset_y)
            ),
            "post_release_clearance_distance": (
                self.post_release_clearance_distance
                if post_release_clearance_distance is None
                else float(post_release_clearance_distance)
            ),
            "post_release_clearance_z_offset": (
                self.post_release_clearance_z_offset
                if post_release_clearance_z_offset is None
                else float(post_release_clearance_z_offset)
            ),
        }


@dataclass(frozen=True)
class ManualGraspConfig:
    """Generic manual grasp candidate configuration.

    The config is object-agnostic. Task layers decide which objects should use
    these files or directories.
    """

    manual_grasp_pose_file: Path | None = None
    manual_grasp_pose_dir: Path | None = None
    pregrasp_distance: float | None = None
    selection_mode: str = "nearest_reachable"
    validation_mode: str = "pregrasp"
    selection_seed_offset: int = 271828

    def selection_seed(self, *, episode_seed: int, object_name: str, attempt_index: int) -> int:
        return (
            int(episode_seed)
            + int(self.selection_seed_offset)
            + int(attempt_index)
            + int(zlib.crc32(str(object_name).encode("utf-8")))
        ) & 0xFFFFFFFF
