"Planning layer — configurations and algorithms. Optional components are loaded on first use."

from importlib import import_module

_EXPORTS = {
    "ManualGraspConfig": ("bench.execution.planning.config", "ManualGraspConfig"),
    "MplibPlannerConfig": ("bench.execution.planning.config", "MplibPlannerConfig"),
    "NavigationConfig": ("bench.execution.planning.config", "NavigationConfig"),
    "PickPlaceConfig": ("bench.execution.planning.config", "PickPlaceConfig"),
    "navigation_config_from_mapping": (
        "bench.execution.planning.config",
        "navigation_config_from_mapping",
    ),
    "check_base_path_collision": (
        "bench.execution.planning.collision_monitor",
        "check_base_path_collision",
    ),
    "check_monitor_collision": (
        "bench.execution.planning.collision_monitor",
        "check_monitor_collision",
    ),
    "collision_monitor_summary": (
        "bench.execution.planning.collision_monitor",
        "collision_monitor_summary",
    ),
    "make_base_path_monitor": (
        "bench.execution.planning.collision_monitor",
        "make_base_path_monitor",
    ),
    "make_collision_monitor": (
        "bench.execution.planning.collision_monitor",
        "make_collision_monitor",
    ),
    "preflight_arm_trajectory_collisions": (
        "bench.execution.planning.collision_monitor",
        "preflight_arm_trajectory_collisions",
    ),
    "build_joint_mapping": ("bench.execution.planning.model_alignment", "build_joint_mapping"),
    "mat_to_quat_xyzw": ("bench.execution.planning.model_alignment", "mat_to_quat_xyzw"),
    "matrix_from_base_obs": ("bench.execution.planning.model_alignment", "matrix_from_base_obs"),
    "matrix_from_json_pose": ("bench.execution.planning.model_alignment", "matrix_from_json_pose"),
    "matrix_from_pos_quat_wxyz": (
        "bench.execution.planning.model_alignment",
        "matrix_from_pos_quat_wxyz",
    ),
    "matrix_from_pos_quat_xyzw": (
        "bench.execution.planning.model_alignment",
        "matrix_from_pos_quat_xyzw",
    ),
    "matrix_from_site_pose": ("bench.execution.planning.model_alignment", "matrix_from_site_pose"),
    "normalize_joint_name": ("bench.execution.planning.model_alignment", "normalize_joint_name"),
    "pose7_from_matrix_wxyz": (
        "bench.execution.planning.model_alignment",
        "pose7_from_matrix_wxyz",
    ),
    "pose_error": ("bench.execution.planning.model_alignment", "pose_error"),
    "pose_from_matrix": ("bench.execution.planning.model_alignment", "pose_from_matrix"),
    "quat_xyzw_to_mat": ("bench.execution.planning.model_alignment", "quat_xyzw_to_mat"),
    "rotation_error_deg": ("bench.execution.planning.model_alignment", "rotation_error_deg"),
    "trajectory_for_robocasa": (
        "bench.execution.planning.model_alignment",
        "trajectory_for_robocasa",
    ),
    "OccupancyGrid": ("bench.execution.planning.nav_algo", "OccupancyGrid"),
    "astar_heading_path": ("bench.execution.planning.nav_algo", "astar_heading_path"),
    "astar_path": ("bench.execution.planning.nav_algo", "astar_path"),
    "base_footprint_collision_summary": (
        "bench.execution.planning.nav_algo",
        "base_footprint_collision_summary",
    ),
    "base_footprint_polygon": ("bench.execution.planning.nav_algo", "base_footprint_polygon"),
    "base_motion_sweep_collision_summary": (
        "bench.execution.planning.nav_algo",
        "base_motion_sweep_collision_summary",
    ),
    "base_path_sweep_collision_summary": (
        "bench.execution.planning.nav_algo",
        "base_path_sweep_collision_summary",
    ),
    "build_occupancy_grid": ("bench.execution.planning.nav_algo", "build_occupancy_grid"),
    "build_occupancy_grid_from_obstacles": (
        "bench.execution.planning.nav_algo",
        "build_occupancy_grid_from_obstacles",
    ),
    "convex_hull": ("bench.execution.planning.nav_algo", "convex_hull"),
    "nearest_obstacle_distance": ("bench.execution.planning.nav_algo", "nearest_obstacle_distance"),
    "normalize_angle": ("bench.execution.planning.nav_algo", "normalize_angle"),
    "point_in_polygon": ("bench.execution.planning.nav_algo", "point_in_polygon"),
    "point_polygon_distance": ("bench.execution.planning.nav_algo", "point_polygon_distance"),
    "polygon_area": ("bench.execution.planning.nav_algo", "polygon_area"),
    "polygon_polygon_distance": ("bench.execution.planning.nav_algo", "polygon_polygon_distance"),
    "polygons_intersect": ("bench.execution.planning.nav_algo", "polygons_intersect"),
    "project_point_to_segment": ("bench.execution.planning.nav_algo", "project_point_to_segment"),
    "segments_intersect": ("bench.execution.planning.nav_algo", "segments_intersect"),
    "select_station_candidates": ("bench.execution.planning.nav_algo", "select_station_candidates"),
    "simplify_waypoints": ("bench.execution.planning.nav_algo", "simplify_waypoints"),
    "validate_station_candidates": (
        "bench.execution.planning.nav_algo",
        "validate_station_candidates",
    ),
    "eef_pose_for_object_reference_target": (
        "bench.execution.planning.pick_place_targets",
        "eef_pose_for_object_reference_target",
    ),
    "green_closing_pose_to_panda_tcp": (
        "bench.execution.planning.pick_place_targets",
        "green_closing_pose_to_panda_tcp",
    ),
    "green_closing_rot_to_panda_tcp": (
        "bench.execution.planning.pick_place_targets",
        "green_closing_rot_to_panda_tcp",
    ),
    "make_manual_object_pick_target_poses": (
        "bench.execution.planning.pick_place_targets",
        "make_manual_object_pick_target_poses",
    ),
    "make_object_pick_target_poses": (
        "bench.execution.planning.pick_place_targets",
        "make_object_pick_target_poses",
    ),
    "make_place_entry_exit_ring_pose_candidates": (
        "bench.execution.planning.pick_place_targets",
        "make_place_entry_exit_ring_pose_candidates",
    ),
    "make_receptacle_pick_place_target_poses": (
        "bench.execution.planning.pick_place_targets",
        "make_receptacle_pick_place_target_poses",
    ),
    "make_receptacle_place_target_poses": (
        "bench.execution.planning.pick_place_targets",
        "make_receptacle_place_target_poses",
    ),
    "make_world_place_target_pose_candidates": (
        "bench.execution.planning.pick_place_targets",
        "make_world_place_target_pose_candidates",
    ),
    "make_world_place_target_poses": (
        "bench.execution.planning.pick_place_targets",
        "make_world_place_target_poses",
    ),
    "matrix_with_pose": ("bench.execution.planning.pick_place_targets", "matrix_with_pose"),
    "normalize_placement_reference": (
        "bench.execution.planning.pick_place_targets",
        "normalize_placement_reference",
    ),
    "object_reference_pos": ("bench.execution.planning.pick_place_targets", "object_reference_pos"),
    "build_static_scene_point_cloud": (
        "bench.execution.planning.static_scene_point_cloud",
        "build_static_scene_point_cloud",
    ),
    "parse_keyword_csv": ("bench.execution.planning.static_scene_point_cloud", "parse_keyword_csv"),
    "parse_vector3": ("bench.execution.planning.static_scene_point_cloud", "parse_vector3"),
    "sample_aabb_surface": (
        "bench.execution.planning.static_scene_point_cloud",
        "sample_aabb_surface",
    ),
}

__all__ = [
    "NavigationConfig",
    "navigation_config_from_mapping",
    "OccupancyGrid",
    "PickPlaceConfig",
    "ManualGraspConfig",
    "MplibPlannerConfig",
    "astar_heading_path",
    "astar_path",
    "base_footprint_collision_summary",
    "base_footprint_polygon",
    "base_motion_sweep_collision_summary",
    "base_path_sweep_collision_summary",
    "build_joint_mapping",
    "build_occupancy_grid",
    "build_occupancy_grid_from_obstacles",
    "build_static_scene_point_cloud",
    "check_base_path_collision",
    "check_monitor_collision",
    "collision_monitor_summary",
    "convex_hull",
    "eef_pose_for_object_reference_target",
    "green_closing_pose_to_panda_tcp",
    "green_closing_rot_to_panda_tcp",
    "make_manual_object_pick_target_poses",
    "make_base_path_monitor",
    "make_collision_monitor",
    "make_object_pick_target_poses",
    "make_place_entry_exit_ring_pose_candidates",
    "make_receptacle_pick_place_target_poses",
    "make_receptacle_place_target_poses",
    "make_world_place_target_pose_candidates",
    "make_world_place_target_poses",
    "mat_to_quat_xyzw",
    "matrix_from_base_obs",
    "matrix_from_json_pose",
    "matrix_from_pos_quat_wxyz",
    "matrix_from_pos_quat_xyzw",
    "matrix_from_site_pose",
    "matrix_with_pose",
    "nearest_obstacle_distance",
    "normalize_placement_reference",
    "normalize_joint_name",
    "normalize_angle",
    "object_reference_pos",
    "parse_keyword_csv",
    "parse_vector3",
    "point_in_polygon",
    "point_polygon_distance",
    "pose7_from_matrix_wxyz",
    "pose_error",
    "pose_from_matrix",
    "polygon_area",
    "polygon_polygon_distance",
    "polygons_intersect",
    "preflight_arm_trajectory_collisions",
    "project_point_to_segment",
    "quat_xyzw_to_mat",
    "rotation_error_deg",
    "sample_aabb_surface",
    "segments_intersect",
    "select_station_candidates",
    "simplify_waypoints",
    "trajectory_for_robocasa",
    "validate_station_candidates",
]


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
