"Navigation controller — station sampling + path planning + base movement. Optional components are loaded on first use."

from importlib import import_module

_EXPORTS = {
    "base_yaw": ("bench.execution.navigation.base_motion", "base_yaw"),
    "move_base_along_waypoints": (
        "bench.execution.navigation.base_motion",
        "move_base_along_waypoints",
    ),
    "move_base_to_pose": ("bench.execution.navigation.base_motion", "move_base_to_pose"),
    "qpos_for_controller": ("bench.execution.navigation.base_motion", "qpos_for_controller"),
    "sync_arm_controller_to_current_qpos": (
        "bench.execution.navigation.base_motion",
        "sync_arm_controller_to_current_qpos",
    ),
    "BaseDriver": ("bench.execution.navigation.controller", "BaseDriver"),
    "BaseDriverAdapter": ("bench.execution.navigation.controller", "BaseDriverAdapter"),
    "BasePoseProvider": ("bench.execution.navigation.controller", "BasePoseProvider"),
    "NavigationController": ("bench.execution.navigation.controller", "NavigationController"),
    "StationSampler": ("bench.execution.navigation.controller", "StationSampler"),
    "ManualBasePathExecutor": (
        "bench.execution.navigation.manual_base_path",
        "ManualBasePathExecutor",
    ),
    "ManualBasePathSample": ("bench.execution.navigation.manual_base_path", "ManualBasePathSample"),
    "RoboCasaNavigationAdapter": (
        "bench.execution.navigation.env_adapter",
        "RoboCasaNavigationAdapter",
    ),
    "build_fixture_obstacles": (
        "bench.execution.navigation.env_adapter",
        "build_fixture_obstacles",
    ),
    "current_base_xy_from_obs": (
        "bench.execution.navigation.env_adapter",
        "current_base_xy_from_obs",
    ),
    "fixture_footprint_polygon": (
        "bench.execution.navigation.env_adapter",
        "fixture_footprint_polygon",
    ),
    "generate_station_candidates": (
        "bench.execution.navigation.env_adapter",
        "generate_station_candidates",
    ),
    "safe_name": ("bench.execution.navigation.env_adapter", "safe_name"),
    "station_navigation_inspection": (
        "bench.execution.navigation.env_adapter",
        "station_navigation_inspection",
    ),
    "station_specs_from_candidates": (
        "bench.execution.navigation.env_adapter",
        "station_specs_from_candidates",
    ),
    "StationPathPlanner": ("bench.execution.navigation.grid_planner", "StationPathPlanner"),
    "StationPoseSampler": ("bench.execution.navigation.station_poses", "StationPoseSampler"),
    "StationPoseSelector": ("bench.execution.navigation.station_poses", "StationPoseSelector"),
    "load_station_pose_payload": (
        "bench.execution.navigation.station_poses",
        "load_station_pose_payload",
    ),
    "station_id_for_navigation": (
        "bench.execution.navigation.station_poses",
        "station_id_for_navigation",
    ),
    "station_pose_include_points": (
        "bench.execution.navigation.station_poses",
        "station_pose_include_points",
    ),
}

__all__ = [
    "BaseDriver",
    "BaseDriverAdapter",
    "BasePoseProvider",
    "NavigationController",
    "RoboCasaNavigationAdapter",
    "ManualBasePathExecutor",
    "ManualBasePathSample",
    "StationSampler",
    "StationPoseSampler",
    "StationPoseSelector",
    "StationPathPlanner",
    "base_yaw",
    "build_fixture_obstacles",
    "current_base_xy_from_obs",
    "fixture_footprint_polygon",
    "generate_station_candidates",
    "load_station_pose_payload",
    "move_base_along_waypoints",
    "move_base_to_pose",
    "qpos_for_controller",
    "safe_name",
    "station_navigation_inspection",
    "station_id_for_navigation",
    "station_pose_include_points",
    "station_specs_from_candidates",
    "sync_arm_controller_to_current_qpos",
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
