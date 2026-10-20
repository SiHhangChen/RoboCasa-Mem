"""Reset-time simulator mutations for WX-01."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic
from bench.runtime.contracts import InitializationReport
from bench.execution.navigation.station_poses import (
    StationPoseSelector,
    load_station_pose_payload,
    station_sample_records,
)
from bench.execution.planning import NavigationConfig
from bench.tasks.wx01.domain.contracts import (
    EpisodePlan,
    EpisodeTargets,
    InitialStateConfig,
    RobotBaseConfig,
    RobotBaseMode,
)


def apply_robot_base_initial_state(
    env: Any,
    obs: dict[str, Any],
    *,
    config: RobotBaseConfig,
    navigation: NavigationConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the one resolved Base initialization mode."""

    if config.mode is RobotBaseMode.PRESERVE_RESET_POSE:
        return obs, _base_summary(obs, enabled=False, mode=config.mode.value)
    if config.mode is RobotBaseMode.EXPLICIT_POSE:
        if config.xy is None or config.yaw is None:
            raise ValueError("explicit_pose requires resolved xy and yaw")
        return _set_base_pose(
            env,
            obs,
            target_xy=np.asarray(config.xy, dtype=float),
            target_yaw=config.yaw,
            mode=config.mode.value,
        )
    if config.mode is RobotBaseMode.MANUAL_PATH_SAMPLE:
        if navigation.manual_base_path_file is None or config.sample_index is None:
            raise ValueError("manual_path_sample requires a path and sample index")
        path = Path(navigation.manual_base_path_file)
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = {
            int(item.get("index", fallback)): dict(item)
            for fallback, item in enumerate(payload.get("samples", []))
        }
        sample = samples.get(config.sample_index)
        if sample is None:
            raise KeyError(
                f"manual base sample {config.sample_index} not found in {path}"
            )
        xy = np.asarray(sample.get("xy", []), dtype=float).reshape(-1)
        yaw = float(sample.get("yaw"))
        if xy.size != 2 or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"invalid manual base sample {config.sample_index}")
        updated_obs, summary = _set_base_pose(
            env,
            obs,
            target_xy=xy,
            target_yaw=yaw,
            mode=config.mode.value,
        )
        summary.update(
            {
                "manual_base_path_file": str(path),
                "sample_index": config.sample_index,
                "sample_name": sample.get("name"),
                "sample": sample,
            }
        )
        return updated_obs, summary
    if config.mode is RobotBaseMode.STATION_SAMPLE:
        if navigation.station_base_pose_file is None or config.station_id is None:
            raise ValueError("station_sample requires station poses and station id")
        payload = load_station_pose_payload(navigation.station_base_pose_file)
        station = (payload.get("stations", {}) or {}).get(config.station_id)
        if station is None:
            raise KeyError(f"initial station {config.station_id!r} not found")
        records = station_sample_records(list(station.get("samples", [])))
        if not records:
            raise ValueError(f"initial station {config.station_id!r} has no samples")
        reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
        selector = StationPoseSelector.from_config(navigation, seed=int(seed))
        record, selection = selector.select(
            config.station_id, records, base_xy=reset_xy
        )
        updated_obs, summary = _set_base_pose(
            env,
            obs,
            target_xy=np.asarray(record["target_xy"], dtype=float),
            target_yaw=float(record["target_yaw"]),
            mode=config.mode.value,
        )
        summary.update(
            {
                "station_id": config.station_id,
                "station_base_pose_file": str(navigation.station_base_pose_file),
                "selection": selection,
                "sample_index": int(record["sample_index"]),
                "sample": dict(record["sample"]),
            }
        )
        return updated_obs, summary
    raise ValueError(f"unsupported WX-01 Base mode: {config.mode!r}")


def apply_initial_object_plan(
    env: Any,
    *,
    episode: EpisodePlan,
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply episode-seeded original object positions before displacement."""

    summary: dict[str, Any] = {
        "enabled": bool(enabled),
        "seed": episode.seed,
        "coordinate_frame": "world",
        "objects": {},
    }
    if not enabled:
        return env._get_observations(force_update=True), summary
    resolved_positions: dict[str, np.ndarray] = {}
    for object_name, plan in episode.initial_object_poses.items():
        before = object_joint_position(env, object_name)
        if before is None:
            raise ValueError(f"cannot read reset pose for WX-01 object {object_name!r}")
        target = np.asarray([plan.xy[0], plan.xy[1], before[2]], dtype=float)
        applied = set_object_free_joint_pose(
            env,
            object_name,
            position=target,
            quaternion_wxyz=_yaw_quaternion(plan.yaw_deg),
        )
        if not applied:
            raise RuntimeError(f"failed to place WX-01 object {object_name!r}")
        resolved_positions[object_name] = target
        summary["objects"][object_name] = {
            "applied": True,
            "before_pos": before.tolist(),
            "after_pos": target.tolist(),
            "region_center_xy": list(plan.region_center_xy),
            "region_size_xy": list(plan.region_size_xy),
            "yaw_deg": plan.yaw_deg,
            "seed": plan.seed,
        }
    env.sim.forward()
    env.set_original_object_poses(
        cleaning_tool=resolved_positions["cleaning_tool"],
        blocking_object=resolved_positions["blocking_object"],
    )
    return env._get_observations(force_update=True), summary


def initialize_short_start_state(
    env: Any,
    setup_controller: Any,
    *,
    targets: EpisodeTargets,
    config: InitialStateConfig,
    grasp_pose_file: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Physically prepare the six-stage start before the recorder is installed."""

    from bench.execution.manipulation import (
        evaluate_object_transport,
        get_object_hold_status,
        get_object_transport_snapshot,
        lift_held_object_for_transport,
    )

    start = config.short_start
    execute = bool(setup_controller.runtime_config.execute)
    summary: dict[str, Any] = {
        "mode": start.mode,
        "setup_operation": start.setup_operation,
        "physical_setup_executed": execute,
        "recorded_in_hdf5_actions": False,
    }

    before = object_joint_position(env, "blocking_object")
    quaternion = object_joint_quaternion_wxyz(env, "blocking_object")
    if before is None or quaternion is None:
        raise ValueError("cannot read blocking_object free-joint pose")
    target = np.asarray(targets.temporary_blocking_pose, dtype=float)
    applied = set_object_free_joint_pose(
        env,
        "blocking_object",
        position=target,
        quaternion_wxyz=quaternion,
    )
    env.set_original_object_poses(
        cleaning_tool=targets.original_tool_pose,
        blocking_object=targets.original_blocking_pose,
    )
    env.sim.forward()
    summary["blocking_object_initialization"] = {
        "enabled": True,
        "mode": "preplaced_at_temporary_pose",
        "applied": bool(applied),
        "before_pos": before.tolist(),
        "temporary_pose": target.tolist(),
        "remembered_target_pose": list(targets.original_blocking_pose),
    }
    if not applied:
        raise RuntimeError("failed to preplace blocking_object at temporary pose")

    if not execute:
        obs = env._get_observations(force_update=True)
        setup_controller._sync_obs(obs)
        summary["cleaning_tool_initialization"] = {
            "enabled": False,
            "mode": "dry_run_unheld_tool",
            "grasp_pose_file": str(Path(grasp_pose_file)),
            "pass": True,
            "reason": "execute_false",
        }
        return obs, summary

    setup_controller._sync_obs(env._get_observations(force_update=True))
    pick_result = setup_controller.pick(
        "cleaning_tool",
        manual_grasp_pose_file=Path(grasp_pose_file),
        lift_height=start.cleaning_tool_lift_height,
    )
    hold_after_pick = get_object_hold_status(
        env,
        "cleaning_tool",
        max_end_effector_distance=start.max_end_effector_distance,
    )
    tool_summary: dict[str, Any] = {
        "enabled": True,
        "mode": "physical_pick_before_recording",
        "grasp_pose_file": str(Path(grasp_pose_file)),
        **compact_setup_operation_summary(pick_result),
        "hold_after_pick": hold_after_pick,
    }
    summary["cleaning_tool_initialization"] = tool_summary
    if not bool(pick_result.get("pass", False)) or not bool(
        hold_after_pick.get("held", False)
    ):
        raise RuntimeError(
            "WX-01 physical cleaning-tool setup pick failed: "
            + json.dumps(tool_summary, ensure_ascii=False, sort_keys=True)
        )

    transport_before = get_object_transport_snapshot(env, "cleaning_tool")
    probe_result = lift_held_object_for_transport(
        setup_controller,
        "cleaning_tool",
        stage="wx01_setup_transport_probe",
        lift_height=start.transport_probe_height,
        settle_steps=max(1, int(setup_controller.runtime_config.settle_steps)),
    )
    transport_after = get_object_transport_snapshot(env, "cleaning_tool")
    transport = evaluate_object_transport(
        transport_before,
        transport_after,
        minimum_end_effector_motion=0.5 * start.transport_probe_height,
        minimum_motion_ratio=start.minimum_motion_ratio,
        max_relative_offset_drift=start.max_relative_offset_drift,
    )
    hold_after_probe = get_object_hold_status(
        env,
        "cleaning_tool",
        max_end_effector_distance=start.max_end_effector_distance,
    )
    probe_summary = {
        "operation": compact_setup_operation_summary(probe_result),
        "transport": transport,
        "hold_after_probe": hold_after_probe,
        "pass": bool(
            probe_result.get("pass", False)
            and transport.get("pass", False)
            and hold_after_probe.get("held", False)
        ),
    }
    tool_summary["transport_probe"] = probe_summary
    tool_summary["pass"] = bool(probe_summary["pass"])
    if not bool(probe_summary["pass"]):
        raise RuntimeError(
            "WX-01 cleaning tool did not follow the end effector during setup: "
            + json.dumps(probe_summary, ensure_ascii=False, sort_keys=True)
        )

    obs = env._get_observations(force_update=True)
    setup_controller._sync_obs(obs)
    return obs, summary


def object_joint_position(env: Any, object_name: str) -> np.ndarray | None:
    """Read a free-joint XYZ while tolerating minimal fake environments."""

    try:
        obj = env.objects[object_name]
        qpos = np.asarray(
            env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
        ).reshape(-1)
        if qpos.size >= 3 and np.all(np.isfinite(qpos[:3])):
            return qpos[:3]
    except Exception:
        pass
    try:
        return np.asarray(
            env.sim.data.body_xpos[env.obj_body_id[object_name]], dtype=float
        )[:3]
    except Exception:
        return None


def object_joint_quaternion_wxyz(env: Any, object_name: str) -> np.ndarray | None:
    """Read a free-joint quaternion in MuJoCo WXYZ ordering."""

    try:
        obj = env.objects[object_name]
        qpos = np.asarray(
            env.sim.data.get_joint_qpos(obj.joints[0]), dtype=float
        ).reshape(-1)
        if qpos.size >= 7 and np.all(np.isfinite(qpos[3:7])):
            return qpos[3:7]
    except Exception:
        return None
    return None


def set_object_free_joint_pose(
    env: Any,
    object_name: str,
    *,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> bool:
    """Set one task object's free joint and synchronize RoboCasa placement metadata."""

    try:
        obj = env.objects[object_name]
        pos = np.asarray(position, dtype=float)[:3]
        quat = np.asarray(quaternion_wxyz, dtype=float)[:4]
        env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat]))
        if object_name in getattr(env, "object_placements", {}):
            env.object_placements[object_name] = (
                tuple(float(value) for value in pos),
                tuple(float(value) for value in quat),
                obj,
            )
        return True
    except Exception:
        return False


def compact_setup_operation_summary(result: Any) -> dict[str, Any]:
    """Keep setup diagnostics useful without duplicating full planner payloads."""

    compact: dict[str, Any] = {"pass": bool(result.get("pass", False))}
    for key in (
        "primitive",
        "object_name",
        "grasp_strategy",
        "checks",
        "grasp_validation",
        "grasp_validation_after_close",
        "weak_grasp_after_close",
        "reason",
    ):
        if key in result:
            compact[key] = result[key]
    stages = []
    for stage in list(result.get("stages") or []):
        execution = stage.get("execution") or {}
        stages.append(
            {
                "stage": stage.get("stage"),
                "pass": bool(stage.get("pass", False)),
                "failure_reason": stage.get("failure_reason"),
                "final_position_error_m": execution.get("final_position_error_m"),
                "final_qpos_error": execution.get("final_qpos_error"),
            }
        )
    if stages:
        compact["stages"] = stages
    return compact


def _set_base_pose(
    env: Any,
    obs: dict[str, Any],
    *,
    target_xy: np.ndarray,
    target_yaw: float,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_xy = np.asarray(obs["robot0_base_pos"][:2], dtype=float)
    updated = set_base_pose_kinematic(
        env,
        target_xy=np.asarray(target_xy, dtype=float),
        target_yaw=float(target_yaw),
    )
    summary = _base_summary(updated, enabled=True, mode=mode)
    summary.update(
        {
            "reset_base_xy": reset_xy.tolist(),
            "target_xy": np.asarray(target_xy, dtype=float)[:2].tolist(),
            "target_yaw": float(target_yaw),
        }
    )
    return updated, summary


def _base_summary(obs: dict[str, Any], *, enabled: bool, mode: str) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "mode": str(mode),
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(
            obs["robot0_base_quat"], dtype=float
        ).tolist(),
    }


def _yaw_quaternion(yaw_deg: float) -> np.ndarray:
    half = np.deg2rad(float(yaw_deg)) * 0.5
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


__all__ = [
    "apply_initial_object_plan",
    "apply_robot_base_initial_state",
    "compact_setup_operation_summary",
    "initialize_short_start_state",
    "object_joint_position",
    "object_joint_quaternion_wxyz",
    "set_object_free_joint_pose",
]


def initialize_episode(runtime: Any, episode: Any) -> InitializationReport:
    """Expose reset facts already applied by the task runtime."""
    del episode
    return InitializationReport.applied(facts=dict(getattr(runtime, "initialization", {}) or {}))
