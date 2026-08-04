"""Interactively calibrate one anchor-relative 6D world pose."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _early_configure_mujoco_gl() -> None:
    if "--no-render" in sys.argv:
        return
    requested = None
    for idx, token in enumerate(sys.argv):
        if token == "--mujoco-gl" and idx + 1 < len(sys.argv):
            requested = sys.argv[idx + 1]
        elif token.startswith("--mujoco-gl="):
            requested = token.split("=", 1)[1]
    if requested:
        os.environ["MUJOCO_GL"] = requested
        return
    current = os.environ.get("MUJOCO_GL", "").strip().lower()
    if current in {"", "egl", "osmesa", "glfw"}:
        os.environ["MUJOCO_GL"] = "glx"


_early_configure_mujoco_gl()

with contextlib.redirect_stdout(sys.stderr):
    import mujoco  # noqa: E402

    from bench.env.robocasa_runtime import (  # noqa: E402
        make_joint_position_controller_config,
        make_robocasa_env,
        reset_env_with_initial_arm_qpos,
    )
    from bench.execution.navigation.base_motion import set_base_pose_kinematic  # noqa: E402
    from bench.execution.planning import (  # noqa: E402
        mat_to_quat_xyzw,
        matrix_from_json_pose,
        pose_from_matrix,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a single 6D pose near an anchor. The output is pose JSON only; "
            "task-specific manipulation code consumes it later."
        )
    )
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target", "all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-id", type=int, default=None)
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help=(
            "Optional task configuration used to reproduce task-owned environment "
            "initialization. WR02ReturnCondimentPair uses its default config when omitted."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-format",
        choices=["anchor_pose", "manual_grasp_pose"],
        default="anchor_pose",
        help="Output either a generic anchor pose JSON or an object-local manual grasp pose JSON.",
    )
    parser.add_argument("--pose-name", default="anchor_pose")
    parser.add_argument(
        "--grasp-object",
        default=None,
        help="Object name used when --output-format manual_grasp_pose is selected.",
    )
    parser.add_argument(
        "--grasp-asset",
        default=None,
        help="Optional asset label/path written into manual grasp pose payloads.",
    )
    parser.add_argument(
        "--pregrasp-offset-tcp",
        default="0,0,-0.075",
        help="Local TCP pregrasp offset written into manual grasp pose payloads.",
    )
    parser.add_argument(
        "--initial-pose-file",
        type=Path,
        default=None,
        help=(
            "Optional saved anchor pose JSON. When set, initialize the marker pose "
            "from pose.world (matrix or p + q_xyzw)."
        ),
    )
    parser.add_argument(
        "--reanchor-initial-pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "With --initial-pose-file, preserve its rotation and saved "
            "offset_from_anchor_world while moving it onto the live --anchor."
        ),
    )
    parser.add_argument(
        "--move-gripper-to-initial-pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When set with --initial-pose-file, plan and move the robot gripper to that pose first.",
    )
    parser.add_argument(
        "--anchor",
        default="world",
        help=(
            "Anchor selector: world, sink_water_site, site:<name>, body:<name>, "
            "geom:<name>, object:<name>, fixture:<env_attr>, or env_attr:<name>."
        ),
    )
    parser.add_argument(
        "--anchor-world",
        default=None,
        help="x,y,z anchor used by --anchor world, or fallback for other anchors.",
    )
    parser.add_argument("--initial-offset", default="0,0,0")
    parser.add_argument("--initial-roll-deg", type=float, default=0.0)
    parser.add_argument("--initial-pitch-deg", type=float, default=0.0)
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    parser.add_argument("--initial-base-x", type=float, default=None)
    parser.add_argument("--initial-base-y", type=float, default=None)
    parser.add_argument("--initial-base-yaw", type=float, default=None)
    parser.add_argument(
        "--initial-torso-height",
        type=float,
        default=None,
        help="Optional PandaOmron torso joint height applied before pose calibration.",
    )
    parser.add_argument(
        "--torso-tolerance",
        type=float,
        default=0.002,
        help="Allowed torso-height error in meters.",
    )
    parser.add_argument(
        "--torso-max-steps",
        type=int,
        default=240,
        help="Maximum environment steps used to reach --initial-torso-height.",
    )
    parser.add_argument(
        "--initial-object-name",
        default=None,
        help="Optional free object to stage before interactive calibration.",
    )
    parser.add_argument(
        "--initial-object-body-world",
        default=None,
        help="World x,y,z body-center pose for --initial-object-name.",
    )
    parser.add_argument(
        "--initial-object-quat-xyzw",
        default=None,
        help="Optional world quaternion x,y,z,w for the staged object.",
    )
    parser.add_argument(
        "--pick-setup-file",
        type=Path,
        default=None,
        help=(
            "Optional pick-setup JSON. Restores the saved object world pose and "
            "mobile-base pose before initializing the calibration target."
        ),
    )
    parser.add_argument("--preview-storage-fixture", default=None)
    parser.add_argument("--tabletop-storage-world-xyz", default=None)
    parser.add_argument("--tabletop-storage-world-yaw-deg", type=float, default=None)
    parser.add_argument("--tool-name", default=None)
    parser.add_argument("--tool-object-group", default=None)
    parser.add_argument("--tool-object-scale", type=float, default=None)
    parser.add_argument(
        "--slide-door-open-fraction",
        type=float,
        default=None,
        help="Directly set a supported slide door for pose calibration only.",
    )
    parser.add_argument(
        "--slide-door-levels",
        default=None,
        help=(
            "Optional comma-separated level ids for environments with multiple "
            "slide doors, for example lower,upper."
        ),
    )
    parser.add_argument(
        "--stage-tools-on-storage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move supported tool objects to deterministic cabinet-top staging poses.",
    )
    parser.add_argument(
        "--initial-orientation-grasp-file",
        type=Path,
        default=None,
        help=(
            "Initialize marker orientation from an object-local manual grasp pose. "
            "Composite pose files are supported with --initial-orientation-candidate."
        ),
    )
    parser.add_argument("--initial-orientation-object", default=None)
    parser.add_argument(
        "--initial-orientation-candidate",
        type=int,
        default=1,
        help="One-based candidate index for a composite manual grasp pose file.",
    )
    parser.add_argument("--step-pos", type=float, default=0.005)
    parser.add_argument("--step-rot-deg", type=float, default=2.0)
    parser.add_argument("--axis-length", type=float, default=0.10)
    parser.add_argument("--axis-radius", type=float, default=0.005)
    parser.add_argument("--planner-conda-env", default="membench")
    parser.add_argument("--planner-script", type=Path, default=ROOT / "bench/planning/mplib_planner.py")
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--qpos-step", type=float, default=0.02)
    parser.add_argument("--rrt-range", type=float, default=0.1)
    parser.add_argument("--max-joint-step", type=float, default=0.015)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--position-threshold", type=float, default=0.04)
    parser.add_argument("--orientation-threshold-deg", type=float, default=8.0)
    parser.add_argument("--qpos-threshold", type=float, default=0.12)
    parser.add_argument("--initial-gripper-open-steps", type=int, default=25)
    parser.add_argument(
        "--interactive-gripper-steps",
        type=int,
        default=30,
        help="Steps used by the interactive open / close commands.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--render-camera", default="none")
    parser.add_argument("--viewer-distance", type=float, default=1.4)
    parser.add_argument("--viewer-elevation", type=float, default=-25.0)
    parser.add_argument("--viewer-azimuth", type=float, default=145.0)
    parser.add_argument(
        "--mujoco-gl",
        choices=["glfw", "glx", "egl", "osmesa"],
        default=os.environ.get("MUJOCO_GL", "glx"),
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--save-immediately", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def vector3(value: Any) -> np.ndarray:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"expected x,y,z vector, got {value!r}")
        return np.array([float(part) for part in parts], dtype=float)
    arr = np.array(value, dtype=float).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"expected at least three values, got {value!r}")
    return arr[:3].astype(float)


def optional_vector3(value: Any) -> tuple[float, float, float] | None:
    if value is None or str(value).strip() == "":
        return None
    arr = vector3(value)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rotation_y(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def euler_xyz_deg_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return (
        _rotation_z(np.deg2rad(float(yaw)))
        @ _rotation_y(np.deg2rad(float(pitch)))
        @ _rotation_x(np.deg2rad(float(roll)))
    )


def pose_matrix(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.array(rot, dtype=float).reshape(3, 3)
    mat[:3, 3] = np.array(pos, dtype=float).reshape(3)
    return mat


def load_pose_world_matrix(path: Path, env: Any | None = None) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    pose = payload.get("pose", {}).get("world")
    if isinstance(pose, dict) and (
        pose.get("matrix") is not None
        or (pose.get("p") is not None and pose.get("q_xyzw") is not None)
    ):
        return np.array(matrix_from_json_pose(pose), dtype=float).reshape(4, 4)

    selected_world = (payload.get("selected") or {}).get("eef_site_world")
    matrix = selected_world.get("matrix") if isinstance(selected_world, dict) else None
    if matrix is not None:
        return np.array(matrix, dtype=float).reshape(4, 4)

    tcp_pose_object = payload.get("tcp_pose_object")
    object_name = payload.get("object_name")
    if isinstance(tcp_pose_object, dict) and object_name:
        if env is None:
            raise ValueError(
                f"manual grasp pose requires an initialized environment: {path}"
            )
        from bench.execution.planning.pick_place_targets import object_pose_matrix

        tcp_object = pose_matrix(
            np.array(tcp_pose_object["p"], dtype=float),
            _manual_grasp_object_rotation(payload),
        )
        return object_pose_matrix(env, str(object_name)) @ tcp_object

    raise ValueError(
        "pose file has no pose.world matrix or p + q_xyzw, "
        "selected.eef_site_world.matrix, "
        f"or object-relative manual grasp pose: {path}"
    )


def viewer_render_camera(args: argparse.Namespace) -> str | None:
    value = str(args.render_camera).strip()
    if value.lower() in {"", "none", "free", "layout"}:
        return None
    return value


def explicit_initial_base_pose(args: argparse.Namespace) -> tuple[np.ndarray, float] | None:
    values = (args.initial_base_x, args.initial_base_y, args.initial_base_yaw)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "--initial-base-x, --initial-base-y, and --initial-base-yaw must be provided together"
        )
    return (
        np.array([float(args.initial_base_x), float(args.initial_base_y)], dtype=float),
        float(args.initial_base_yaw),
    )


def apply_initial_base_pose(
    env: Any,
    obs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reset_base_xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    if args.pick_setup_file is not None:
        if explicit_initial_base_pose(args) is not None:
            raise ValueError(
                "--pick-setup-file cannot be combined with explicit --initial-base-*"
            )
        path = Path(args.pick_setup_file)
        payload = json.loads(path.read_text())
        cup_world = ((payload.get("cup") or {}).get("world") or {})
        base_setup = payload.get("base") or {}
        if cup_world.get("p") is None or cup_world.get("q_xyzw") is None:
            raise ValueError(f"pick setup file has no cup world pose: {path}")
        if base_setup.get("xy") is None or base_setup.get("yaw") is None:
            raise ValueError(f"pick setup file has no base pose: {path}")
        env.set_object_pose("cup", cup_world["p"], cup_world["q_xyzw"])
        new_obs = set_base_pose_kinematic(
            env,
            target_xy=np.array(base_setup["xy"], dtype=float),
            target_yaw=float(base_setup["yaw"]),
        )
        new_obs = env._get_observations(force_update=True)
        return new_obs, {
            "enabled": True,
            "mode": "pick_setup_file",
            "file": str(path),
            "reset_base_xy": reset_base_xy.astype(float).tolist(),
            "target_xy": np.array(base_setup["xy"], dtype=float).tolist(),
            "target_yaw": float(base_setup["yaw"]),
            "cup_world": cup_world,
        }
    explicit_pose = explicit_initial_base_pose(args)
    if explicit_pose is None:
        return obs, {
            "enabled": False,
            "mode": "raw_robocasa_reset",
            "reset_base_xy": reset_base_xy.astype(float).tolist(),
        }
    target_xy, target_yaw = explicit_pose
    new_obs = set_base_pose_kinematic(env, target_xy=target_xy, target_yaw=target_yaw)
    return new_obs, {
        "enabled": True,
        "mode": "explicit_pose",
        "reset_base_xy": reset_base_xy.astype(float).tolist(),
        "target_xy": target_xy.astype(float).tolist(),
        "target_yaw": float(target_yaw),
        "actual_base_pos": np.array(new_obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.array(new_obs["robot0_base_quat"], dtype=float).tolist(),
    }


def apply_initial_torso_height(
    env: Any,
    obs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one measured torso height while holding the base and right arm still."""

    from bench.env.robocasa_runtime import qpos_for_controller
    from bench.execution.navigation.base_motion import step_base_and_hold_arm

    robot = env.robots[0]
    composite = robot.composite_controller
    torso = composite.part_controllers.get("torso")
    current = (
        None
        if torso is None
        else float(qpos_for_controller(env, torso).reshape(-1)[0])
    )
    if args.initial_torso_height is None:
        return obs, {
            "applied": False,
            "reason": "initial_torso_height_not_requested",
            "requested_height": None,
            "initial_height": current,
            "actual_height": current,
        }

    target = float(args.initial_torso_height)
    tolerance = float(args.torso_tolerance)
    max_steps = int(args.torso_max_steps)
    if not np.isfinite(target):
        raise ValueError("--initial-torso-height must be finite")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("--torso-tolerance must be a positive finite number")
    if max_steps <= 0:
        raise ValueError("--torso-max-steps must be positive")
    if torso is None:
        raise RuntimeError(
            f"--initial-torso-height={target:.4f}m requested, "
            "but the robot has no torso controller"
        )

    joint_indexes = np.asarray(torso.joint_index, dtype=int).reshape(-1)
    if joint_indexes.size < 1:
        raise RuntimeError("torso controller does not expose a joint index")
    joint_range = np.asarray(
        env.sim.model.jnt_range[int(joint_indexes[0])],
        dtype=float,
    ).reshape(-1)
    if (
        joint_range.size < 2
        or not np.all(np.isfinite(joint_range[:2]))
        or target < float(joint_range[0]) - 1e-9
        or target > float(joint_range[1]) + 1e-9
    ):
        raise ValueError(
            f"--initial-torso-height={target:.4f}m is outside joint range "
            f"{joint_range[:2].tolist()}"
        )

    right = composite.part_controllers.get("right")
    if right is None:
        raise RuntimeError("robot does not expose the right-arm controller")
    base_start, base_end = composite._action_split_indexes["base"]
    base_action = np.zeros(int(base_end) - int(base_start), dtype=float)
    initial = float(current)
    actual = initial
    executed_steps = 0
    for _ in range(max_steps):
        error = target - actual
        if abs(error) <= tolerance:
            break
        torso_action = float(np.clip(error / 0.05, -1.0, 1.0))
        obs = step_base_and_hold_arm(
            env,
            arm_qpos=qpos_for_controller(env, right),
            base_action=base_action,
            base_mode=1.0,
            gripper_action=0.0,
            torso_action=torso_action,
        )
        executed_steps += 1
        actual = float(qpos_for_controller(env, torso).reshape(-1)[0])
    if abs(target - actual) > tolerance:
        raise RuntimeError(
            f"torso failed to reach {target:.4f}m within {max_steps} steps; "
            f"actual={actual:.4f}m"
        )

    return env._get_observations(force_update=True), {
        "applied": True,
        "reason": "initial_torso_height_reached",
        "requested_height": target,
        "initial_height": initial,
        "actual_height": actual,
        "error": actual - target,
        "tolerance": tolerance,
        "steps": executed_steps,
        "joint_range": joint_range[:2].astype(float).tolist(),
    }


def apply_initial_object_pose(env: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Stage one free object's body center and orientation for calibration."""
    object_name = str(args.initial_object_name or "").strip()
    body_world = optional_vector3(args.initial_object_body_world)
    quat_value = args.initial_object_quat_xyzw
    if not object_name:
        if body_world is not None or quat_value is not None:
            raise ValueError(
                "--initial-object-name is required with initial object pose options"
            )
        return {"enabled": False}
    if object_name not in getattr(env, "objects", {}):
        raise KeyError(f"initial object {object_name!r} is not present in environment")

    obj = env.objects[object_name]
    joint_name = obj.joints[0]
    qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=float).copy()
    if quat_value is not None:
        parts = [part.strip() for part in str(quat_value).split(",") if part.strip()]
        if len(parts) != 4:
            raise ValueError(
                f"expected x,y,z,w quaternion, got {quat_value!r}"
            )
        quat_xyzw = np.array([float(part) for part in parts], dtype=float)
        norm = float(np.linalg.norm(quat_xyzw))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("initial object quaternion must be finite and nonzero")
        quat_xyzw /= norm
        qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]
        env.sim.data.set_joint_qpos(joint_name, qpos)
        env.sim.forward()
    else:
        quat_xyzw = None

    if body_world is not None:
        if hasattr(env, "set_object_body_position"):
            env.set_object_body_position(object_name, body_world)
        else:
            body_id = env.obj_body_id[object_name]
            current_body = np.array(env.sim.data.body_xpos[body_id], dtype=float)
            qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=float).copy()
            qpos[:3] += np.array(body_world, dtype=float) - current_body
            env.sim.data.set_joint_qpos(joint_name, qpos)
            env.sim.forward()

    body_id = env.obj_body_id[object_name]
    return {
        "enabled": True,
        "object_name": object_name,
        "body_world": np.array(env.sim.data.body_xpos[body_id], dtype=float).tolist(),
        "body_quat_wxyz": np.array(
            env.sim.data.body_xquat[body_id], dtype=float
        ).tolist(),
    }


def apply_slide_door_open_fraction(
    env: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Stage either one previous storage door or named multi-level doors."""

    if args.slide_door_open_fraction is None:
        if args.slide_door_levels:
            raise ValueError(
                "--slide-door-open-fraction is required with --slide-door-levels"
            )
        return {"enabled": False}

    fraction = float(args.slide_door_open_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("--slide-door-open-fraction must be within [0, 1]")
    levels = [
        value.strip()
        for value in str(args.slide_door_levels or "").split(",")
        if value.strip()
    ]
    if levels:
        setter = getattr(env, "set_level_door_open_fraction", None)
        if not callable(setter):
            raise ValueError(
                "--slide-door-levels requires an environment with "
                "set_level_door_open_fraction(level, fraction)"
            )
        for level in levels:
            setter(level, fraction)
        env.sim.forward()
        return {
            "enabled": True,
            "mode": "named_levels",
            "levels": levels,
            "fraction": fraction,
        }

    cabinet = getattr(env, "storage_cabinet", None)
    open_slot = getattr(cabinet, "open_slot", None)
    if not callable(open_slot):
        raise ValueError(
            "environment has no storage_cabinet.open_slot(); "
            "provide --slide-door-levels for a multi-level environment"
        )
    open_slot(
        "cabinet_slot_0",
        env=env,
        min=fraction,
        max=fraction,
    )
    env.sim.forward()
    return {
        "enabled": True,
        "mode": "storage_cabinet",
        "levels": ["cabinet_slot_0"],
        "fraction": fraction,
    }


def _site_pos(env: Any, name: str) -> np.ndarray:
    site_id = env.sim.model.site_name2id(name)
    return np.array(env.sim.data.site_xpos[site_id], dtype=float).reshape(3)


def _body_pos(env: Any, name: str) -> np.ndarray:
    body_id = env.sim.model.body_name2id(name)
    return np.array(env.sim.data.body_xpos[body_id], dtype=float).reshape(3)


def _geom_pos(env: Any, name: str) -> np.ndarray:
    geom_id = env.sim.model.geom_name2id(name)
    return np.array(env.sim.data.geom_xpos[geom_id], dtype=float).reshape(3)


def _env_attr_pos(env: Any, name: str) -> np.ndarray:
    value = getattr(env, name)
    if callable(value):
        value = value()
    if hasattr(value, "pos"):
        value = getattr(value, "pos")
    return vector3(value)


def _object_pos(env: Any, name: str) -> np.ndarray:
    if hasattr(env, "obj_body_id") and name in env.obj_body_id:
        return np.array(env.sim.data.body_xpos[env.obj_body_id[name]], dtype=float).reshape(3)
    return _body_pos(env, name)


def _sink_water_site_pos(env: Any) -> np.ndarray:
    sink = getattr(env, "sink", None)
    if sink is None:
        raise ValueError("env has no sink attribute")
    site = getattr(sink, "water_site", None)
    if isinstance(site, dict) and site.get("name"):
        return _site_pos(env, str(site["name"]))
    if hasattr(site, "get") and site.get("name"):
        return _site_pos(env, str(site.get("name")))
    raise ValueError("sink water site name is unavailable")


def resolve_anchor_world(env: Any, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    selector = str(args.anchor or "world").strip()
    if selector == "world":
        if args.anchor_world is None:
            pos = np.zeros(3, dtype=float)
        else:
            pos = vector3(args.anchor_world)
        return pos, {"selector": selector, "source": "anchor_world_arg"}
    try:
        if selector == "sink_water_site":
            pos = _sink_water_site_pos(env)
        elif selector.startswith("site:"):
            pos = _site_pos(env, selector.split(":", 1)[1])
        elif selector.startswith("body:"):
            pos = _body_pos(env, selector.split(":", 1)[1])
        elif selector.startswith("geom:"):
            pos = _geom_pos(env, selector.split(":", 1)[1])
        elif selector.startswith("object:"):
            pos = _object_pos(env, selector.split(":", 1)[1])
        elif selector.startswith("fixture:"):
            pos = _env_attr_pos(env, selector.split(":", 1)[1])
        elif selector.startswith("env_attr:"):
            pos = _env_attr_pos(env, selector.split(":", 1)[1])
        else:
            raise ValueError(f"unsupported anchor selector {selector!r}")
        return pos, {"selector": selector, "source": "environment"}
    except Exception:
        if args.anchor_world is None:
            raise
        pos = vector3(args.anchor_world)
        return pos, {"selector": selector, "source": "anchor_world_fallback"}


def apply_viewer_camera(env: Any, args: argparse.Namespace, lookat: np.ndarray) -> dict[str, Any] | None:
    if args.no_render or viewer_render_camera(args) is not None:
        return None
    config = {
        "lookat": np.array(lookat, dtype=float).reshape(3).tolist(),
        "distance": float(args.viewer_distance),
        "azimuth": float(args.viewer_azimuth),
        "elevation": float(args.viewer_elevation),
    }
    env.renderer_config = dict(getattr(env, "renderer_config", {}) or {})
    env.renderer_config["cam_config"] = config
    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        viewer.camera_config = config
        mj_viewer = getattr(viewer, "viewer", None)
        if mj_viewer is not None:
            mj_viewer.cam.type = 0
            mj_viewer.cam.lookat = config["lookat"]
            mj_viewer.cam.distance = config["distance"]
            mj_viewer.cam.azimuth = config["azimuth"]
            mj_viewer.cam.elevation = config["elevation"]
    return config


def print_help() -> None:
    print(
        "\nAnchor pose controls:\n"
        "  x+/x-/y+/y-/z+/z-      move pose in world axes\n"
        "                           optionally: x+ 0.02 uses a one-off meter step\n"
        "  up/down                aliases for z+/z-\n"
        "  rx+/rx-/ry+/ry-/rz+/rz- rotate pose about its local axes\n"
        "                           optionally: rz+ 5 uses a one-off degree step\n"
        "  reset-rot              reset rotation to identity\n"
        "  test                   plan to the current marker without moving the arm\n"
        "  move                   plan and move the real gripper to the current marker\n"
        "  close/open             close or open the gripper while holding the arm\n"
        "  state                  print live gripper and task interaction state\n"
        "  p                      print current pose\n"
        "  save                   write pose JSON and exit\n"
        "  h                      show this help\n"
        "  quit                   exit without writing\n",
        flush=True,
    )


MOVE_COMMANDS = {"x+", "x-", "y+", "y-", "z+", "z-", "up", "down"}
ROT_COMMANDS = {"rx+", "rx-", "ry+", "ry-", "rz+", "rz-"}
STEP_COMMANDS = MOVE_COMMANDS | ROT_COMMANDS


def _is_float_token(token: str) -> bool:
    try:
        value = float(token)
    except ValueError:
        return False
    return bool(np.isfinite(value))


def command_step(tokens: list[str], default: float, *, unit: str) -> float:
    if len(tokens) <= 1:
        return float(default)
    if len(tokens) > 2:
        raise ValueError(f"expected '<direction> <step_{unit}>'")
    value = float(tokens[1])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"step_{unit} must be a positive finite number")
    return value


def command_stream_context(require_interactive: bool):
    if sys.stdin is not None and sys.stdin.isatty():
        return contextlib.nullcontext(sys.stdin)
    try:
        return open("/dev/tty", "r", buffering=1)
    except OSError as exc:
        if require_interactive:
            raise ValueError(
                "interactive calibration requires a TTY; use --save-immediately "
                "for noninteractive runs"
            ) from exc
        return contextlib.nullcontext(None)


def _line_to_command_tokens(line: str) -> list[list[str]]:
    tokens = [token.lower() for token in line.strip().split() if token.strip()]
    commands: list[list[str]] = []
    index = 0
    while index < len(tokens):
        command = tokens[index]
        if (
            command in STEP_COMMANDS
            and index + 1 < len(tokens)
            and _is_float_token(tokens[index + 1])
        ):
            commands.append([command, tokens[index + 1]])
            index += 2
        else:
            commands.append([command])
            index += 1
    return commands


def drain_terminal_commands(command_file: Any | None) -> list[list[str]]:
    if command_file is None:
        return []
    commands: list[list[str]] = []
    while True:
        ready, _, _ = select.select([command_file], [], [], 0.0)
        if not ready:
            return commands
        line = command_file.readline()
        if line == "":
            return commands
        commands.extend(_line_to_command_tokens(line))


def apply_command(
    pos: np.ndarray,
    rot: np.ndarray,
    tokens: list[str],
    args: argparse.Namespace,
) -> bool:
    command = tokens[0] if tokens else ""
    if command in MOVE_COMMANDS:
        if command in {"up", "down"}:
            axis = 2
            sign = 1.0 if command == "up" else -1.0
        else:
            axis = {"x": 0, "y": 1, "z": 2}[command[0]]
            sign = 1.0 if command.endswith("+") else -1.0
        step = command_step(tokens, float(args.step_pos), unit="meters")
        pos[axis] += sign * step
        return True
    if command in ROT_COMMANDS:
        sign = 1.0 if command.endswith("+") else -1.0
        step_deg = command_step(tokens, float(args.step_rot_deg), unit="degrees")
        angle = np.deg2rad(sign * step_deg)
        if command.startswith("rx"):
            delta = _rotation_x(angle)
        elif command.startswith("ry"):
            delta = _rotation_y(angle)
        else:
            delta = _rotation_z(angle)
        rot[:] = rot @ delta
        u, _, vt = np.linalg.svd(rot)
        rot[:] = u @ vt
        if np.linalg.det(rot) < 0.0:
            u[:, -1] *= -1.0
            rot[:] = u @ vt
        return True
    if command == "reset-rot":
        rot[:] = np.eye(3, dtype=float)
        return True
    return False


def summary_line(anchor_world: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> str:
    quat = mat_to_quat_xyzw(rot)
    offset = np.array(pos, dtype=float) - np.array(anchor_world, dtype=float)
    return (
        f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
        f"offset=({offset[0]:+.4f},{offset[1]:+.4f},{offset[2]:+.4f}) "
        f"q_xyzw=({quat[0]:+.4f},{quat[1]:+.4f},{quat[2]:+.4f},{quat[3]:+.4f})"
    )


def add_marker_geoms(
    viewer: Any,
    *,
    anchor_world: np.ndarray,
    pos: np.ndarray,
    rot: np.ndarray,
    axis_length: float,
    axis_radius: float,
) -> None:
    mj_viewer = getattr(viewer, "viewer", None)
    if mj_viewer is None or getattr(mj_viewer, "user_scn", None) is None:
        return
    scn = mj_viewer.user_scn
    scn.ngeom = 0
    if scn.ngeom + 7 >= scn.maxgeom:
        return
    anchor = np.array(anchor_world, dtype=float).reshape(3)
    target = np.array(pos, dtype=float).reshape(3)
    specs = [
        (anchor, np.array([0.05, 0.25, 1.0, 0.9], dtype=float), axis_radius * 1.8),
        (target, np.array([1.0, 0.2, 0.05, 1.0], dtype=float), axis_radius * 2.4),
    ]
    for point, rgba, radius in specs:
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=float),
            point,
            np.eye(3).reshape(-1),
            rgba,
        )
        scn.ngeom += 1

    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([axis_radius * 0.7, 0.0, 0.0], dtype=float),
        anchor,
        np.eye(3).reshape(-1),
        np.array([1.0, 0.85, 0.05, 0.9], dtype=float),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        axis_radius * 0.7,
        anchor,
        target,
    )
    scn.ngeom += 1

    axes = [
        (rot[:, 0], np.array([1.0, 0.05, 0.05, 1.0], dtype=float)),
        (rot[:, 1], np.array([0.05, 0.85, 0.05, 1.0], dtype=float)),
        (rot[:, 2], np.array([0.05, 0.25, 1.0, 1.0], dtype=float)),
    ]
    for axis, rgba in axes:
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([axis_radius, 0.0, 0.0], dtype=float),
            target,
            np.eye(3).reshape(-1),
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            axis_radius,
            target,
            target + np.array(axis, dtype=float) * float(axis_length),
        )
        scn.ngeom += 1


def render_with_marker(
    env: Any,
    args: argparse.Namespace,
    *,
    anchor_world: np.ndarray,
    pos: np.ndarray,
    rot: np.ndarray,
) -> None:
    if args.no_render:
        return
    viewer = getattr(env, "viewer", None)
    if viewer is not None and hasattr(viewer, "update"):
        viewer.update()
    else:
        env.render()
    add_marker_geoms(
        env.viewer,
        anchor_world=anchor_world,
        pos=pos,
        rot=rot,
        axis_length=float(args.axis_length),
        axis_radius=float(args.axis_radius),
    )
    mj_viewer = getattr(getattr(env, "viewer", None), "viewer", None)
    if mj_viewer is not None:
        mj_viewer.sync()


def pose_payload(
    env: Any,
    obs: dict[str, Any],
    args: argparse.Namespace,
    *,
    anchor_world: np.ndarray,
    anchor_meta: dict[str, Any],
    initial_base: dict[str, Any],
    initial_torso: dict[str, Any],
    pos: np.ndarray,
    rot: np.ndarray,
) -> dict[str, Any]:
    mat = pose_matrix(pos, rot)
    if str(args.output_format) == "manual_grasp_pose":
        from bench.execution.navigation.base_motion import base_yaw
        from bench.execution.planning import matrix_from_base_obs
        from bench.execution.planning.pick_place_targets import object_pose_matrix

        object_name = str(
            args.grasp_object
            or args.tool_name
            or ""
        ).strip()
        if not object_name:
            raise ValueError(
                "--grasp-object or --tool-name is required with "
                "--output-format manual_grasp_pose"
            )
        if object_name not in env.obj_body_id:
            raise KeyError(f"grasp object {object_name!r} not present in environment")
        object_world = object_pose_matrix(env, object_name)
        tcp_in_object = np.linalg.inv(object_world) @ mat
        base_world = matrix_from_base_obs(obs)
        tcp_in_base = np.linalg.inv(base_world) @ mat
        tcp_quat_xyzw = np.array(mat_to_quat_xyzw(tcp_in_object[:3, :3]), dtype=float)
        asset = str(
            args.grasp_asset
            or args.tool_object_group
            or object_name
        )
        return {
            "schema_version": 2,
            "config_kind": "manual_grasp_pose",
            "pose_name": str(args.pose_name),
            "task": {
                "env_name": str(args.env_name),
                "split": str(args.split),
                "seed": int(args.seed),
                "layout_id": args.layout_id,
                "style_id": args.style_id,
            },
            "object_name": object_name,
            "asset": asset,
            "pregrasp_offset_tcp": vector3(args.pregrasp_offset_tcp).tolist(),
            "tcp_pose_object": {
                "p": np.array(tcp_in_object[:3, 3], dtype=float).tolist(),
                "q_xyzw": tcp_quat_xyzw.tolist(),
                "q_wxyz": tcp_quat_xyzw[[3, 0, 1, 2]].tolist(),
            },
            "selected": {
                "eef_site_world": pose_from_matrix(mat),
                "eef_site_in_base": pose_from_matrix(tcp_in_base),
                "eef_site_in_object": pose_from_matrix(tcp_in_object),
            },
            "object": {
                "world": pose_from_matrix(object_world),
            },
            "base": {
                "world": pose_from_matrix(base_world),
                "xy": np.array(obs["robot0_base_pos"][:2], dtype=float).tolist(),
                "yaw": float(base_yaw(obs)),
            },
            "anchor": {
                **anchor_meta,
                "world": np.array(anchor_world, dtype=float).astype(float).tolist(),
            },
            "initial_base": initial_base,
            "initial_torso": initial_torso,
            "calibrated_at_unix": float(time.time()),
        }
    return {
        "schema_version": 1,
        "config_kind": "anchor_pose",
        "pose_name": str(args.pose_name),
        "task": {
            "env_name": str(args.env_name),
            "split": str(args.split),
            "seed": int(args.seed),
            "layout_id": args.layout_id,
            "style_id": args.style_id,
        },
        "anchor": {
            **anchor_meta,
            "world": np.array(anchor_world, dtype=float).astype(float).tolist(),
        },
        "initial_base": initial_base,
        "initial_torso": initial_torso,
        "pose": {
            "world": pose_from_matrix(mat),
            "offset_from_anchor_world": (
                np.array(pos, dtype=float) - np.array(anchor_world, dtype=float)
            )
            .astype(float)
            .tolist(),
        },
        "calibrated_at_unix": float(time.time()),
    }


def write_pose(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def make_subprocess_planner(args: argparse.Namespace):
    from bench.execution.planning.mplib_planner import run_mplib_planner_subprocess

    def planner(payload: dict[str, Any]) -> dict[str, Any]:
        code, result = run_mplib_planner_subprocess(
            script_path=args.planner_script,
            plan_request=payload,
            conda_env=str(args.planner_conda_env),
        )
        if result is None:
            return {
                "subprocess_returncode": int(code),
                "plan": {"status": "planner_subprocess_no_json", "attempts": []},
            }
        result["subprocess_returncode"] = int(code)
        return result

    return planner


def _manual_grasp_candidate_payload(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.initial_orientation_grasp_file)
    payload = json.loads(path.read_text())
    if payload.get("config_kind") != "poses":
        return payload
    candidates = (
        (payload.get("operations") or {}).get("pick", {}).get("candidates", [])
    )
    index = int(args.initial_orientation_candidate) - 1
    if not 0 <= index < len(candidates):
        raise IndexError(
            f"manual grasp candidate must be within [1, {len(candidates)}], "
            f"got {args.initial_orientation_candidate}"
        )
    return dict(candidates[index])


def _manual_grasp_object_rotation(payload: dict[str, Any]) -> np.ndarray:
    pose = payload.get("tcp_pose_object")
    if not isinstance(pose, dict):
        pose = (payload.get("selected") or {}).get("eef_site_in_object")
    if not isinstance(pose, dict):
        raise ValueError("manual grasp payload has no object-local TCP pose")
    matrix = pose.get("matrix")
    if matrix is not None:
        return np.array(matrix, dtype=float).reshape(4, 4)[:3, :3]
    quat = pose.get("q_xyzw")
    if quat is None and pose.get("q_wxyz") is not None:
        q = np.array(pose["q_wxyz"], dtype=float).reshape(4)
        quat = q[[1, 2, 3, 0]]
    if quat is None:
        raise ValueError("manual grasp object-local TCP pose has no quaternion")
    x, y, z, w = np.array(quat, dtype=float).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def initial_marker_pose(
    env: Any,
    args: argparse.Namespace,
    anchor_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.initial_pose_file is not None:
        pose_path = Path(args.initial_pose_file)
        payload = json.loads(pose_path.read_text(encoding="utf-8"))
        if bool(args.reanchor_initial_pose):
            if str(payload.get("config_kind", "")) == "manual_grasp_pose":
                from bench.execution.planning import matrix_from_json_pose
                from bench.execution.planning.pick_place_targets import object_pose_matrix

                object_name = str(payload.get("object_name") or "").strip()
                if not object_name:
                    raise ValueError(
                        "manual grasp initial pose has no object_name: "
                        f"{pose_path}"
                    )
                expected_selector = f"object:{object_name}"
                live_selector = str(args.anchor or "")
                if live_selector != expected_selector:
                    raise ValueError(
                        "reanchored manual grasp requires its object anchor: "
                        f"expected={expected_selector!r} live={live_selector!r}"
                    )
                selected = payload.get("selected") or {}
                eef_in_object = (
                    selected.get("eef_site_in_object")
                    if isinstance(selected, dict)
                    else None
                )
                if not isinstance(eef_in_object, dict):
                    eef_in_object = payload.get("tcp_pose_object")
                if not isinstance(eef_in_object, dict):
                    raise ValueError(
                        "manual grasp initial pose has no object-relative EEF pose: "
                        f"{pose_path}"
                    )
                mat = (
                    object_pose_matrix(env, object_name)
                    @ matrix_from_json_pose(eef_in_object)
                )
                return (
                    np.array(mat[:3, 3], dtype=float),
                    np.array(mat[:3, :3], dtype=float),
                    {
                        "mode": "reanchored_manual_grasp_pose_file",
                        "pose_file": str(pose_path),
                        "selector": live_selector,
                        "object_name": object_name,
                        "eef_in_object_source": (
                            "selected.eef_site_in_object"
                            if isinstance(
                                (payload.get("selected") or {}).get(
                                    "eef_site_in_object"
                                ),
                                dict,
                            )
                            else "tcp_pose_object"
                        ),
                    },
                )
            pose_payload_data = payload.get("pose") or {}
            offset = pose_payload_data.get("offset_from_anchor_world")
            if offset is None:
                raise ValueError(
                    "--reanchor-initial-pose requires "
                    "pose.offset_from_anchor_world in the initial pose file"
                )
            saved_selector = str((payload.get("anchor") or {}).get("selector", ""))
            live_selector = str(args.anchor or "")
            if saved_selector and saved_selector != live_selector:
                raise ValueError(
                    "reanchored initial pose selector mismatch: "
                    f"saved={saved_selector!r} live={live_selector!r}"
                )
            mat = load_pose_world_matrix(pose_path, env)
            resolved_offset = vector3(offset)
            mat = np.array(mat, dtype=float, copy=True).reshape(4, 4)
            mat[:3, 3] = np.asarray(anchor_world, dtype=float) + resolved_offset
            return (
                np.array(mat[:3, 3], dtype=float),
                np.array(mat[:3, :3], dtype=float),
                {
                    "mode": "reanchored_pose_file",
                    "pose_file": str(pose_path),
                    "selector": saved_selector or live_selector,
                    "offset_from_anchor_world": resolved_offset.tolist(),
                },
            )
        mat = load_pose_world_matrix(pose_path, env)
        return (
            np.array(mat[:3, 3], dtype=float),
            np.array(mat[:3, :3], dtype=float),
            {
                "mode": "pose_file",
                "pose_file": str(pose_path),
            },
        )
    if args.initial_orientation_grasp_file is not None:
        if not args.initial_orientation_object:
            raise ValueError(
                "--initial-orientation-object is required with "
                "--initial-orientation-grasp-file"
            )
        object_name = str(args.initial_orientation_object)
        body_id = env.obj_body_id[object_name]
        object_rot = np.array(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
        payload = _manual_grasp_candidate_payload(args)
        grasp_rot = object_rot @ _manual_grasp_object_rotation(payload)
        return (
            np.array(anchor_world, dtype=float) + vector3(args.initial_offset),
            grasp_rot,
            {
                "mode": "manual_grasp_orientation",
                "grasp_file": str(Path(args.initial_orientation_grasp_file)),
                "candidate": int(args.initial_orientation_candidate),
                "object_name": object_name,
                "initial_offset": vector3(args.initial_offset).astype(float).tolist(),
            },
        )
    return (
        np.array(anchor_world, dtype=float) + vector3(args.initial_offset),
        euler_xyz_deg_to_matrix(
            args.initial_roll_deg,
            args.initial_pitch_deg,
            args.initial_yaw_deg,
        ),
        {
            "mode": "offset_and_euler",
            "initial_offset": vector3(args.initial_offset).astype(float).tolist(),
            "initial_roll_deg": float(args.initial_roll_deg),
            "initial_pitch_deg": float(args.initial_pitch_deg),
            "initial_yaw_deg": float(args.initial_yaw_deg),
        },
    )


def move_gripper_to_initial_pose(
    env: Any,
    obs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if args.initial_pose_file is None or not bool(args.move_gripper_to_initial_pose):
        return obs, None

    from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
    from bench.execution.planning import MplibPlannerConfig, matrix_from_base_obs
    from bench.execution.planning.pick_place_targets import (
        make_manual_object_pick_target_poses,
        pose_pair_from_world,
    )

    pose_path = Path(args.initial_pose_file)
    pose_payload_data = json.loads(pose_path.read_text())
    is_manual_grasp = bool(
        isinstance(pose_payload_data.get("tcp_pose_object"), dict)
        and pose_payload_data.get("object_name")
    )
    if is_manual_grasp:
        object_name = str(pose_payload_data["object_name"])
        pick_targets = make_manual_object_pick_target_poses(
            env,
            obs,
            object_name=object_name,
            grasp_pose_file=pose_path,
            pregrasp_distance=None,
            lift_height=0.0,
        )
        stage_targets = [
            ("initial_pregrasp", pick_targets["poses"]["pregrasp"]["base"]),
            ("initial_grasp", pick_targets["poses"]["grasp"]["base"]),
        ]
        target_metadata = {
            "mode": "manual_grasp_sequence",
            "object_name": object_name,
            "pregrasp_offset_tcp": pick_targets["manual_pregrasp_offset_tcp"],
            "pregrasp_distance": pick_targets["manual_pregrasp_distance"],
            "pregrasp_source": pick_targets["manual_pregrasp_source"],
        }
    else:
        grasp_world = load_pose_world_matrix(pose_path, env)
        base_world = matrix_from_base_obs(obs)
        target_pose = pose_pair_from_world(base_world, grasp_world)
        stage_targets = [("initial_anchor_pose", target_pose["base"])]
        target_metadata = {"mode": "single_anchor_pose"}

    planner_config = MplibPlannerConfig(
        planning_time=float(args.planning_time),
        time_step=float(args.time_step),
        qpos_step=float(args.qpos_step),
        rrt_range=float(args.rrt_range),
    )
    runtime_config = ManipulationRuntimeConfig(
        execute=True,
        max_joint_step=float(args.max_joint_step),
        settle_steps=int(args.settle_steps),
        position_threshold=float(args.position_threshold),
        orientation_threshold_deg=float(args.orientation_threshold_deg),
        qpos_threshold=float(args.qpos_threshold),
        planner_args=planner_config.planner_args(),
    )
    controller = ManipulationController(
        env,
        obs,
        planner=make_subprocess_planner(args),
        runtime_config=runtime_config,
    )
    open_result = controller.driver.run_gripper(
        arm_qpos=controller.driver.current_arm_qpos(),
        gripper_action=-1.0,
        steps=int(args.initial_gripper_open_steps),
    )
    stages = []
    for stage_name, stage_target in stage_targets:
        stage = controller.stage_executor.run(
            stage_target,
            stage=stage_name,
            gripper_action=-1.0,
            execute=True,
            record_trajectory=False,
        )
        controller._sync_obs(controller.stage_executor.obs)
        stages.append(stage)
        if not bool(stage.get("pass", False)):
            break
    passed = bool(
        len(stages) == len(stage_targets)
        and all(bool(stage.get("pass", False)) for stage in stages)
    )
    summary = {
        "pose_file": str(pose_path),
        **target_metadata,
        "open_gripper": _without_obs(open_result),
        "move_stages": _without_obs(stages),
        "move_stage": _without_obs(stages[-1]),
        "pass": passed,
    }
    if not bool(summary["pass"]):
        raise RuntimeError(
            "failed to move gripper to --initial-pose-file target; "
            f"stage={json.dumps(jsonable(summary['move_stage']), sort_keys=True)}"
        )
    return controller.obs, summary


def plan_or_move_gripper_to_marker(
    env: Any,
    obs: dict[str, Any],
    args: argparse.Namespace,
    *,
    pos: np.ndarray,
    rot: np.ndarray,
    execute: bool,
    gripper_action: float = -1.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check marker reachability, optionally moving the physical arm there."""

    from bench.execution.manipulation import ManipulationController, ManipulationRuntimeConfig
    from bench.execution.planning import MplibPlannerConfig, matrix_from_base_obs
    from bench.execution.planning.pick_place_targets import pose_pair_from_world

    current_obs = env._get_observations(force_update=True)
    target_world = pose_matrix(pos, rot)
    target = pose_pair_from_world(
        matrix_from_base_obs(current_obs),
        target_world,
    )["base"]
    planner_config = MplibPlannerConfig(
        planning_time=float(args.planning_time),
        time_step=float(args.time_step),
        qpos_step=float(args.qpos_step),
        rrt_range=float(args.rrt_range),
    )
    controller = ManipulationController(
        env,
        current_obs,
        planner=make_subprocess_planner(args),
        runtime_config=ManipulationRuntimeConfig(
            execute=bool(execute),
            max_joint_step=float(args.max_joint_step),
            settle_steps=int(args.settle_steps),
            position_threshold=float(args.position_threshold),
            orientation_threshold_deg=float(args.orientation_threshold_deg),
            qpos_threshold=float(args.qpos_threshold),
            planner_args=planner_config.planner_args(),
        ),
    )
    stage = controller.stage_executor.run(
        target,
        stage="interactive_marker_move" if execute else "interactive_marker_test",
        gripper_action=float(gripper_action),
        execute=bool(execute),
        record_trajectory=False,
        commit_virtual_state=False,
    )
    controller._sync_obs(controller.stage_executor.obs)
    result = {
        "mode": "move" if execute else "test",
        "gripper_action": float(gripper_action),
        "target_world": target_world.tolist(),
        "stage": _without_obs(stage),
        "pass": bool(stage.get("pass", False)),
    }
    return controller.obs if execute else obs, result


def run_interactive_gripper_command(
    env: Any,
    obs: dict[str, Any],
    *,
    gripper_action: float,
    steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open or close the gripper without moving the arm or mobile base."""

    if int(steps) <= 0:
        raise ValueError("--interactive-gripper-steps must be positive")
    from bench.execution.manipulation.drivers import RoboCasaArmDriver

    driver = RoboCasaArmDriver(env, obs)
    result = driver.run_gripper(
        arm_qpos=driver.current_arm_qpos(),
        gripper_action=float(gripper_action),
        steps=int(steps),
    )
    return result["obs"], _without_obs(result)


def interactive_state(env: Any, obs: dict[str, Any]) -> dict[str, Any]:
    """Return generic live state plus task-native control state when available."""

    current = env._get_observations(force_update=True)
    result: dict[str, Any] = {
        "timestep": int(getattr(env, "timestep", 0)),
        "gripper_qpos": np.asarray(
            current.get("robot0_gripper_qpos", ()), dtype=float
        ).tolist(),
    }
    for key, attribute in (
        ("native_knob_qpos_rad", "native_knob_qpos"),
        ("native_knob_qvel_rad_s", "native_knob_qvel"),
    ):
        value = getattr(env, attribute, None)
        if callable(value):
            result[key] = float(value())
    stove = getattr(env, "stove", None)
    knob = getattr(env, "knob", None)
    if stove is not None and knob is not None:
        result["target_knob"] = str(knob)
        result["target_burner_on"] = bool(
            stove.is_burner_on(env=env, burner_loc=knob)
        )
    diagnostics = getattr(env, "task_diagnostics", None)
    if callable(diagnostics):
        result["task_diagnostics"] = diagnostics()
    del obs
    return result


def _without_obs(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_obs(item) for key, item in value.items() if key != "obs"}
    if isinstance(value, list):
        return [_without_obs(item) for item in value]
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.reanchor_initial_pose) and bool(args.move_gripper_to_initial_pose):
        raise ValueError(
            "--reanchor-initial-pose cannot be combined with "
            "--move-gripper-to-initial-pose"
        )
    env_kwargs = {"render_camera": viewer_render_camera(args), "renderer": "mjviewer"}
    wr02_setup = None
    if str(args.env_name) == "WR02ReturnCondimentPair":
        from bench.tasks.wr02.config import load_task_config
        from bench.tasks.wr02.domain.episode import resolve_episode_plan

        wr02_config = (
            load_task_config()
            if args.task_config is None
            else load_task_config(Path(args.task_config))
        )
        wr02_episode = resolve_episode_plan(
            seed=int(args.seed),
            **dict(wr02_config.episode),
        )
        wr02_semantics = dict(wr02_config.semantics)
        env_kwargs.update(
            {
                "home_assignment": dict(wr02_episode.home_assignment),
                "home_region_placements": {
                    name: dict(values)
                    for name, values in wr02_config.initial_state.home_regions.items()
                },
                "use_assignment": dict(wr02_episode.use_assignment),
                "episode_case_id": int(wr02_episode.case_id),
                "home_threshold": float(
                    wr02_semantics.get("home_threshold", 0.16)
                ),
                "sink_stage_threshold": float(
                    wr02_semantics.get("sink_stage_threshold", 0.20)
                ),
                "return_region_xy_margin": float(
                    wr02_semantics.get("return_region_xy_margin", 0.04)
                ),
                "return_region_height_tolerance": float(
                    wr02_semantics.get("return_region_height_tolerance", 0.10)
                ),
                "return_level_min_separation": float(
                    wr02_semantics.get("return_level_min_separation", 0.20)
                ),
            }
        )
        wr02_setup = (wr02_config, wr02_episode)
    if args.preview_storage_fixture:
        env_kwargs["preview_storage_fixture"] = str(args.preview_storage_fixture)
    tabletop_storage_world_xyz = optional_vector3(args.tabletop_storage_world_xyz)
    if tabletop_storage_world_xyz is not None:
        env_kwargs["tabletop_storage_world_xyz"] = tabletop_storage_world_xyz
    if args.tabletop_storage_world_yaw_deg is not None:
        env_kwargs["tabletop_storage_world_yaw_deg"] = float(args.tabletop_storage_world_yaw_deg)
    if args.tool_name or args.tool_object_group or args.tool_object_scale is not None:
        if not args.tool_name or not args.tool_object_group:
            raise ValueError(
                "--tool-name and --tool-object-group are required together"
            )
        tool_name = str(args.tool_name)
        env_kwargs["tool_names"] = (tool_name,)
        env_kwargs["tool_object_groups"] = {
            tool_name: str(args.tool_object_group)
        }
        if args.tool_object_scale is not None:
            env_kwargs["tool_object_scales"] = {
                tool_name: float(args.tool_object_scale)
            }

    env = make_robocasa_env(
        env_name=str(args.env_name),
        split=str(args.split),
        seed=int(args.seed),
        controller_config=make_joint_position_controller_config(),
        camera_names=["robot0_agentview_right"],
        camera_widths=int(args.camera_width),
        camera_heights=int(args.camera_height),
        layout_id=args.layout_id,
        style_id=args.style_id,
        has_renderer=not bool(args.no_render),
        has_offscreen_renderer=bool(args.no_render),
        use_camera_obs=bool(args.no_render),
        env_kwargs=env_kwargs,
    )
    try:
        obs = reset_env_with_initial_arm_qpos(env)
        wr02_initialization = None
        if wr02_setup is not None:
            from bench.tasks.wr02.initialization import initialize_episode_state

            wr02_config, wr02_episode = wr02_setup
            obs, wr02_initialization = initialize_episode_state(
                env,
                obs,
                episode=wr02_episode,
                robot_base=wr02_config.initial_state.robot_base,
                objects=dict(wr02_config.initial_state.objects),
            )
        if bool(args.stage_tools_on_storage):
            stage_tools = getattr(env, "_fix_storage_tool_placements", None)
            if not callable(stage_tools):
                raise ValueError(
                    "--stage-tools-on-storage requires an environment with "
                    "_fix_storage_tool_placements()"
                )
            stage_tools()
            env.sim.forward()
            obs = env._get_observations(force_update=True)
        initial_object = apply_initial_object_pose(env, args)
        obs = env._get_observations(force_update=True)
        obs, initial_base = apply_initial_base_pose(env, obs, args)
        obs, initial_torso = apply_initial_torso_height(env, obs, args)
        slide_doors = apply_slide_door_open_fraction(env, args)
        anchor_world, anchor_meta = resolve_anchor_world(env, args)
        pos, rot, initial_pose_meta = initial_marker_pose(env, args, anchor_world)
        obs, initial_gripper_move = move_gripper_to_initial_pose(env, obs, args)
        viewer_config = apply_viewer_camera(env, args, pos)

        print_help()
        print(f"output={args.output}", flush=True)
        print(f"initial_object={json.dumps(jsonable(initial_object), sort_keys=True)}", flush=True)
        print(f"slide_doors={json.dumps(jsonable(slide_doors), sort_keys=True)}", flush=True)
        print(f"anchor={json.dumps(jsonable({**anchor_meta, 'world': anchor_world}), sort_keys=True)}", flush=True)
        print(f"initial_base={json.dumps(jsonable(initial_base), sort_keys=True)}", flush=True)
        print(f"initial_torso={json.dumps(jsonable(initial_torso), sort_keys=True)}", flush=True)
        if wr02_initialization is not None:
            print(
                "wr02_initialization="
                f"{json.dumps(jsonable(wr02_initialization), sort_keys=True)}",
                flush=True,
            )
        print(f"initial_pose={json.dumps(jsonable(initial_pose_meta), sort_keys=True)}", flush=True)
        if initial_gripper_move is not None:
            print(f"initial_gripper_move={json.dumps(jsonable(initial_gripper_move), sort_keys=True)}", flush=True)
        if viewer_config is not None:
            print(f"viewer_config={json.dumps(jsonable(viewer_config), sort_keys=True)}", flush=True)
        print(summary_line(anchor_world, pos, rot), flush=True)

        iterations = 0
        last_line = ""
        active_gripper_action = -1.0
        require_interactive = not bool(args.save_immediately) and args.max_iterations is None
        with command_stream_context(require_interactive) as command_file:
            while True:
                iterations += 1
                should_exit = False
                should_save = bool(args.save_immediately)
                force_print = False
                changed = False
                for command_tokens in drain_terminal_commands(command_file):
                    command = command_tokens[0] if command_tokens else ""
                    if command in {"q", "quit", "exit"}:
                        should_exit = True
                    elif command in {"h", "help", "?"}:
                        print_help()
                    elif command == "p":
                        force_print = True
                    elif command in {"close", "open"}:
                        active_gripper_action = 1.0 if command == "close" else -1.0
                        try:
                            obs, gripper_result = run_interactive_gripper_command(
                                env,
                                obs,
                                gripper_action=active_gripper_action,
                                steps=int(args.interactive_gripper_steps),
                            )
                            print(
                                f"gripper_{command}="
                                f"{json.dumps(jsonable(gripper_result), sort_keys=True)}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                f"gripper_{command}_error={type(exc).__name__}: {exc}",
                                flush=True,
                            )
                    elif command == "state":
                        try:
                            print(
                                "interactive_state="
                                f"{json.dumps(jsonable(interactive_state(env, obs)), sort_keys=True)}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                f"interactive_state_error={type(exc).__name__}: {exc}",
                                flush=True,
                            )
                    elif command in {"test", "move"}:
                        try:
                            obs, marker_result = plan_or_move_gripper_to_marker(
                                env,
                                obs,
                                args,
                                pos=pos,
                                rot=rot,
                                execute=(command == "move"),
                                gripper_action=active_gripper_action,
                            )
                            print(
                                f"marker_{command}="
                                f"{json.dumps(jsonable(marker_result), sort_keys=True)}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                f"marker_{command}_error={type(exc).__name__}: {exc}",
                                flush=True,
                            )
                    elif command == "save":
                        should_save = True
                        should_exit = True
                    else:
                        try:
                            command_applied = apply_command(pos, rot, command_tokens, args)
                        except ValueError as exc:
                            print(f"invalid command {command_tokens!r}: {exc}", flush=True)
                            command_applied = False
                        if command_applied:
                            changed = True
                        else:
                            print(f"unknown command {command!r}; type h for help", flush=True)

                line = summary_line(anchor_world, pos, rot)
                if changed or force_print or line != last_line:
                    print(line, flush=True)
                    last_line = line

                render_with_marker(env, args, anchor_world=anchor_world, pos=pos, rot=rot)

                if should_save:
                    current_obs = env._get_observations(force_update=True)
                    payload = pose_payload(
                        env,
                        current_obs,
                        args,
                        anchor_world=anchor_world,
                        anchor_meta=anchor_meta,
                        initial_base=initial_base,
                        initial_torso=initial_torso,
                        pos=pos,
                        rot=rot,
                    )
                    write_pose(Path(args.output), payload)
                    return {
                        "saved": True,
                        "output": str(args.output),
                        "config_kind": payload.get("config_kind"),
                        "pose": payload.get("pose"),
                        "tcp_pose_object": (payload.get("tcp_pose_object") or None),
                    }
                if should_exit:
                    return {"saved": False, "reason": "quit_without_save"}
                if args.max_iterations is not None and iterations >= int(args.max_iterations):
                    return {"saved": False, "reason": "max_iterations_reached"}
                if not args.no_render:
                    mj_viewer = getattr(getattr(env, "viewer", None), "viewer", None)
                    if (
                        mj_viewer is not None
                        and hasattr(mj_viewer, "is_running")
                        and not mj_viewer.is_running()
                    ):
                        return {"saved": False, "reason": "viewer_closed"}
                time.sleep(0.02)
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = run(args)
    except KeyboardInterrupt:
        print("\ninterrupted; no pose write was performed unless you already typed save", file=sys.stderr)
        return 130
    print(json.dumps(jsonable(result), indent=2, sort_keys=True))
    return 0 if bool(result.get("saved", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
