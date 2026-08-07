"""Teleoperate and record fixed base + torso station poses for navigation."""

from __future__ import annotations

import argparse
import contextlib
import json
import select
import sys
import termios
import time
import tty
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_STATIONS = (
    "water_button_station,cup_pick_station,sink_fill_station,service_area_station"
)
TM03_DEFAULT_STATIONS = (
    "toaster_oven_turn_clear_candidate,toaster_oven_preapproach_candidate,"
    "toaster_oven_station_candidate"
)
TM04_DEFAULT_STATIONS = (
    "water_button_station,cabinet_door_station,can_retrieval_station,"
    "can_place_station,sugar_station,saucepan_station,completion_button_station"
)
DEFAULT_SUMMARY_FIXTURES = "counter,dining_table,sink,stool"
TORSO_ACTION_DELTA_METERS = 0.05
TASK_LAYOUT_STYLE_DEFAULTS = {
    "TM01": (39, 24),
    "TM02": (15, 34),
    "TM03": (39, 24),
    "TM04": (15, 34),
    "WA-02": (15, 34),
    "WR-01": (24, 24),
}
TASK_ENV_DEFAULTS = {
    "TM01": "TM01BoilWaterAndInspectTwoDrawers",
    "TM03": "TM03BoilWaterInspectDrawersAndRunToasterOven",
    "TM04": "TM04FillPotRetrieveCanAndAddTwoSugars",
    "WA-02": "WA02ServeHotColdWaterCups",
}
TASK_STATION_DEFAULTS = {
    "TM03": TM03_DEFAULT_STATIONS,
    "TM04": TM04_DEFAULT_STATIONS,
    "WA-02": DEFAULT_STATIONS,
}
TASK_OUTPUT_DEFAULTS = {
    "TM03": (
        ROOT
        / "outputs"
        / "tm03_calibration"
        / "toaster_oven_navigation_candidates.json"
    ),
    "TM04": ROOT / "outputs" / "tm04_calibration" / "station_candidates.json",
    "WA-02": (
        ROOT
        / "bench"
        / "tasks"
        / "wa02"
        / "resources"
        / "station"
        / "stations.json"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="WA-02")
    parser.add_argument(
        "--env-name",
        default=None,
        help="Defaults to the registered environment associated with --task.",
    )
    parser.add_argument(
        "--stations",
        default=None,
        help="Comma-separated station ids to record; number keys select them in order.",
    )
    parser.add_argument("--layout-id", type=int, default=None)
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="pretrain", choices=["pretrain", "target", "all"])
    parser.add_argument(
        "--target-sugar-count",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Optional TS01 target count used to reproduce the recording scene.",
    )
    parser.add_argument("--samples-per-station", type=int, default=5)
    parser.add_argument("--linear-speed", type=float, default=0.45)
    parser.add_argument("--angular-speed", type=float, default=0.65)
    parser.add_argument(
        "--torso-step",
        type=float,
        default=0.025,
        help="Default relative torso-height increment for up/down commands, in meters.",
    )
    parser.add_argument(
        "--torso-tolerance",
        type=float,
        default=0.002,
        help="Torso target tolerance in meters.",
    )
    parser.add_argument(
        "--torso-max-steps",
        type=int,
        default=120,
        help="Maximum simulation steps used to reach one requested torso height.",
    )
    parser.add_argument(
        "--torso-limit-margin",
        type=float,
        default=0.005,
        help="Margin retained inside the torso joint limits, in meters.",
    )
    parser.add_argument(
        "--start-torso-height",
        type=float,
        default=None,
        help="Optional initial torso height in meters before recording.",
    )
    parser.add_argument("--step-hz", type=float, default=20.0)
    parser.add_argument("--input-mode", choices=["line", "raw"], default="line")
    parser.add_argument("--command-duration", type=float, default=0.5)
    parser.add_argument(
        "--render-camera",
        default="none",
        help="Viewer camera. Use 'none' for RoboCasa's layout free camera; use a camera name for a fixed camera.",
    )
    parser.add_argument("--renderer", default="mjviewer")
    parser.add_argument("--viewer-preset", choices=["topdown", "layout"], default="topdown")
    parser.add_argument("--viewer-distance-scale", type=float, default=1.4)
    parser.add_argument("--viewer-elevation", type=float, default=-75.0)
    parser.add_argument("--viewer-azimuth", type=float, default=90.0)
    parser.add_argument("--no-start-base-override", action="store_true")
    parser.add_argument("--start-base-x", type=float, default=None)
    parser.add_argument("--start-base-y", type=float, default=None)
    parser.add_argument("--start-base-yaw", type=float, default=None)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--summary-fixtures",
        default=DEFAULT_SUMMARY_FIXTURES,
        help="Comma-separated fixture attributes to print if present.",
    )
    parser.add_argument(
        "--required-fixtures",
        default="",
        help="Comma-separated fixture attributes that must be present.",
    )
    parser.add_argument(
        "--required-objects",
        default="",
        help="Comma-separated object names that must be present.",
    )
    parser.add_argument(
        "--record-current-station",
        default=None,
        help="Record the reset/current base pose once for this station and exit.",
    )
    parser.add_argument(
        "--wr01-door-open-fraction",
        type=float,
        default=None,
        help=(
            "Optional WR-01 calibration-only reset opening for both task-owned "
            "sliding cabinets. Use 0.82 to match production storage setup."
        ),
    )
    parser.add_argument("--preview-storage-fixture", default=None)
    parser.add_argument("--tabletop-storage-world-xyz", default=None)
    parser.add_argument("--tabletop-storage-world-yaw-deg", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)
    task_key = normalize_task_key(args.task)
    args.env_name = args.env_name or TASK_ENV_DEFAULTS.get(
        task_key, "WA02ServeHotColdWaterCups"
    )
    args.stations = args.stations or TASK_STATION_DEFAULTS.get(
        task_key, DEFAULT_STATIONS
    )
    args.output = args.output or TASK_OUTPUT_DEFAULTS.get(
        task_key,
        ROOT
        / "bench"
        / "tasks"
        / "wa02"
        / "resources"
        / "station"
        / "stations.json",
    )
    default_layout_id, default_style_id = TASK_LAYOUT_STYLE_DEFAULTS.get(
        task_key,
        (38, 38),
    )
    if args.layout_id is None:
        args.layout_id = default_layout_id
    if args.style_id is None:
        args.style_id = default_style_id
    args.station_ids = parse_csv(args.stations)
    if not args.station_ids:
        raise ValueError("--stations must contain at least one station id")
    if args.record_current_station is not None and args.record_current_station not in args.station_ids:
        raise ValueError(
            f"--record-current-station {args.record_current_station!r} is not in --stations {args.station_ids}"
        )
    if args.torso_step <= 0.0:
        raise ValueError("--torso-step must be positive")
    if args.torso_tolerance <= 0.0:
        raise ValueError("--torso-tolerance must be positive")
    if args.torso_max_steps <= 0:
        raise ValueError("--torso-max-steps must be positive")
    if args.torso_limit_margin < 0.0:
        raise ValueError("--torso-limit-margin must be non-negative")
    return args


def normalize_task_key(value: Any) -> str:
    raw = str(value).strip().upper().replace("_", "-")
    aliases = {
        "TM-01": "TM01",
        "TM-03": "TM03",
        "TM-04": "TM04",
        "WA02": "WA-02",
    }
    return aliases.get(raw, raw)


def parse_csv(value: Any) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in str(value or "").replace(";", ",").split(",")
        if item.strip()
    )


def parse_xyz(value: Any) -> tuple[float, float, float] | None:
    if value is None or str(value).strip() == "":
        return None
    parts = [float(item.strip()) for item in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError("xyz value must contain exactly three comma-separated floats")
    return (parts[0], parts[1], parts[2])


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def empty_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "task": args.task,
        "env_name": args.env_name,
        "layout_id": int(args.layout_id),
        "style_id": int(args.style_id),
        "seed": int(args.seed),
        "split": args.split,
        "samples_per_station": int(args.samples_per_station),
        "torso_height_units": "m",
        "stations": {station: {"samples": []} for station in args.station_ids},
    }


def load_payload(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.append and path.exists():
        payload = json.loads(path.read_text())
        payload["schema_version"] = max(int(payload.get("schema_version", 1)), 2)
        payload.setdefault("torso_height_units", "m")
        for station in args.station_ids:
            payload.setdefault("stations", {}).setdefault(station, {"samples": []})
        return payload
    return empty_payload(args)


def current_torso_height(env: Any, torso: Any | None) -> float | None:
    if torso is None:
        return None
    qpos = np.array(env.sim.data.qpos[torso.qpos_index], dtype=float).reshape(-1)
    if qpos.size < 1 or not np.isfinite(qpos[0]):
        raise RuntimeError("torso controller does not expose a finite qpos")
    return float(qpos[0])


def torso_height_limits(
    env: Any,
    torso: Any,
    *,
    margin: float = 0.0,
) -> tuple[float, float]:
    joint_indexes = np.array(torso.joint_index, dtype=int).reshape(-1)
    if joint_indexes.size < 1:
        raise RuntimeError("torso controller does not expose a joint index")
    raw_limits = np.array(
        env.sim.model.jnt_range[int(joint_indexes[0])],
        dtype=float,
    ).reshape(-1)
    if raw_limits.size < 2 or not np.all(np.isfinite(raw_limits[:2])):
        raise RuntimeError("torso joint does not expose finite joint limits")
    lower = float(raw_limits[0]) + float(margin)
    upper = float(raw_limits[1]) - float(margin)
    if lower > upper:
        raise ValueError(
            f"torso limit margin {margin:.4f} leaves no valid range "
            f"inside [{raw_limits[0]:.4f}, {raw_limits[1]:.4f}]"
        )
    return lower, upper


def clipped_torso_target(
    env: Any,
    torso: Any,
    requested_height: float,
    *,
    margin: float,
) -> tuple[float, bool]:
    if not np.isfinite(requested_height):
        raise ValueError("torso height must be finite")
    lower, upper = torso_height_limits(env, torso, margin=margin)
    target = float(np.clip(float(requested_height), lower, upper))
    return target, not np.isclose(target, float(requested_height), atol=1e-12)


def current_pose_sample(
    env,
    obs: dict[str, Any],
    station_id: str,
    sample_index: int,
    *,
    torso: Any | None = None,
) -> dict[str, Any]:
    from bench.execution.navigation.base_motion import base_yaw

    base_pos = np.array(obs["robot0_base_pos"], dtype=float)
    base_quat = np.array(obs["robot0_base_quat"], dtype=float)
    sample = {
        "station_id": station_id,
        "sample_index": int(sample_index),
        "base_pos": base_pos,
        "base_xy": base_pos[:2],
        "base_quat_xyzw": base_quat,
        "base_yaw": float(base_yaw(obs)),
        "sim_time": float(env.sim.data.time),
        "recorded_at_unix": float(time.time()),
    }
    torso_height = current_torso_height(env, torso)
    if torso_height is not None:
        sample["torso_height"] = torso_height
    return jsonable(sample)


class RawTerminal:
    def __enter__(self):
        self.tty_file = open("/dev/tty", "r", buffering=1)
        self.fd = self.tty_file.fileno()
        self.previous_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous_settings)
        self.tty_file.close()


class LineTerminal:
    def __enter__(self):
        self.tty_file = open("/dev/tty", "r", buffering=1)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.tty_file.close()


def read_key(tty_file, timeout: float) -> str | None:
    readable, _, _ = select.select([tty_file], [], [], timeout)
    if not readable:
        return None
    return tty_file.read(1)


def station_key_help(stations: tuple[str, ...]) -> str:
    return " / ".join(f"{index + 1}: select {station}" for index, station in enumerate(stations[:9]))


def print_help(input_mode: str, stations: tuple[str, ...]) -> None:
    station_help = station_key_help(stations)
    if input_mode == "line":
        print(
            "\nLine-command controls:\n"
            "  Type commands in the terminal that launched this script, then press Enter.\n"
            "  f [sec] / b [sec]: forward/backward, default sec is --command-duration\n"
            "  left [sec] / right [sec]: strafe left/right\n"
            "  tl [sec] / tr [sec]: rotate left/right\n"
            "  up [m] / down [m]: raise/lower torso, default m is --torso-step\n"
            "  z <m>: move torso to an absolute height in meters\n"
            f"  {station_help}\n"
            "  save: save current base pose and torso height for selected station\n"
            "  undo: undo last sample for selected station\n"
            "  p: print current base pose, torso height, and counts\n"
            "  h: show this help\n"
            "  done or blank line: write JSON and exit after both stations have enough samples\n"
            "  quit: exit without final write\n",
            flush=True,
        )
        return
    print(
        "\nRaw-key controls:\n"
        "  w/s or i/k: forward/backward base command\n"
        "  a/d or j/l: left/right strafe base command\n"
        "  q/e or z/c: rotate left/right\n"
        "  r/f: raise/lower torso by --torso-step\n"
        "  x: zero command\n"
        f"  {station_help}\n"
        "  space: save current base pose and torso height for selected station\n"
        "  u: undo last sample for selected station\n"
        "  p: print current base pose and counts\n"
        "  h: show this help\n"
        "  enter: write JSON and exit after both stations have enough samples\n"
        "  Ctrl-C: exit without final write\n",
        flush=True,
    )


def status_line(
    env: Any,
    obs: dict[str, Any],
    payload: dict[str, Any],
    station_id: str,
    *,
    torso: Any | None = None,
) -> str:
    from bench.execution.navigation.base_motion import base_yaw

    xy = np.array(obs["robot0_base_pos"][:2], dtype=float)
    torso_height = current_torso_height(env, torso)
    torso_status = "unavailable" if torso_height is None else f"{torso_height:.3f}m"
    counts = {
        station: len(payload["stations"][station]["samples"])
        for station in payload["stations"]
    }
    return (
        f"station={station_id} xy=({xy[0]:+.3f},{xy[1]:+.3f}) "
        f"yaw={base_yaw(obs):+.3f} torso_z={torso_status} counts={counts}"
    )


def all_done(payload: dict[str, Any], station_ids: tuple[str, ...], samples_per_station: int) -> bool:
    return all(
        len(payload["stations"][station]["samples"]) >= samples_per_station
        for station in station_ids
    )


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def safe_name(value: Any) -> str | None:
    for attr in ("name", "naming_prefix"):
        result = getattr(value, attr, None)
        if result is not None:
            return str(result)
    return None


def safe_position(value: Any) -> str:
    try:
        arr = np.array(value, dtype=float).reshape(-1)
    except Exception:
        return "pos=?"
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return "pos=?"
    return f"pos=({arr[0]:+.3f},{arr[1]:+.3f},{arr[2]:+.3f})" if arr.size >= 3 else f"pos=({arr[0]:+.3f},{arr[1]:+.3f})"


def object_position(env, object_name: str) -> np.ndarray | None:
    obj_body_id = getattr(env, "obj_body_id", {}) or {}
    if object_name not in obj_body_id:
        return None
    return np.array(env.sim.data.body_xpos[obj_body_id[object_name]], dtype=float)


def viewer_render_camera(args: argparse.Namespace) -> str | None:
    value = str(args.render_camera).strip()
    if value.lower() in {"", "none", "free", "layout"}:
        return None
    return value


def finite_xyz(value: Any) -> np.ndarray | None:
    try:
        arr = np.array(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    if np.max(np.abs(arr[:3])) > 100.0:
        return None
    return arr[:3]


def viewer_camera_points(env, obs: dict[str, Any]) -> np.ndarray:
    points: list[np.ndarray] = []
    for value in (obs.get("robot0_base_pos"),):
        point = finite_xyz(value)
        if point is not None:
            points.append(point)
    for object_name in sorted(getattr(env, "objects", {}) or {}):
        point = object_position(env, object_name)
        if point is not None:
            points.append(point)
    for fixture in (getattr(env, "fixtures", {}) or {}).values():
        point = finite_xyz(getattr(fixture, "pos", None))
        if point is not None:
            points.append(point)
        try:
            for ext_point in fixture.get_ext_sites(relative=False):
                point = finite_xyz(ext_point)
                if point is not None:
                    points.append(point)
        except Exception:
            pass
    if not points:
        return np.array([[1.0, -4.5, 0.0], [3.0, -6.5, 1.0]], dtype=float)
    return np.array(points, dtype=float)


def topdown_viewer_config(env, obs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    points = viewer_camera_points(env, obs)
    xy_min = np.min(points[:, :2], axis=0)
    xy_max = np.max(points[:, :2], axis=0)
    xy_span = float(np.max(xy_max - xy_min))
    z_center = float(np.clip(np.percentile(points[:, 2], 55), 0.7, 1.4))
    distance = max(7.0, xy_span * float(args.viewer_distance_scale))
    return {
        "lookat": [float((xy_min[0] + xy_max[0]) / 2.0), float((xy_min[1] + xy_max[1]) / 2.0), z_center],
        "distance": float(distance),
        "azimuth": float(args.viewer_azimuth),
        "elevation": float(args.viewer_elevation),
    }


def apply_viewer_preset(env, obs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    if args.viewer_preset != "topdown" or viewer_render_camera(args) is not None:
        return None
    config = topdown_viewer_config(env, obs, args)
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


def print_scene_summary(env, args: argparse.Namespace) -> None:
    from robosuite.environments.base import REGISTERED_ENVS

    registered = args.env_name in REGISTERED_ENVS
    print(
        f"env={args.env_name} class={type(env).__name__} registered={registered} "
        f"layout={args.layout_id} style={args.style_id} seed={args.seed}",
        flush=True,
    )
    if not registered:
        available = ", ".join(sorted(REGISTERED_ENVS))
        raise RuntimeError(f"env {args.env_name!r} is not registered; available envs: {available}")

    camera_names = list(getattr(env.sim.model, "camera_names", []))
    render_camera = viewer_render_camera(args)
    render_camera_label = "none (RoboCasa layout free camera)" if render_camera is None else render_camera
    print(
        f"render_camera={render_camera_label} viewer_preset={args.viewer_preset} "
        f"available_cameras={camera_names}",
        flush=True,
    )
    if render_camera is not None and render_camera not in camera_names:
        raise RuntimeError(
            f"render camera {render_camera!r} not found; available cameras: {camera_names}"
        )

    for label in parse_csv(args.summary_fixtures):
        fixture = getattr(env, label, None)
        print(
            f"fixture {label}: class={type(fixture).__name__ if fixture is not None else None} "
            f"name={safe_name(fixture)} {safe_position(getattr(fixture, 'pos', None))}",
            flush=True,
        )

    for object_name in sorted(getattr(env, "objects", {}) or {}):
        pos = object_position(env, object_name)
        print(f"object {object_name}: {safe_position(pos)}", flush=True)

    required_fixtures = parse_csv(args.required_fixtures)
    missing_fixtures = [name for name in required_fixtures if getattr(env, name, None) is None]
    required_objects = parse_csv(args.required_objects)
    object_names = set((getattr(env, "objects", {}) or {}).keys())
    missing_objects = [name for name in required_objects if name not in object_names]
    if missing_fixtures or missing_objects:
        raise RuntimeError(
            "scene is incomplete: "
            f"missing_fixtures={missing_fixtures} missing_objects={missing_objects}"
        )


def render_interactive(env) -> None:
    viewer = getattr(env, "viewer", None)
    if viewer is not None and hasattr(viewer, "update"):
        viewer.update()
    else:
        env.render()


def set_command(command: np.ndarray, *, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
    command[:] = 0.0
    if command.size >= 1:
        command[0] = float(x)
    if command.size >= 2:
        command[1] = float(y)
    if command.size >= 3:
        command[2] = float(yaw)


def teleop_translation_world_vector(
    base_yaw_rad: float,
    *,
    forward: float = 0.0,
    left: float = 0.0,
) -> np.ndarray:
    yaw = float(base_yaw_rad)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [
            float(forward) * c - float(left) * s,
            float(forward) * s + float(left) * c,
        ],
        dtype=float,
    )


def set_current_yaw_translation_command(
    command: np.ndarray,
    *,
    env: Any,
    obs: dict[str, Any],
    forward: float = 0.0,
    left: float = 0.0,
    speed: float,
    base_yaw_fn: Any,
    world_xy_to_controller_action_fn: Any,
) -> None:
    world_vec = teleop_translation_world_vector(
        base_yaw_fn(obs),
        forward=forward,
        left=left,
    )
    raw_xy = world_xy_to_controller_action_fn(env, world_vec * float(speed))
    set_command(command, x=float(raw_xy[0]), y=float(raw_xy[1]))


def command_steps(tokens: list[str], args: argparse.Namespace) -> int:
    duration = float(args.command_duration)
    if len(tokens) >= 2:
        duration = float(tokens[1])
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    return max(1, int(round(duration * max(float(args.step_hz), 1e-6))))


def torso_delta(tokens: list[str], args: argparse.Namespace) -> float:
    if len(tokens) > 2:
        raise ValueError("up/down accepts at most one height increment")
    delta = float(args.torso_step) if len(tokens) < 2 else float(tokens[1])
    if delta <= 0.0 or not np.isfinite(delta):
        raise ValueError("torso height increment must be a positive finite number")
    return delta


def start_base_override(args: argparse.Namespace) -> tuple[np.ndarray, float] | None:
    if args.no_start_base_override:
        return None
    if args.start_base_x is not None or args.start_base_y is not None or args.start_base_yaw is not None:
        if args.start_base_x is None or args.start_base_y is None:
            raise ValueError("--start-base-x and --start-base-y must be provided together")
        yaw = np.pi / 2 if args.start_base_yaw is None else float(args.start_base_yaw)
        return np.array([float(args.start_base_x), float(args.start_base_y)], dtype=float), yaw
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    from bench.env.robocasa_runtime import (
        bootstrap_local_robocasa_sources,
        make_joint_position_controller_config,
        make_robocasa_env,
        qpos_for_controller,
        reset_env_with_initial_arm_qpos,
    )
    from bench.execution.navigation.base_motion import (
        base_yaw,
        set_base_pose_kinematic,
        step_base_and_hold_arm,
        world_xy_to_controller_action,
    )

    payload = load_payload(args.output, args)
    env_kwargs = {"render_camera": viewer_render_camera(args), "renderer": str(args.renderer)}
    if (
        str(args.task).strip().upper() == "WA-02"
        or str(args.env_name) == "WA02ServeHotColdWaterCups"
    ):
        from bench.tasks.wa02.config import load_task_config
        from bench.tasks.wa02.domain.geometry import (
            resolve_cup_region_constraints,
            resolve_cup_sampling_regions,
        )

        wa02_config = load_task_config()
        objects = dict(wa02_config.initial_state.objects)
        cup_regions = resolve_cup_sampling_regions(dict(objects.get("cups") or {}))
        constraints = resolve_cup_region_constraints(
            dict(objects.get("cup_region_constraints") or {})
        )
        water_button_positions = wa02_config.path(
            wa02_config.resources.get("water_button_positions")
        )
        env_kwargs.update(
            {
                "water_button_positions_file": (
                    None
                    if water_button_positions is None
                    else str(water_button_positions)
                ),
                "service_threshold": float(
                    wa02_config.semantics.get("service_threshold", 0.12)
                ),
                "cup_upright_threshold_deg": float(
                    wa02_config.semantics.get("cup_upright_threshold_deg", 15.0)
                ),
                "cup_placement_seed": int(args.seed),
                "cup_sampling_regions": {
                    cup_id: {
                        "region_center_xy": list(region.center_xy),
                        "region_size_xy": list(region.size_xy),
                        "yaw_rad": float(region.yaw_rad),
                    }
                    for cup_id, region in cup_regions.items()
                },
                "cup_region_constraints": {
                    "sink_exterior_local_x_max": float(
                        constraints.sink_exterior_local_x_max
                    ),
                    "minimum_sink_clearance": float(
                        constraints.minimum_sink_clearance
                    ),
                    "minimum_inter_region_gap": float(
                        constraints.minimum_inter_region_gap
                    ),
                },
            }
        )
    if (
        str(args.task).strip().upper() == "TM01"
        or str(args.env_name) == "TM01BoilWaterAndInspectTwoDrawers"
    ):
        bootstrap_local_robocasa_sources()
        from bench.tasks.tm01.config import (
            load_task_config,
            resolve_observer_config,
        )
        from bench.tasks.tm01.domain.episode import resolve_scene_config
        from bench.tasks.tm01.environment import (
            TM01BoilWaterAndInspectTwoDrawers,
        )

        del TM01BoilWaterAndInspectTwoDrawers
        tm01_config = load_task_config()
        tm01_scene = resolve_scene_config(
            env_split=str(args.split),
            **dict(tm01_config.scene),
        )
        if int(args.layout_id) != int(tm01_scene.layout_id):
            raise ValueError(
                f"TM01 calibration requires layout {tm01_scene.layout_id}, "
                f"got {args.layout_id}"
            )
        if int(args.style_id) != int(tm01_scene.style_id):
            raise ValueError(
                f"TM01 calibration requires style {tm01_scene.style_id}, "
                f"got {args.style_id}"
            )
        env_kwargs.update(
            {
                "drawer_names": tm01_scene.drawer_names,
                "drawer_world_xyzs": tm01_scene.drawer_world_xyzs,
                "drawer_world_yaw_deg": tm01_scene.drawer_world_yaw_deg,
                "completion_button_world_xyzs": dict(
                    tm01_scene.completion_button_world_xyzs
                ),
                "expected_stove_asset": tm01_scene.expected_stove_asset,
                "pot_name": tm01_scene.pot_name,
                "pot_xml": tm01_scene.pot_xml,
                "observer_config": asdict(resolve_observer_config(tm01_config)),
            }
        )
    if (
        str(args.task).strip().upper() == "TM03"
        or str(args.env_name)
        == "TM03BoilWaterInspectDrawersAndRunToasterOven"
    ):
        bootstrap_local_robocasa_sources()
        from bench.tasks.tm03.config import (
            load_task_config,
            resolve_observer_config,
        )
        from bench.tasks.tm03.domain.episode import resolve_scene_config
        from bench.tasks.tm03.environment import (
            TM03BoilWaterInspectDrawersAndRunToasterOven,
        )

        del TM03BoilWaterInspectDrawersAndRunToasterOven
        tm03_config = load_task_config()
        tm03_scene = resolve_scene_config(
            env_split=str(args.split),
            **dict(tm03_config.scene),
        )
        if int(args.layout_id) != int(tm03_scene.layout_id):
            raise ValueError(
                f"TM03 calibration requires layout {tm03_scene.layout_id}, "
                f"got {args.layout_id}"
            )
        if int(args.style_id) != int(tm03_scene.style_id):
            raise ValueError(
                f"TM03 calibration requires style {tm03_scene.style_id}, "
                f"got {args.style_id}"
            )
        env_kwargs.update(
            {
                "drawer_names": tm03_scene.drawer_names,
                "drawer_world_xyzs": tm03_scene.drawer_world_xyzs,
                "drawer_world_yaw_deg": tm03_scene.drawer_world_yaw_deg,
                "completion_button_world_xyzs": dict(
                    tm03_scene.completion_button_world_xyzs
                ),
                "expected_stove_asset": tm03_scene.expected_stove_asset,
                "pot_name": tm03_scene.pot_name,
                "pot_xml": tm03_scene.pot_xml,
                "expected_toaster_oven_asset": (
                    tm03_scene.expected_toaster_oven_asset
                ),
                "toaster_oven_food_name": tm03_scene.toaster_oven_food_name,
                "toaster_oven_food_xml": tm03_scene.toaster_oven_food_xml,
                "toaster_oven_time_joint": tm03_scene.toaster_oven_time_joint,
                "toaster_oven_time_collision_geom": (
                    tm03_scene.toaster_oven_time_collision_geom
                ),
                "observer_config": asdict(resolve_observer_config(tm03_config)),
            }
        )
    if (
        str(args.task).strip().upper() == "TM04"
        or str(args.env_name) == "TM04FillPotRetrieveCanAndAddTwoSugars"
    ):
        bootstrap_local_robocasa_sources()
        from bench.tasks.tm04.config import (
            load_task_config,
            resolve_observer_config,
        )
        from bench.tasks.tm04.domain.episode import (
            resolve_episode_plan,
            resolve_scene_config,
        )
        from bench.tasks.tm04.environment import (
            TM04FillPotRetrieveCanAndAddTwoSugars,
        )

        del TM04FillPotRetrieveCanAndAddTwoSugars
        tm04_config = load_task_config()
        tm04_scene = resolve_scene_config(
            env_split=str(args.split),
            **dict(tm04_config.scene),
        )
        tm04_plan = resolve_episode_plan(
            seed=int(args.seed),
            **dict(tm04_config.episode),
        )
        if int(args.layout_id) != int(tm04_scene.layout_id):
            raise ValueError(
                f"TM04 calibration requires layout {tm04_scene.layout_id}, "
                f"got {args.layout_id}"
            )
        if int(args.style_id) != int(tm04_scene.style_id):
            raise ValueError(
                f"TM04 calibration requires style {tm04_scene.style_id}, "
                f"got {args.style_id}"
            )
        env_kwargs.update(
            {
                "water_button_world_xyz": tm04_scene.water_button_world_xyz,
                "completion_button_world_xyz": (
                    tm04_scene.completion_button_world_xyz
                ),
                "cabinet_world_xyz": tm04_scene.cabinet_world_xyz,
                "cabinet_yaw_deg": tm04_scene.cabinet_yaw_deg,
                "can_place_region_world_xyz": (
                    tm04_scene.can_place_region_world_xyz
                ),
                "can_place_region_size_xy": tm04_scene.can_place_region_size_xy,
                "water_pot_name": tm04_scene.water_pot_name,
                "can_name": tm04_scene.can_name,
                "sugar_plate_name": tm04_scene.sugar_plate_name,
                "hot_chocolate_name": tm04_scene.hot_chocolate_name,
                "sugar_names": tm04_scene.sugar_names,
                "sugar_plate_world_xyz": tm04_scene.sugar_plate_world_xyz,
                "hot_chocolate_world_xyz": tm04_scene.hot_chocolate_world_xyz,
                "target_sugar_count": tm04_plan.target_sugar_count,
                "total_sugar_count": tm04_plan.total_sugar_count,
                "selected_sugar_names": tm04_plan.selected_sugar_names,
                "observer_config": asdict(resolve_observer_config(tm04_config)),
            }
        )
    wr02_setup = None
    if (
        str(args.task).strip().upper() == "WR-02"
        or str(args.env_name) == "WR02ReturnCondimentPair"
    ):
        from bench.tasks.wr02.config import load_task_config
        from bench.tasks.wr02.domain.episode import resolve_episode_plan

        wr02_config = load_task_config()
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
                "home_threshold": float(wr02_semantics.get("home_threshold", 0.16)),
                "use_station_threshold": float(
                    wr02_semantics.get("use_station_threshold", 0.20)
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
    if args.target_sugar_count is not None:
        env_kwargs["target_sugar_count"] = int(args.target_sugar_count)
    if args.preview_storage_fixture:
        env_kwargs["preview_storage_fixture"] = str(args.preview_storage_fixture)
    tabletop_storage_world_xyz = parse_xyz(args.tabletop_storage_world_xyz)
    if tabletop_storage_world_xyz is not None:
        env_kwargs["tabletop_storage_world_xyz"] = tabletop_storage_world_xyz
    if args.tabletop_storage_world_yaw_deg is not None:
        env_kwargs["tabletop_storage_world_yaw_deg"] = float(args.tabletop_storage_world_yaw_deg)
    env = make_robocasa_env(
        env_name=args.env_name,
        split=args.split,
        seed=args.seed,
        controller_config=make_joint_position_controller_config(),
        camera_names=["robot0_agentview_right"],
        camera_widths=int(args.camera_width),
        camera_heights=int(args.camera_height),
        layout_id=args.layout_id,
        style_id=args.style_id,
        has_renderer=not args.no_render,
        has_offscreen_renderer=bool(args.no_render),
        use_camera_obs=bool(args.no_render),
        env_kwargs=env_kwargs,
    )
    try:
        obs = reset_env_with_initial_arm_qpos(env)
        if wr02_setup is not None:
            from bench.tasks.wr02.initialization import initialize_episode_state

            wr02_config, wr02_episode = wr02_setup
            obs, initialization_summary = initialize_episode_state(
                env,
                obs,
                episode=wr02_episode,
                robot_base=wr02_config.initial_state.robot_base,
                objects=dict(wr02_config.initial_state.objects),
            )
            print(
                "[calibration] applied WR-02 fixed initialization: "
                f"episode_case_id={wr02_episode.case_id} "
                f"home_assignment={dict(wr02_episode.home_assignment)} "
                f"plate_after={initialization_summary.get('plate_after')}",
                flush=True,
            )
        if args.wr01_door_open_fraction is not None:
            if str(args.task).strip().upper() != "WR-01":
                raise ValueError("--wr01-door-open-fraction is only valid for WR-01")
            fraction = float(args.wr01_door_open_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("--wr01-door-open-fraction must be within [0, 1]")
            set_fraction = getattr(env, "set_level_door_open_fraction", None)
            if not callable(set_fraction):
                raise RuntimeError(
                    "WR-01 environment does not expose set_level_door_open_fraction"
                )
            for level in ("lower", "upper"):
                set_fraction(level, fraction)
            env.sim.forward()
            obs = env._get_observations(force_update=True)
            print(
                "[calibration] WR-01 cabinet doors set to "
                f"open_fraction={fraction:.3f}"
            )
        start_override = start_base_override(args)
        if start_override is not None:
            start_xy, start_yaw = start_override
            obs = set_base_pose_kinematic(env, target_xy=start_xy, target_yaw=start_yaw)
            print(
                f"start_base_override xy=({start_xy[0]:+.3f},{start_xy[1]:+.3f}) yaw={start_yaw:+.3f}",
                flush=True,
            )
        viewer_config = apply_viewer_preset(env, obs, args)
        print_scene_summary(env, args)
        if viewer_config is not None:
            print(f"viewer_config={json.dumps(jsonable(viewer_config), sort_keys=True)}", flush=True)
        robot = env.robots[0]
        part_controllers = robot.composite_controller.part_controllers
        right = part_controllers["right"]
        torso = part_controllers.get("torso")
        start, end = robot.composite_controller._action_split_indexes["base"]
        command = np.zeros(end - start, dtype=float)
        translation_intent: tuple[float, float] | None = None

        station_id = args.station_ids[0]
        dt = 1.0 / max(float(args.step_hz), 1e-6)
        last_status = 0.0
        torso_target = current_torso_height(env, torso)
        if torso is None:
            print(
                "[calibration] torso controller unavailable; only base pose will be recorded",
                flush=True,
            )
        else:
            lower, upper = torso_height_limits(
                env,
                torso,
                margin=float(args.torso_limit_margin),
            )
            print(
                f"[calibration] torso_z={torso_target:.3f}m "
                f"safe_range=[{lower:.3f},{upper:.3f}]m",
                flush=True,
            )

        def clear_base_command() -> None:
            nonlocal translation_intent
            translation_intent = None
            command[:] = 0.0

        def set_translation_intent(*, forward: float = 0.0, left: float = 0.0) -> None:
            nonlocal translation_intent
            translation_intent = (float(forward), float(left))
            command[:] = 0.0

        def set_yaw_command(yaw: float) -> None:
            nonlocal translation_intent
            translation_intent = None
            set_command(command, yaw=yaw)

        def refresh_translation_command() -> None:
            if translation_intent is None:
                return
            forward, left = translation_intent
            set_current_yaw_translation_command(
                command,
                env=env,
                obs=obs,
                forward=forward,
                left=left,
                speed=float(args.linear_speed),
                base_yaw_fn=base_yaw,
                world_xy_to_controller_action_fn=world_xy_to_controller_action,
            )

        def step_for(steps: int) -> None:
            nonlocal obs
            for _ in range(steps):
                refresh_translation_command()
                torso_action = None
                if torso is not None and torso_target is not None:
                    height_error = float(torso_target) - float(
                        current_torso_height(env, torso)
                    )
                    torso_action = (
                        0.0
                        if abs(height_error) <= float(args.torso_tolerance)
                        else float(
                            np.clip(
                                height_error / TORSO_ACTION_DELTA_METERS,
                                -1.0,
                                1.0,
                            )
                        )
                    )
                obs = step_base_and_hold_arm(
                    env,
                    arm_qpos=qpos_for_controller(env, right),
                    base_action=command,
                    base_mode=1.0,
                    gripper_action=0.0,
                    torso_action=torso_action,
                )
                if not args.no_render:
                    render_interactive(env)
            clear_base_command()

        def move_torso_to(requested_height: float) -> float:
            nonlocal torso_target
            if torso is None:
                raise ValueError("this robot does not expose a torso controller")
            target, was_clipped = clipped_torso_target(
                env,
                torso,
                requested_height,
                margin=float(args.torso_limit_margin),
            )
            if was_clipped:
                print(
                    f"requested torso_z={requested_height:.3f}m was clipped to "
                    f"{target:.3f}m by the safe joint range",
                    flush=True,
                )
            torso_target = target
            clear_base_command()
            for _ in range(int(args.torso_max_steps)):
                actual = float(current_torso_height(env, torso))
                if abs(target - actual) <= float(args.torso_tolerance):
                    return actual
                step_for(1)
            actual = float(current_torso_height(env, torso))
            raise RuntimeError(
                f"torso failed to reach {target:.3f}m within "
                f"{args.torso_max_steps} steps; actual={actual:.3f}m"
            )

        if args.start_torso_height is not None:
            actual = move_torso_to(float(args.start_torso_height))
            print(f"start_torso_height actual={actual:.3f}m", flush=True)

        if args.record_current_station is not None:
            station_id = str(args.record_current_station)
            samples = payload["stations"][station_id]["samples"]
            samples.append(
                current_pose_sample(
                    env,
                    obs,
                    station_id,
                    len(samples),
                    torso=torso,
                )
            )
            payload["updated_at_unix"] = float(time.time())
            save_payload(args.output, payload)
            print(f"recorded current pose for {station_id} -> {args.output}", flush=True)
            return payload

        print_help(args.input_mode, args.station_ids)
        print(status_line(env, obs, payload, station_id, torso=torso), flush=True)
        if not args.no_render:
            render_interactive(env)

        if args.input_mode == "line":
            with LineTerminal() as terminal:
                while True:
                    print("calib> ", end="", flush=True)
                    line = terminal.tty_file.readline()
                    if line == "":
                        print("\ntty closed; no final JSON write was performed", flush=True)
                        return payload
                    tokens = line.strip().split()
                    key = tokens[0].lower() if tokens else "done"
                    try:
                        if key in ("f", "forward"):
                            set_translation_intent(forward=1.0)
                            step_for(command_steps(tokens, args))
                        elif key in ("b", "back", "backward"):
                            set_translation_intent(forward=-1.0)
                            step_for(command_steps(tokens, args))
                        elif key in ("left", "l", "sl"):
                            set_translation_intent(left=1.0)
                            step_for(command_steps(tokens, args))
                        elif key in ("right", "r", "sr"):
                            set_translation_intent(left=-1.0)
                            step_for(command_steps(tokens, args))
                        elif key in ("tl", "rotl", "ccw"):
                            set_yaw_command(args.angular_speed)
                            step_for(command_steps(tokens, args))
                        elif key in ("tr", "rotr", "cw"):
                            set_yaw_command(-args.angular_speed)
                            step_for(command_steps(tokens, args))
                        elif key in ("up", "raise"):
                            height = current_torso_height(env, torso)
                            if height is None:
                                raise ValueError("this robot does not expose a torso controller")
                            move_torso_to(height + torso_delta(tokens, args))
                        elif key in ("down", "lower"):
                            height = current_torso_height(env, torso)
                            if height is None:
                                raise ValueError("this robot does not expose a torso controller")
                            move_torso_to(height - torso_delta(tokens, args))
                        elif key in ("z", "height", "torso"):
                            if len(tokens) != 2:
                                raise ValueError("z/height requires one absolute height in meters")
                            move_torso_to(float(tokens[1]))
                        elif key.isdigit() and 1 <= int(key) <= min(len(args.station_ids), 9):
                            station_id = args.station_ids[int(key) - 1]
                            print(f"selected {station_id}", flush=True)
                        elif key == "save":
                            samples = payload["stations"][station_id]["samples"]
                            samples.append(
                                current_pose_sample(
                                    env,
                                    obs,
                                    station_id,
                                    len(samples),
                                    torso=torso,
                                )
                            )
                            print(f"saved {station_id} sample {len(samples)}/{args.samples_per_station}", flush=True)
                        elif key in ("u", "undo"):
                            samples = payload["stations"][station_id]["samples"]
                            if samples:
                                removed = samples.pop()
                                print(f"removed {station_id} sample {removed.get('sample_index')}", flush=True)
                        elif key in ("p", "status"):
                            pass
                        elif key == "h":
                            print_help(args.input_mode, args.station_ids)
                        elif key in ("done", "write"):
                            if all_done(payload, args.station_ids, int(args.samples_per_station)):
                                break
                            print("not enough samples yet; use p/status to inspect counts", flush=True)
                        elif key in ("quit", "exit"):
                            print("exiting without final JSON write", flush=True)
                            return payload
                        else:
                            print(f"unknown command {key!r}; use h for help", flush=True)
                    except ValueError as exc:
                        clear_base_command()
                        print(f"invalid command: {exc}", flush=True)
                    print(
                        status_line(env, obs, payload, station_id, torso=torso),
                        flush=True,
                    )
        else:
            with RawTerminal() as terminal:
                while True:
                    key = read_key(terminal.tty_file, dt)
                    if key is not None:
                        if key in ("w", "i"):
                            set_translation_intent(forward=1.0)
                        elif key in ("s", "k"):
                            set_translation_intent(forward=-1.0)
                        elif key in ("a", "j"):
                            set_translation_intent(left=1.0)
                        elif key in ("d", "l"):
                            set_translation_intent(left=-1.0)
                        elif key in ("q", "z"):
                            set_yaw_command(args.angular_speed)
                        elif key in ("e", "c"):
                            set_yaw_command(-args.angular_speed)
                        elif key == "r":
                            height = current_torso_height(env, torso)
                            if height is None:
                                print("\ntorso controller unavailable", flush=True)
                            else:
                                move_torso_to(height + float(args.torso_step))
                        elif key == "f":
                            height = current_torso_height(env, torso)
                            if height is None:
                                print("\ntorso controller unavailable", flush=True)
                            else:
                                move_torso_to(height - float(args.torso_step))
                        elif key == "x":
                            clear_base_command()
                        elif key.isdigit() and 1 <= int(key) <= min(len(args.station_ids), 9):
                            station_id = args.station_ids[int(key) - 1]
                            clear_base_command()
                            print(f"\nselected {station_id}", flush=True)
                        elif key == " ":
                            samples = payload["stations"][station_id]["samples"]
                            samples.append(
                                current_pose_sample(
                                    env,
                                    obs,
                                    station_id,
                                    len(samples),
                                    torso=torso,
                                )
                            )
                            clear_base_command()
                            print(f"\nsaved {station_id} sample {len(samples)}/{args.samples_per_station}", flush=True)
                        elif key == "u":
                            samples = payload["stations"][station_id]["samples"]
                            if samples:
                                removed = samples.pop()
                                print(f"\nremoved {station_id} sample {removed.get('sample_index')}", flush=True)
                        elif key == "p":
                            print(
                                "\n"
                                + status_line(
                                    env,
                                    obs,
                                    payload,
                                    station_id,
                                    torso=torso,
                                ),
                                flush=True,
                            )
                        elif key == "h":
                            print_help(args.input_mode, args.station_ids)
                        elif key in ("\n", "\r"):
                            if all_done(payload, args.station_ids, int(args.samples_per_station)):
                                break
                            print("\nnot enough samples yet; press p to inspect counts", flush=True)

                    step_for(1)
                    now = time.time()
                    if now - last_status > 2.0:
                        print(
                            "\r"
                            + status_line(
                                env,
                                obs,
                                payload,
                                station_id,
                                torso=torso,
                            ),
                            end="",
                            flush=True,
                        )
                        last_status = now

        payload["updated_at_unix"] = float(time.time())
        save_payload(args.output, payload)
        print(f"\nwrote {args.output}", flush=True)
        return payload
    finally:
        env.close()


def main() -> int:
    args = parse_args()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            payload = run(args)
    except KeyboardInterrupt:
        print("\ninterrupted; no final JSON write was performed", file=sys.stderr)
        return 130
    print(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
