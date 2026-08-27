#!/usr/bin/env python3
"""Replay TS02 actions and measure mobile-base station accuracy."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from math import atan2, pi
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.suite import load_eval_suite
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


DEFAULT_SUITE = ROOT / "configs/eval/suites/ts02_smoke_v1.yaml"
DEFAULT_STATIONS = ROOT / "bench/tasks/ts02/resources/station/stations.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path, help="evaluation actions.npz")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None, help="optional trace CSV")
    return parser.parse_args()


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _seed_from_path(path: Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(path))
    return None if match is None else int(match.group(1))


def _normalize_angle(value: float) -> float:
    return float((float(value) + pi) % (2.0 * pi) - pi)


def _yaw_from_xyzw(quat: Any) -> float:
    x, y, z, w = np.asarray(quat, dtype=float).reshape(-1)[:4]
    return float(atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _load_stations(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for name, station in payload["stations"].items():
        sample = station["samples"][0]
        result[str(name)] = {
            "xy": np.asarray(sample["base_xy"], dtype=float),
            "yaw": float(sample["base_yaw"]),
        }
    return result


def _physics_only_descriptor(descriptor: Any) -> Any:
    sensors = replace(
        descriptor.sensors,
        camera_depths=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_observations=False,
    )
    return replace(descriptor, sensors=sensors)


def _make_runtime(*, seed: int, suite_path: Path) -> Any:
    suite = load_eval_suite(suite_path)
    raw_options = dict(suite.task_options)
    task_options = raw_options.get("task_options", {})
    if not isinstance(task_options, dict):
        raise ValueError("task.options.task_options must be a map")

    episode = get_task_definition("TS02").resolve(
        EpisodeRequest(
            task_id="TS02",
            seed=seed,
            variant=str(raw_options.get("variant", "memory")),
            split="evaluation",
            env_split=suite.env_split,
            options=task_options,
            task_config_file=_resolve_path(raw_options.get("task_config_file")),
            observation_profile_file=_resolve_path(
                raw_options.get("observation_profile_file")
            ),
            allow_partial_calibration=False,
        )
    )
    runtime_api = resolve_builtin_task_plugin("TS02").load_runtime_api()
    if runtime_api is None:
        raise LookupError("TS02 has no canonical runtime API")
    descriptor = _physics_only_descriptor(runtime_api.build_descriptor(episode, execute=True))
    return RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
        descriptor,
        expert=False,
    )


def _pose_row(
    *,
    step: int,
    obs: dict[str, Any],
    action: np.ndarray | None,
    stations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    xy = np.asarray(obs["robot0_base_pos"], dtype=float).reshape(-1)[:2]
    yaw = _yaw_from_xyzw(obs["robot0_base_quat"])
    row: dict[str, Any] = {
        "step": int(step),
        "base_x": float(xy[0]),
        "base_y": float(xy[1]),
        "base_yaw": yaw,
        "base_forward": None if action is None else float(action[8]),
        "base_side": None if action is None else float(action[9]),
        "base_yaw_command": None if action is None else float(action[10]),
        "base_mode": None if action is None else float(action[12]),
    }
    for name, station in stations.items():
        row[f"{name}_distance"] = float(np.linalg.norm(xy - station["xy"]))
        row[f"{name}_yaw_error"] = abs(_normalize_angle(yaw - station["yaw"]))
    return row


def _base_mode_runs(actions: np.ndarray) -> list[dict[str, Any]]:
    enabled = np.asarray(actions[:, 12] > 0.0, dtype=bool)
    starts = np.flatnonzero(np.r_[True, enabled[1:] != enabled[:-1]])
    ends = np.r_[starts[1:], len(enabled)]
    return [
        {"start": int(start), "stop": int(stop), "enabled": bool(enabled[start])}
        for start, stop in zip(starts, ends, strict=True)
    ]


def _summarize(
    *,
    seed: int,
    actions: np.ndarray,
    rows: list[dict[str, Any]],
    stations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    station_summary: dict[str, Any] = {}
    for name in stations:
        distances = np.asarray([row[f"{name}_distance"] for row in rows], dtype=float)
        yaw_errors = np.asarray([row[f"{name}_yaw_error"] for row in rows], dtype=float)
        closest = int(np.argmin(distances))
        station_summary[name] = {
            "min_distance_m": float(distances[closest]),
            "step_at_min_distance": int(rows[closest]["step"]),
            "yaw_error_at_min_distance_rad": float(yaw_errors[closest]),
            "entered_xy_2p5cm": bool(np.any(distances <= 0.025)),
            "entered_xy_5cm": bool(np.any(distances <= 0.05)),
            "entered_pose_tolerance": bool(
                np.any((distances <= 0.025) & (yaw_errors <= 0.06))
            ),
            "final_distance_m": float(distances[-1]),
            "final_yaw_error_rad": float(yaw_errors[-1]),
        }
    return {
        "seed": int(seed),
        "action_count": int(len(actions)),
        "base_mode_enabled_fraction": float(np.mean(actions[:, 12] > 0.0)),
        "base_mode_runs": _base_mode_runs(actions),
        "initial_pose": {
            "x": rows[0]["base_x"],
            "y": rows[0]["base_y"],
            "yaw": rows[0]["base_yaw"],
        },
        "final_pose": {
            "x": rows[-1]["base_x"],
            "y": rows[-1]["base_y"],
            "yaw": rows[-1]["base_yaw"],
        },
        "stations": station_summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    action_path = args.actions.expanduser().resolve()
    actions = np.asarray(np.load(action_path)["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 13 or not np.all(np.isfinite(actions)):
        raise ValueError(f"invalid action array at {action_path}: {actions.shape}")
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        actions = actions[: args.max_steps]
    seed = args.seed if args.seed is not None else _seed_from_path(action_path)
    if seed is None:
        raise ValueError("could not infer seed from path; pass --seed")
    if args.sample_every <= 0:
        raise ValueError("--sample-every must be positive")

    stations = _load_stations(args.stations.expanduser().resolve())
    runtime = _make_runtime(seed=seed, suite_path=args.suite.expanduser().resolve())
    rows = [
        _pose_row(
            step=0,
            obs=dict(runtime.observation),
            action=None,
            stations=stations,
        )
    ]
    try:
        for step, action in enumerate(actions, start=1):
            runtime.step(action)
            row = _pose_row(
                step=step,
                obs=dict(runtime.observation),
                action=action,
                stations=stations,
            )
            rows.append(row)
            if step % args.sample_every == 0 or step == len(actions):
                nearest = min(
                    stations,
                    key=lambda name: float(row[f"{name}_distance"]),
                )
                print(
                    f"step={step:04d} pose=({row['base_x']:.3f}, "
                    f"{row['base_y']:.3f}, {row['base_yaw']:.3f}) "
                    f"action=({row['base_forward']:.3f}, {row['base_side']:.3f}, "
                    f"{row['base_yaw_command']:.3f}) mode={row['base_mode']:.3f} "
                    f"nearest={nearest} distance={row[f'{nearest}_distance']:.3f} "
                    f"yaw_error={row[f'{nearest}_yaw_error']:.3f}",
                    flush=True,
                )
    finally:
        runtime.close()

    if args.output is not None:
        _write_csv(args.output.expanduser().resolve(), rows)
    print(json.dumps(_summarize(seed=seed, actions=actions, rows=rows, stations=stations), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
