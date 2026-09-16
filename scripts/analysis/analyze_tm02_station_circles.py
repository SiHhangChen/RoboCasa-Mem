#!/usr/bin/env python3
"""200-seed TM02 base sampling analysis (TM01/WA01-style)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.execution.navigation.station_poses import quat_xyzw_to_yaw
from bench.tasks.tm02.execution.navigation import TM02_TURN_INDICES, sample_episode_path_poses

DEFAULT_PATH = ROOT / "bench/tasks/tm02/resources/navigation/manual_base_path_layout15_style34_seed0.json"
ROUTE = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 11, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path-file", type=Path, default=DEFAULT_PATH)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/analysis/tm02_station_circles")
    p.add_argument("--count", type=int, default=200)
    return p.parse_args()


def collect(payload: dict, count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = {int(x["index"]): dict(x) for x in payload["samples"]}
    waypoints: list[dict] = []
    visits: list[dict] = []
    for seed in range(count):
        poses, summaries = sample_episode_path_poses(payload, seed=seed)
        for index, item in samples.items():
            pose, summary = poses[index], summaries[index]
            xy = np.asarray(pose.position[:2], dtype=float)
            center = np.asarray(item["circle"]["center_xy"], dtype=float)
            yaw = quat_xyzw_to_yaw(pose.quaternion)
            waypoints.append({
                "seed": seed, "path_index": index, "name": item.get("name"),
                "role": summary["route_role"], "x": xy[0], "y": xy[1],
                "center_x": center[0], "center_y": center[1],
                "radius_m": float(np.linalg.norm(xy - center)),
                "radius_bound_m": float(item["circle"]["radius_m"]),
                "yaw_rad": yaw, "yaw_deg": math.degrees(yaw),
                "turn_offset_deg": math.degrees(float(summary["sampled_turn_offset"])) if "sampled_turn_offset" in summary else np.nan,
                "nominal_turn_deg": math.degrees(float(summary["nominal_turn_angle"])) if "nominal_turn_angle" in summary else np.nan,
                "inside_disk": float(np.linalg.norm(xy - center)) <= float(item["circle"]["radius_m"]) + 1e-9,
            })
        previous = None
        cumulative = 0.0
        for visit_index, index in enumerate(ROUTE):
            xy = np.asarray(poses[index].position[:2], dtype=float)
            leg = 0.0 if previous is None else float(np.linalg.norm(xy - previous))
            cumulative += leg
            visits.append({"seed": seed, "visit_index": visit_index, "path_index": index,
                           "name": samples[index].get("name"), "x": xy[0], "y": xy[1],
                           "yaw_deg": math.degrees(quat_xyzw_to_yaw(poses[index].quaternion)),
                           "leg_m": leg, "cumulative_path_m": cumulative})
            previous = xy
    return pd.DataFrame(waypoints), pd.DataFrame(visits)


def summarize(wp: pd.DataFrame, visits: pd.DataFrame, payload: dict, count: int) -> dict:
    turns = wp[wp.path_index.isin(TM02_TURN_INDICES)].pivot(index="seed", columns="path_index", values="turn_offset_deg")
    turns = turns[list(TM02_TURN_INDICES)]
    per_point = {}
    for index, g in wp.groupby("path_index"):
        per_point[str(index)] = {"sample_name": str(g.iloc[0]["name"]),
                                 "role": str(g.iloc[0]["role"]), "radius_mean_m": float(g.radius_m.mean()),
                                 "radius_p95_m": float(np.percentile(g.radius_m, 95)),
                                 "radius_max_m": float(g.radius_m.max()),
                                 "inside_disk": bool(g.inside_disk.all())}
    terminal = visits.groupby("seed", as_index=False).tail(1)
    lengths = terminal.cumulative_path_m
    per_turn = {}
    for index, g in wp[wp.path_index.isin(TM02_TURN_INDICES)].groupby("path_index"):
        per_turn[str(index)] = {
            "sample_name": str(g.iloc[0]["name"]),
            "offset_min_deg": float(g.turn_offset_deg.min()),
            "offset_max_deg": float(g.turn_offset_deg.max()),
            "offset_mean_deg": float(g.turn_offset_deg.mean()),
            "offset_std_deg": float(g.turn_offset_deg.std()),
        }
    return {"task": "TM02", "n_seeds": count, "seed_range": [0, count - 1],
            "unique_waypoints_per_seed": len(payload["samples"]),
            "waypoint_draw_count": int(len(wp)), "route_visit_count": int(len(visits)),
            "route_indices": list(ROUTE), "turn_indices": list(TM02_TURN_INDICES),
            "turn_half_range_deg": 5.0,
            "turn_sum_max_abs_deg": float(np.max(np.abs(turns.sum(axis=1)))),
            "turn_covariance_deg2": turns.cov().to_numpy().tolist(),
            "aggregate_radius_mean_m": float(wp.radius_m.mean()),
            "aggregate_radius_p95_m": float(np.percentile(wp.radius_m, 95)),
            "aggregate_radius_max_m": float(wp.radius_m.max()),
            "all_points_inside_disks": bool(wp.inside_disk.all()),
            "route_length_mean_m": float(lengths.mean()), "route_length_std_m": float(lengths.std()),
            "route_length_min_m": float(lengths.min()), "route_length_max_m": float(lengths.max()),
            "per_waypoint": per_point, "per_turn": per_turn}


def plot(wp: pd.DataFrame, payload: dict, out: Path) -> None:
    samples = {int(x["index"]): x for x in payload["samples"]}
    n = len(samples)
    colors = plt.cm.turbo(np.linspace(.05, .95, n))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax = axes[0, 0]
    centers = np.array([samples[i]["circle"]["center_xy"] for i in range(n)])
    ax.plot(centers[:, 0], centers[:, 1], "k--", alpha=.5)
    for i in range(n):
        g = wp[wp.path_index == i]; ax.scatter(g.x, g.y, s=5, alpha=.25, color=colors[i]);
        ax.add_patch(Circle(centers[i], samples[i]["circle"]["radius_m"], fill=False, color=colors[i]))
        ax.text(*centers[i], str(i), fontsize=8)
    ax.set_aspect("equal"); ax.set_title("TM02 spatial support: 200 seeds"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax = axes[0, 1]
    vals = [wp[wp.path_index == i].radius_m.to_numpy() for i in range(n)]
    ax.violinplot(vals, positions=range(n), showextrema=False); ax.axhline(wp.radius_bound_m.iloc[0] * 2 / 3, ls="--", color="0.3"); ax.set_title("Station-circle radial draws"); ax.set_xlabel("path index"); ax.set_ylabel("radius (m)")
    ax = axes[1, 0]
    vals = [wp[wp.path_index == i].turn_offset_deg.dropna().to_numpy() for i in TM02_TURN_INDICES]
    ax.violinplot(vals, positions=range(len(vals)), showextrema=False); ax.axhline(5, ls="--", color="0.3"); ax.axhline(-5, ls="--", color="0.3"); ax.set_xticks(range(len(vals)), [str(i) for i in TM02_TURN_INDICES]); ax.set_title("Jointly constrained turn-angle draws"); ax.set_ylabel("offset (deg)")
    ax = axes[1, 1]
    t = wp[wp.path_index.isin(TM02_TURN_INDICES)].pivot(index="seed", columns="path_index", values="turn_offset_deg")[list(TM02_TURN_INDICES)]
    cumulative = np.column_stack((np.zeros(len(t)), np.cumsum(t.to_numpy(), axis=1)))
    ax.plot(cumulative[:50].T, color="#6baed6", alpha=.2); ax.plot(np.median(cumulative, axis=0), "o-", color="#08519c"); ax.axhline(0, color="0.4"); ax.set_title("Accumulated turn noise closes"); ax.set_xticks(range(6), ["start", *map(str, TM02_TURN_INDICES)]); ax.set_ylabel("degrees")
    fig.suptitle("TM02 base randomization: seeds 0–199", fontsize=16, fontweight="bold")
    fig.savefig(out.with_suffix(".png"), dpi=180); fig.savefig(out.with_suffix(".pdf")); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.count <= 0: raise ValueError("count must be positive")
    payload = json.loads(args.path_file.read_text(encoding="utf-8")); args.output_dir.mkdir(parents=True, exist_ok=True)
    wp, visits = collect(payload, args.count)
    wp.to_csv(args.output_dir / "tm02_200seed_waypoint_draws.csv", index=False); visits.to_csv(args.output_dir / "tm02_200seed_episode_route_visits.csv", index=False)
    summary = summarize(wp, visits, payload, args.count); summary["path_file"] = str(args.path_file.resolve())
    (args.output_dir / "tm02_station_circle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "meta.json").write_text(json.dumps({"task": "TM02", "n_seeds": args.count, "seed_range": [0, args.count - 1], "path_file": str(args.path_file.resolve())}, indent=2), encoding="utf-8")
    plot(wp, payload, args.output_dir / "tm02_200seed_temporal_distribution")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
