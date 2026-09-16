#!/usr/bin/env python3
"""Collect and analyze WA-02's seeded station/turn route randomization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.execution.planning import NavigationConfig
from bench.tasks.wa02.execution.navigation import build_episode_station_sampler


WAYPOINTS = (4, 5, 6, 7, 8, 10, 14, 17)
CIRCLE_POINTS = (4, 6, 8, 10, 14, 17)
TURN_POINTS = (5, 7)
RADIUS_M = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/wa02_station_circles"))
    parser.add_argument("--path-file", type=Path, default=Path("bench/tasks/wa02/resources/navigation/manual_base_path_layout15_style34_seed0.json"))
    parser.add_argument("--station-file", type=Path, default=Path("bench/tasks/wa02/resources/station/stations.json"))
    return parser.parse_args()


def lag1(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    denom = float(np.dot(centered, centered))
    return 0.0 if denom == 0.0 else float(np.dot(centered[:-1], centered[1:]) / denom)


def collect(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = NavigationConfig(
        station_base_pose_file=args.station_file.resolve(),
        manual_base_path_file=args.path_file.resolve(),
        manual_base_path_enabled=True,
    )
    payload = json.loads(args.path_file.read_text(encoding="utf-8"))
    samples = {int(item["index"]): item for item in payload["samples"]}
    rows: list[dict] = []
    visits: list[dict] = []
    for seed in range(args.count):
        sampler = build_episode_station_sampler(config, seed=seed)
        for index in WAYPOINTS:
            pose = sampler.route_poses[index]
            xy = np.asarray(pose.position[:2], dtype=float)
            summary = sampler.route_summaries[index]
            circle = (summary.get("sample") or {}).get("circle") or (samples[index].get("circle") or {})
            center = np.asarray(circle.get("center_xy", xy), dtype=float)
            nominal = summary.get("nominal_turn_angle")
            sampled = summary.get("sampled_turn_angle")
            rows.append({
                "seed": seed, "path_index": index, "name": samples[index].get("name"),
                "role": summary.get("route_role"), "x": float(xy[0]), "y": float(xy[1]),
                "center_x": float(center[0]), "center_y": float(center[1]),
                "radius_m": float(np.linalg.norm(xy - center)) if index in CIRCLE_POINTS else np.nan,
                "yaw_rad": float(math.atan2(2 * pose.quaternion[3] * pose.quaternion[2], 1 - 2 * pose.quaternion[2] ** 2)),
                "yaw_deg": float(math.degrees(math.atan2(2 * pose.quaternion[3] * pose.quaternion[2], 1 - 2 * pose.quaternion[2] ** 2))),
                "turn_offset_deg": float(math.degrees(sampled - nominal)) if sampled is not None else np.nan,
                "nominal_turn_deg": float(math.degrees(nominal)) if nominal is not None else np.nan,
                "inside_disk": bool(index not in CIRCLE_POINTS or np.linalg.norm(xy - center) <= RADIUS_M + 1e-9),
            })
        previous = None
        cumulative = 0.0
        for visit, index in enumerate(WAYPOINTS):
            xy = np.asarray(sampler.route_poses[index].position[:2], dtype=float)
            leg = 0.0 if previous is None else float(np.linalg.norm(xy - previous))
            cumulative += leg
            visits.append({"seed": seed, "visit_index": visit, "path_index": index, "x": xy[0], "y": xy[1], "leg_m": leg, "cumulative_path_m": cumulative})
            previous = xy
    return pd.DataFrame(rows), pd.DataFrame(visits)


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points, visits = collect(args)
    point_csv = args.output_dir / "wa02_200seed_waypoint_draws.csv"
    visit_csv = args.output_dir / "wa02_200seed_episode_route_visits.csv"
    points.to_csv(point_csv, index=False); visits.to_csv(visit_csv, index=False)
    disks = points[points.path_index.isin(CIRCLE_POINTS)]
    radii = np.sort(disks.radius_m.to_numpy()); empirical = np.arange(1, len(radii) + 1) / len(radii)
    theoretical = np.clip((radii / RADIUS_M) ** 2, 0.0, 1.0)
    turns = points[points.path_index.isin(TURN_POINTS)]
    summary = {
        "task": "WA-02", "n_seeds": args.count, "seed_range": [0, args.count - 1],
        "sampled_waypoints": list(WAYPOINTS), "circle_waypoints": list(CIRCLE_POINTS), "turn_waypoints": list(TURN_POINTS),
        "radius_m": RADIUS_M, "uniform_disk_expected_mean_radius_m": 2 * RADIUS_M / 3,
        "aggregate_radial_mean_m": float(disks.radius_m.mean()), "aggregate_radial_p95_m": float(np.percentile(radii, 95)),
        "aggregate_radial_max_m": float(radii.max()), "aggregate_radial_cdf_ks_distance": float(np.max(np.abs(empirical - theoretical))),
        "all_points_inside_disks": bool(disks.inside_disk.all()),
        "turn_offset_min_deg": float(turns.turn_offset_deg.min()), "turn_offset_max_deg": float(turns.turn_offset_deg.max()),
        "turn_offset_mean_deg": float(turns.turn_offset_deg.mean()), "turn_offset_std_deg": float(turns.turn_offset_deg.std()),
        "route_length_mean_m": float(visits.groupby("seed").tail(1).cumulative_path_m.mean()),
        "route_length_std_m": float(visits.groupby("seed").tail(1).cumulative_path_m.std()),
        "waypoint_csv": str(point_csv.resolve()), "route_visit_csv": str(visit_csv.resolve()),
        "path_file": str(args.path_file.resolve()), "station_file": str(args.station_file.resolve()),
    }
    (args.output_dir / "wa02_station_circle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "meta.json").write_text(json.dumps({"task": "WA-02", "n_seeds": args.count, "radius_m": RADIUS_M, "turn_half_range_deg": 5.0}, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for index, group in points.groupby("path_index"):
        axes[0].plot(group.seed, group.radius_m if index in CIRCLE_POINTS else group.turn_offset_deg, ".", ms=3, label=f"P{index}")
    axes[0].set(xlabel="seed", ylabel="radius (m) / turn offset (deg)"); axes[0].grid(alpha=.25); axes[0].legend(ncol=4)
    for index, group in disks.groupby("path_index"):
        axes[1].scatter((group.x - group.center_x) * 100, (group.y - group.center_y) * 100, s=5, label=f"P{index}")
    axes[1].add_patch(plt.Circle((0, 0), 5, fill=False, color="black", lw=1)); axes[1].set_aspect("equal"); axes[1].set(xlabel="delta x (cm)", ylabel="delta y (cm)"); axes[1].grid(alpha=.25); axes[1].legend(ncol=3)
    fig.suptitle("WA-02 base randomization · seeds 0–%d" % (args.count - 1))
    fig.savefig(args.output_dir / "wa02_200seed_temporal_distribution.png", dpi=220); fig.savefig(args.output_dir / "wa02_200seed_temporal_distribution.pdf"); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
