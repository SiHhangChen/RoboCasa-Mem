#!/usr/bin/env python3
"""WA-01-style 200-seed analysis for TM01 sampled base waypoints."""

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
from bench.tasks.tm01.execution.navigation import (
    MAIN_LOOP,
    MAIN_LOOP_TURN_INDICES,
    sample_episode_path_poses,
)


L1_ROUTE = (
    0, 19, 3, 4, 5, 6, 7,
    20, 12, 13, 14, 15, 16, 17, 18, 0,
    19, 3, 4, 5, 6, 7,
    8, 9, 10, 11,
)
L2_ROUTE = (
    0, 19, 3, 4, 5, 6, 7,
    20, 12, 13, 14, 15, 16, 17, 18, 0, 1,
)
TURN_INDICES = (*MAIN_LOOP_TURN_INDICES, 2)
RADIUS_M = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-file",
        type=Path,
        default=Path(
            "bench/tasks/tm01/resources/navigation/"
            "manual_base_path_layout39_style24_seed0.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/tm01_station_circles"),
    )
    parser.add_argument("--count", type=int, default=200)
    return parser.parse_args()


def episode_variant(seed: int) -> tuple[str, str, tuple[int, ...]]:
    index = int(seed) % 4
    route = "L1" if index in (0, 2) else "L2"
    drawer_order = "left,right" if index in (0, 1) else "right,left"
    return route, drawer_order, L1_ROUTE if route == "L1" else L2_ROUTE


def lag1_acf(values: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centered[:-1], centered[1:]) / denominator)


def collect(payload: dict, count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = {int(item["index"]): dict(item) for item in payload["samples"]}
    waypoint_rows: list[dict] = []
    visit_rows: list[dict] = []

    for seed in range(count):
        poses, summaries = sample_episode_path_poses(payload, seed=seed)
        route, drawer_order, route_indices = episode_variant(seed)
        for index, item in samples.items():
            pose = poses[index]
            summary = summaries[index]
            xy = np.asarray(pose.position[:2], dtype=float)
            center = np.asarray(item["circle"]["center_xy"], dtype=float)
            row = {
                "seed": seed,
                "seed_mod4": seed % 4,
                "route": route,
                "drawer_order": drawer_order,
                "path_index": index,
                "name": item.get("name"),
                "role": summary["route_role"],
                "x": float(xy[0]),
                "y": float(xy[1]),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "radius_m": float(np.linalg.norm(xy - center)),
                "yaw_rad": quat_xyzw_to_yaw(pose.quaternion),
                "yaw_deg": math.degrees(quat_xyzw_to_yaw(pose.quaternion)),
                "turn_offset_deg": (
                    math.degrees(float(summary["sampled_turn_offset"]))
                    if "sampled_turn_offset" in summary
                    else np.nan
                ),
                "nominal_turn_deg": (
                    math.degrees(float(summary["nominal_turn_angle"]))
                    if "nominal_turn_angle" in summary
                    else np.nan
                ),
                "inside_disk": float(np.linalg.norm(xy - center))
                <= float(item["circle"]["radius_m"]) + 1.0e-9,
            }
            waypoint_rows.append(row)

        previous_xy: np.ndarray | None = None
        previous_path_index: int | None = None
        cumulative = 0.0
        for visit_index, path_index in enumerate(route_indices):
            pose = poses[path_index]
            xy = np.asarray(pose.position[:2], dtype=float)
            leg = 0.0 if previous_xy is None else float(np.linalg.norm(xy - previous_xy))
            cumulative += leg
            visit_rows.append(
                {
                    "seed": seed,
                    "seed_mod4": seed % 4,
                    "route": route,
                    "drawer_order": drawer_order,
                    "visit_index": visit_index,
                    "from_path_index": previous_path_index,
                    "path_index": path_index,
                    "name": samples[path_index].get("name"),
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "yaw_rad": quat_xyzw_to_yaw(pose.quaternion),
                    "yaw_deg": math.degrees(quat_xyzw_to_yaw(pose.quaternion)),
                    "leg_m": leg,
                    "cumulative_path_m": cumulative,
                }
            )
            previous_xy = xy
            previous_path_index = path_index

    return pd.DataFrame(waypoint_rows), pd.DataFrame(visit_rows)


def summarize(
    waypoint_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    payload: dict,
    count: int,
) -> dict:
    per_point = {}
    for index, group in waypoint_df.groupby("path_index"):
        jumps = np.linalg.norm(
            group[["x", "y"]].to_numpy()[1:]
            - group[["x", "y"]].to_numpy()[:-1],
            axis=1,
        )
        per_point[str(index)] = {
            "name": str(group.iloc[0]["name"]),
            "role": str(group.iloc[0]["role"]),
            "radial_mean_m": float(group.radius_m.mean()),
            "radial_p95_m": float(np.percentile(group.radius_m, 95)),
            "radial_max_m": float(group.radius_m.max()),
            "consecutive_seed_xy_jump_mean_m": float(jumps.mean()),
            "radial_lag1_acf": lag1_acf(group.radius_m.to_numpy()),
        }

    turn_stats = {}
    turns = waypoint_df[waypoint_df.path_index.isin(TURN_INDICES)]
    for index, group in turns.groupby("path_index"):
        turn_stats[str(index)] = {
            "name": str(group.iloc[0]["name"]),
            "offset_min_deg": float(group.turn_offset_deg.min()),
            "offset_max_deg": float(group.turn_offset_deg.max()),
            "offset_mean_deg": float(group.turn_offset_deg.mean()),
            "offset_std_deg": float(group.turn_offset_deg.std()),
        }

    terminal = visit_df.groupby("seed", as_index=False).tail(1)
    route_stats = {}
    for route, group in terminal.groupby("route"):
        route_stats[str(route)] = {
            "episodes": int(len(group)),
            "visits_per_episode": int(
                visit_df[visit_df.seed == int(group.iloc[0].seed)].shape[0]
            ),
            "path_length_mean_m": float(group.cumulative_path_m.mean()),
            "path_length_std_m": float(group.cumulative_path_m.std()),
            "path_length_min_m": float(group.cumulative_path_m.min()),
            "path_length_max_m": float(group.cumulative_path_m.max()),
        }

    edge_stats = {}
    edges = visit_df[visit_df.from_path_index.notna()].copy()
    edges["edge"] = (
        edges.from_path_index.astype(int).astype(str)
        + "->"
        + edges.path_index.astype(int).astype(str)
    )
    for edge, group in edges.groupby("edge"):
        edge_stats[str(edge)] = {
            "observations": int(len(group)),
            "length_mean_m": float(group.leg_m.mean()),
            "length_std_m": float(group.leg_m.std()),
            "length_min_m": float(group.leg_m.min()),
            "length_max_m": float(group.leg_m.max()),
        }

    ordered_turns = waypoint_df[
        waypoint_df.path_index.isin(MAIN_LOOP_TURN_INDICES)
    ].pivot(index="seed", columns="path_index", values="turn_offset_deg")
    ordered_turns = ordered_turns[list(MAIN_LOOP_TURN_INDICES)]
    turn_sums = ordered_turns.sum(axis=1)

    radii = np.sort(waypoint_df.radius_m.to_numpy())
    empirical = np.arange(1, len(radii) + 1) / len(radii)
    theoretical = np.clip((radii / RADIUS_M) ** 2, 0.0, 1.0)
    return {
        "task": "TM01",
        "n_seeds": count,
        "seed_range": [0, count - 1],
        "unique_waypoints_per_seed": len(payload["samples"]),
        "waypoint_draw_count": int(len(waypoint_df)),
        "route_visit_count": int(len(visit_df)),
        "radius_m": RADIUS_M,
        "uniform_disk_expected_mean_radius_m": 2.0 * RADIUS_M / 3.0,
        "aggregate_radial_mean_m": float(waypoint_df.radius_m.mean()),
        "aggregate_radial_p95_m": float(np.percentile(waypoint_df.radius_m, 95)),
        "aggregate_radial_max_m": float(waypoint_df.radius_m.max()),
        "aggregate_radial_cdf_ks_distance": float(
            np.max(np.abs(empirical - theoretical))
        ),
        "all_points_inside_disks": bool(waypoint_df.inside_disk.all()),
        "main_loop_turn_indices": list(MAIN_LOOP_TURN_INDICES),
        "turn_half_range_deg": 5.0,
        "main_loop_turn_sum_max_abs_deg": float(np.max(np.abs(turn_sums))),
        "main_loop_turn_covariance_deg2": ordered_turns.cov().to_numpy().tolist(),
        "route_stats": route_stats,
        "directed_edge_stats": edge_stats,
        "per_waypoint": per_point,
        "per_turn": turn_stats,
    }


def plot_analysis(
    waypoint_df: pd.DataFrame,
    visit_df: pd.DataFrame,
    payload: dict,
    output: Path,
) -> None:
    samples = {int(item["index"]): item for item in payload["samples"]}
    centers = np.asarray([samples[index]["circle"]["center_xy"] for index in range(21)])
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, 21))

    fig = plt.figure(figsize=(16, 14))
    grid = fig.add_gridspec(3, 3, height_ratios=(1.25, 1.0, 1.0), hspace=0.48, wspace=0.32)

    spatial = fig.add_subplot(grid[0, :])
    for route_indices, color, label in (
        (L1_ROUTE, "#9e9e9e", "L1 nominal route"),
        (L2_ROUTE, "#424242", "L2 nominal route"),
    ):
        route_centers = centers[list(route_indices)]
        spatial.plot(
            route_centers[:, 0], route_centers[:, 1], color=color, lw=1.2,
            alpha=0.75, label=label, zorder=1,
        )
    for index in range(21):
        group = waypoint_df[waypoint_df.path_index == index]
        spatial.scatter(group.x, group.y, s=4, color=colors[index], alpha=0.22, zorder=2)
        spatial.add_patch(Circle(centers[index], RADIUS_M, fill=False, color=colors[index], lw=0.8))
        spatial.text(
            centers[index, 0], centers[index, 1], str(index), fontsize=7,
            ha="center", va="center", color="black", zorder=3,
        )
    spatial.set_aspect("equal")
    spatial.set_xlabel("world x (m)")
    spatial.set_ylabel("world y (m)")
    spatial.set_title("Spatial support: 200 seeds x 21 episode-stable waypoint disks")
    spatial.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))

    radial = fig.add_subplot(grid[1, 0])
    radial_values = [
        waypoint_df[waypoint_df.path_index == index].radius_m.to_numpy()
        for index in range(21)
    ]
    parts = radial.violinplot(radial_values, positions=range(21), showextrema=False, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    radial.axhline(2 * RADIUS_M / 3, color="0.25", ls="--", lw=1, label="uniform mean")
    radial.axhline(RADIUS_M, color="0.5", ls=":", lw=1, label="5 cm bound")
    radial.set_xticks(range(21), [str(index) for index in range(21)], fontsize=7)
    radial.set_ylabel("distance to center (m)")
    radial.set_title("Radial fill by waypoint")
    radial.legend(frameon=False, fontsize=8)

    turns = fig.add_subplot(grid[1, 1])
    turn_values = [
        waypoint_df[waypoint_df.path_index == index].turn_offset_deg.to_numpy()
        for index in MAIN_LOOP_TURN_INDICES
    ]
    parts = turns.violinplot(
        turn_values,
        positions=range(len(MAIN_LOOP_TURN_INDICES)),
        showextrema=False,
        widths=0.8,
    )
    for body in parts["bodies"]:
        body.set_facecolor("#d95f02")
        body.set_alpha(0.65)
    turns.axhline(5, color="0.3", ls="--")
    turns.axhline(-5, color="0.3", ls="--")
    turns.axhline(0, color="0.65", lw=0.8)
    turns.set_xticks(
        range(len(MAIN_LOOP_TURN_INDICES)),
        [str(index) for index in MAIN_LOOP_TURN_INDICES],
    )
    turns.set_ylabel("turn offset (deg)")
    turns.set_title("Closed-loop turn draws (jointly constrained)")

    route_lengths = fig.add_subplot(grid[1, 2])
    terminal = visit_df.groupby("seed", as_index=False).tail(1)
    length_values = [
        terminal[terminal.route == route].cumulative_path_m.to_numpy()
        for route in ("L1", "L2")
    ]
    parts = route_lengths.violinplot(length_values, positions=(0, 1), showextrema=False, widths=0.7)
    for body, color in zip(parts["bodies"], ("#3182bd", "#e6550d")):
        body.set_facecolor(color)
    route_lengths.set_xticks((0, 1), ("L1: 26 visits", "L2: 17 visits"))
    route_lengths.set_ylabel("sampled route length (m)")
    route_lengths.set_title("Full episode base-path length")

    jumps = fig.add_subplot(grid[2, 0])
    jump_values = []
    for index in range(21):
        values = waypoint_df[waypoint_df.path_index == index][["x", "y"]].to_numpy()
        jump_values.extend(np.linalg.norm(values[1:] - values[:-1], axis=1))
    jumps.violinplot([np.asarray(jump_values)], positions=[0], showextrema=False, widths=0.65)
    jumps.axhline(2 * RADIUS_M, color="0.35", ls="--", label="diameter = 10 cm")
    jumps.set_xticks([0], ["all waypoints"])
    jumps.set_ylabel("consecutive-seed XY jump (m)")
    jumps.set_title("Seed-order temporal jumps")
    jumps.legend(frameon=False, fontsize=8)

    cdf_ax = fig.add_subplot(grid[2, 1])
    radii = np.sort(waypoint_df.radius_m.to_numpy())
    cdf_ax.plot(radii, np.arange(1, len(radii) + 1) / len(radii), label="empirical")
    theoretical_r = np.linspace(0, RADIUS_M, 300)
    cdf_ax.plot(
        theoretical_r,
        (theoretical_r / RADIUS_M) ** 2,
        "k--",
        label="uniform disk theory",
    )
    cdf_ax.set_xlabel("radius (m)")
    cdf_ax.set_ylabel("CDF")
    cdf_ax.set_title("Aggregate radial CDF")
    cdf_ax.legend(frameon=False)

    cumulative_ax = fig.add_subplot(grid[2, 2])
    offsets = waypoint_df[
        waypoint_df.path_index.isin(MAIN_LOOP_TURN_INDICES)
    ].pivot(index="seed", columns="path_index", values="turn_offset_deg")
    offsets = offsets[list(MAIN_LOOP_TURN_INDICES)].to_numpy()
    cumulative = np.column_stack((np.zeros(len(offsets)), np.cumsum(offsets, axis=1)))
    x_values = np.arange(cumulative.shape[1])
    for row in cumulative[:50]:
        cumulative_ax.plot(x_values, row, color="#9ecae1", alpha=0.18, lw=0.7)
    cumulative_ax.fill_between(
        x_values,
        np.percentile(cumulative, 5, axis=0),
        np.percentile(cumulative, 95, axis=0),
        color="#6baed6",
        alpha=0.3,
        label="5-95%",
    )
    cumulative_ax.plot(
        x_values, np.median(cumulative, axis=0), "o-", color="#08519c", label="median"
    )
    cumulative_ax.axhline(0, color="0.5", lw=0.8)
    cumulative_ax.set_xticks(x_values, ["start", *map(str, MAIN_LOOP_TURN_INDICES)], rotation=45)
    cumulative_ax.set_ylabel("accumulated turn noise (deg)")
    cumulative_ax.set_title("Heading perturbation closes at waypoint 18")
    cumulative_ax.legend(frameon=False, fontsize=8)

    fig.suptitle("TM01 base randomization: seeds 0-199", fontsize=18, fontweight="bold", y=0.995)
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_spatial_support(
    waypoint_df: pd.DataFrame,
    payload: dict,
    output: Path,
) -> None:
    """Render each waypoint disk at a readable local scale."""
    samples = {int(item["index"]): item for item in payload["samples"]}
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, 21))
    limit_cm = RADIUS_M * 100.0 + 0.65

    fig, axes = plt.subplots(4, 6, figsize=(24, 16), constrained_layout=True)
    flat_axes = axes.ravel()

    for index in range(21):
        ax = flat_axes[index]
        group = waypoint_df[waypoint_df.path_index == index]
        center = np.asarray(samples[index]["circle"]["center_xy"], dtype=float)
        offsets_cm = (
            group[["x", "y"]].to_numpy(dtype=float) - center[None, :]
        ) * 100.0

        ax.scatter(
            offsets_cm[:, 0],
            offsets_cm[:, 1],
            s=18,
            color=colors[index],
            alpha=0.65,
            edgecolors="none",
            rasterized=True,
        )
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                RADIUS_M * 100.0,
                fill=False,
                color="black",
                lw=1.25,
            )
        )
        ax.scatter(
            [0.0], [0.0], marker="+", s=65, color="black", linewidths=1.25,
            label="nominal center",
        )
        ax.set(
            xlim=(-limit_cm, limit_cm),
            ylim=(-limit_cm, limit_cm),
            xticks=(-5, 0, 5),
            yticks=(-5, 0, 5),
            xlabel="delta x (cm)",
            ylabel="delta y (cm)",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="0.88", lw=0.6)
        name = samples[index].get("name") or "unnamed"
        ax.set_title(f"Waypoint {index}: {name}\n200 samples", fontsize=11)

    aggregate_ax = flat_axes[21]
    aggregate_offsets = np.column_stack(
        (
            (waypoint_df.x - waypoint_df.center_x) * 100.0,
            (waypoint_df.y - waypoint_df.center_y) * 100.0,
        )
    )
    aggregate_ax.scatter(
        aggregate_offsets[:, 0],
        aggregate_offsets[:, 1],
        s=5,
        color="#1769aa",
        alpha=0.22,
        edgecolors="none",
        rasterized=True,
    )
    aggregate_ax.add_patch(
        Circle(
            (0.0, 0.0),
            RADIUS_M * 100.0,
            fill=False,
            color="black",
            lw=1.5,
        )
    )
    aggregate_ax.scatter([0.0], [0.0], marker="+", s=70, color="black", linewidths=1.3)
    aggregate_ax.set(
        xlim=(-limit_cm, limit_cm),
        ylim=(-limit_cm, limit_cm),
        xticks=(-5, 0, 5),
        yticks=(-5, 0, 5),
        xlabel="delta x (cm)",
        ylabel="delta y (cm)",
    )
    aggregate_ax.set_aspect("equal", adjustable="box")
    aggregate_ax.grid(color="0.88", lw=0.6)
    aggregate_ax.set_title("All waypoints aligned\n4,200 samples", fontsize=11)

    for ax in flat_axes[22:]:
        ax.axis("off")

    fig.suptitle(
        "TM01 spatial support: seeds 0-199 (uniform 5 cm disks)",
        fontsize=22,
        fontweight="bold",
    )
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    payload = json.loads(args.path_file.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    waypoint_df, visit_df = collect(payload, args.count)
    waypoint_csv = args.output_dir / "tm01_200seed_waypoint_draws.csv"
    visit_csv = args.output_dir / "tm01_200seed_episode_route_visits.csv"
    waypoint_df.to_csv(waypoint_csv, index=False)
    visit_df.to_csv(visit_csv, index=False)

    summary = summarize(waypoint_df, visit_df, payload, args.count)
    summary["waypoint_csv"] = str(waypoint_csv.resolve())
    summary["route_visit_csv"] = str(visit_csv.resolve())
    summary_path = args.output_dir / "tm01_station_circle_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    meta = {
        "task": "TM01",
        "n_seeds": args.count,
        "seed_range": [0, args.count - 1],
        "radius_m": RADIUS_M,
        "turn_half_range_deg": 5.0,
        "path_file": str(args.path_file.resolve()),
        "waypoint_csv": str(waypoint_csv.resolve()),
        "route_visit_csv": str(visit_csv.resolve()),
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    plot_analysis(
        waypoint_df,
        visit_df,
        payload,
        args.output_dir / "tm01_200seed_temporal_distribution",
    )
    plot_spatial_support(
        waypoint_df,
        payload,
        args.output_dir / "tm01_200seed_spatial_support",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
