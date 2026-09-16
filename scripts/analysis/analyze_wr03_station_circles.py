#!/usr/bin/env python3
"""Collect and visualize the actual WR-03 expert base route for seeds 0-199."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bench.execution.sampling.circle_sampling import sample_uniform_disk
from bench.execution.sampling.rng import SeedStream
from bench.tasks.wr03.config import load_task_config
from bench.tasks.wr03.domain.episode import resolve_episode_plan


RESET_CENTER = np.array([5.216, -1.055])
RADIUS = 0.05
RESET_SALT = 0x57523349
STATION_SALT = 0x53544154
NAVIGATION_NAMESPACE = "wr03_navigation"
LABELS = (
    "P0 reset",
    "P1 front/pre-close",
    "P2 close door 1",
    "P3 close door 2",
    "P4 front/memory",
    "P5 retrieval door",
    "P6 retrieval storage",
    "P7 item/place",
)


def _circle_draw(
    center: np.ndarray,
    seed: int,
    station_id: str,
    count: int,
    radius: float = RADIUS,
) -> np.ndarray:
    stream = SeedStream(base_seed=seed, namespace=NAVIGATION_NAMESPACE)
    draw_seed = stream.seed("circle", station_id, count)
    return sample_uniform_disk(center, radius, seed=draw_seed, salt=STATION_SALT)


def _station_point(stations: dict, seed: int, station_id: str, count: int) -> np.ndarray:
    values = stations[station_id]
    if "circle" in values:
        circle = values["circle"]
        return _circle_draw(
            np.asarray(circle["center_xy"], dtype=float),
            seed,
            station_id,
            count,
            float(circle["radius_m"]),
        )
    return np.asarray(values["samples"][0]["base_xy"], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    config = load_task_config()
    station_path = config.path(config.resources["stations"])
    stations = json.loads(station_path.read_text(encoding="utf-8"))["stations"]
    front_center = np.asarray(
        stations["cabinet_front_station"]["circle"]["center_xy"], dtype=float
    )
    item_center = np.asarray(stations["item_station"]["circle"]["center_xy"], dtype=float)

    rows = []
    for seed in range(args.count):
        plan = resolve_episode_plan(seed=seed, **dict(config.episode))
        first, second = plan.close_order
        level = plan.stored_level
        points = (
            sample_uniform_disk(RESET_CENTER, RADIUS, seed=seed, salt=RESET_SALT),
            _circle_draw(front_center, seed, "cabinet_front_station", 0),
            _station_point(stations, seed, f"{first}_door_station", 0),
            _station_point(stations, seed, f"{second}_door_station", 0),
            _circle_draw(front_center, seed, "cabinet_front_station", 1),
            _station_point(stations, seed, f"{level}_door_station", 1),
            _station_point(stations, seed, f"{level}_storage_station", 0),
            _circle_draw(item_center, seed, "item_station", 0),
        )
        row = {
            "seed": seed,
            "stored_level": level,
            "close_order": f"{first}-{second}",
            "cabinet_material_id": plan.cabinet_material_id,
        }
        for index, point in enumerate(points):
            row[f"p{index}_x"] = float(point[0])
            row[f"p{index}_y"] = float(point[1])
        for index in range(len(points) - 1):
            row[f"leg{index + 1}_m"] = float(
                np.linalg.norm(points[index + 1] - points[index])
            )
        for key, point, center in (
            ("reset", points[0], RESET_CENTER),
            ("front0", points[1], front_center),
            ("door1", points[2], np.asarray(stations[f"{first}_door_station"]["circle"]["center_xy"], dtype=float)),
            ("door2", points[3], np.asarray(stations[f"{second}_door_station"]["circle"]["center_xy"], dtype=float)),
            ("front1", points[4], front_center),
            ("retrieval_door", points[5], np.asarray(stations[f"{level}_door_station"]["circle"]["center_xy"], dtype=float)),
            ("item", points[7], item_center),
        ):
            row[f"{key}_r"] = float(np.linalg.norm(point - center))
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = args.output_dir / "wr03_200seed_station_draws.csv"
    frame.to_csv(csv_path, index=False)

    colors = [
        "#3182bd", "#6baed6", "#756bb1", "#9e9ac8",
        "#31a354", "#fd8d3c", "#e6550d", "#de2d26",
    ]
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(grid[0, :])
    for _, row in frame.iterrows():
        ax.plot(
            [row[f"p{i}_x"] for i in range(8)],
            [row[f"p{i}_y"] for i in range(8)],
            color="0.85", lw=0.35,
        )
    for index, (label, color) in enumerate(zip(LABELS, colors)):
        ax.scatter(
            frame[f"p{index}_x"], frame[f"p{index}_y"],
            s=11, alpha=0.55, color=color, label=label,
        )
    for center, color in (
        (RESET_CENTER, colors[0]), (front_center, colors[1]), (item_center, colors[7])
    ):
        ax.add_patch(plt.Circle(tuple(center), RADIUS, fill=False, color=color, lw=2))
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("Spatial support · actual 8-keypoint expert base route")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)

    ax = fig.add_subplot(grid[1, 0])
    radial_names = ("reset", "front0", "door1", "door2", "front1", "retrieval_door", "item")
    parts = ax.violinplot(
        [frame[f"{name}_r"] for name in radial_names],
        positions=range(len(radial_names)), showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors[:7]):
        body.set_facecolor(color)
    ax.axhline(2 * RADIUS / 3, color="0.35", ls="--")
    ax.set_xticks(range(len(radial_names)), radial_names, rotation=35, ha="right")
    ax.set_ylabel("distance to circle center (m)")
    ax.set_title("Radial fill · uniform-disk mean 3.33 cm")

    ax = fig.add_subplot(grid[1, 1:])
    parts = ax.violinplot(
        [frame[f"leg{i}_m"] for i in range(1, 8)],
        positions=range(7), showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors[1:]):
        body.set_facecolor(color)
    ax.set_xticks(range(7), [f"P{i}→P{i + 1}" for i in range(7)])
    ax.set_ylabel("straight-line XY distance (m)")
    ax.set_title("Seven route legs")

    ax = fig.add_subplot(grid[2, :2])
    for name, color in zip(radial_names, colors[:7]):
        ax.plot(
            frame.seed, frame[f"{name}_r"], ".", ms=4,
            alpha=0.65, color=color, label=name,
        )
    ax.axhline(RADIUS, color="0.35", ls="--")
    ax.set_xlabel("seed")
    ax.set_ylabel("radius (m)")
    ax.set_title("Temporal draw sequence")
    ax.legend(frameon=False, ncol=4)

    ax = fig.add_subplot(grid[2, 2])
    combinations = frame.groupby(["stored_level", "close_order"]).size().unstack(fill_value=0)
    combinations.plot.bar(ax=ax, color=["#9ecae1", "#fdae6b"])
    ax.set_ylabel("episodes")
    ax.set_title("Storage level × close order")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="close order", frameon=False)

    fig.suptitle(
        f"WR-03 base randomization · seeds 0–{args.count - 1}",
        fontsize=18, fontweight="bold", y=0.99,
    )
    png_path = args.output_dir / "wr03_200seed_temporal_distribution.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(
        args.output_dir / "wr03_200seed_temporal_distribution.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    summary = {
        "n_seeds": args.count,
        "seed_range": [0, args.count - 1],
        "radius_m": RADIUS,
        "reset_center_xy": RESET_CENTER.tolist(),
        "item_station_center_xy": item_center.tolist(),
        "cabinet_front_center_xy": front_center.tolist(),
        "route_keypoints": list(LABELS),
        "mean_radius_m": {
            name: float(frame[f"{name}_r"].mean()) for name in radial_names
        },
        "max_radius_m": {
            name: float(frame[f"{name}_r"].max()) for name in radial_names
        },
        "combination_counts": {
            f"{level}|{order}": int(count)
            for (level, order), count in frame.groupby(
                ["stored_level", "close_order"]
            ).size().items()
        },
        "csv": str(csv_path.resolve()),
        "plot_png": str(png_path.resolve()),
        "analysis_reference": (
            "outputs/analysis/wa01_station_circles/"
            "wa01_200seed_temporal_distribution.png"
        ),
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
