#!/usr/bin/env python3
"""Sample production TS01 navigation targets and plot WA01-style statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.execution.contracts import StageKind
from bench.execution.navigation.station_poses import StationPoseSampler, quat_xyzw_to_yaw
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs

STATIONS = ("sugar_station", "saucepan_station", "completion_button_station")
COLORS = ("#168aad", "#e07a32", "#398b5e")


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def collect(count, config_file):
    api = resolve_builtin_task_plugin("TS01").load_runtime_api()
    definition = get_task_definition("TS01")
    rows, episodes = [], []
    for seed in range(count):
        episode = definition.resolve(EpisodeRequest(
            task_id="TS01", seed=seed, variant="memory", split="train",
            env_split="pretrain", task_config_file=config_file,
        ))
        descriptor = api.build_descriptor(episode, True)
        plan = definition.build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
        sampler = StationPoseSampler.from_file(
            descriptor.resources["station_poses"], config=descriptor.navigation, seed=seed,
        )
        previous, previous_yaw = None, None
        last_by_station = {}
        cumulative = 0.0
        visit = 0
        for stage_index, stage in enumerate(plan.stages):
            if stage.kind != StageKind.NAVIGATE:
                continue
            station = stage.target.station_id
            pose = sampler.sample(station, purpose=stage.target.purpose)
            draw = sampler.last_summary(station)
            circle = draw["sample"]["circle"]
            xy = np.asarray(pose.position[:2])
            center = np.asarray(circle["center_xy"])
            yaw = quat_xyzw_to_yaw(pose.quaternion)
            radius = float(np.linalg.norm(xy - center))
            yaw_offset = float(wrap(yaw - circle["base_yaw"]))
            leg = 0.0 if previous is None else float(np.linalg.norm(xy - previous[1]))
            cumulative += leg
            revisit = np.nan if station not in last_by_station else float(np.linalg.norm(xy - last_by_station[station]))
            label = "B" if station == STATIONS[2] else ("S" if station == STATIONS[0] else "P") + str(draw["sample_index"] + 1)
            rows.append({
                "seed": seed, "K": descriptor.episode_plan.target_sugar_count,
                "N": descriptor.episode_plan.total_sugar_count,
                "stage_index": stage_index, "visit_index": visit,
                "visit_label": label, "station": station,
                "draw_index": draw["sample_index"], "draw_seed": draw["station_pose_selection"]["seed"],
                "x": float(xy[0]), "y": float(xy[1]),
                "center_x": float(center[0]), "center_y": float(center[1]),
                "radius_m": radius, "radius_bound_m": circle["radius_m"],
                "yaw_rad": yaw, "yaw_deg": float(np.degrees(yaw)),
                "yaw_offset_deg": float(np.degrees(yaw_offset)),
                "yaw_half_range_deg": float(np.degrees(circle["yaw_half_range_rad"])),
                "inside_disk": radius <= circle["radius_m"] + 1e-9,
                "inside_yaw": abs(yaw_offset) <= circle["yaw_half_range_rad"] + 1e-9,
                "revisit_shift_m": revisit, "leg_m": leg,
                "leg_type": "start" if previous is None else previous[0] + " -> " + station,
                "yaw_change_deg": np.nan if previous_yaw is None else float(np.degrees(wrap(yaw - previous_yaw))),
                "cumulative_path_m": cumulative,
            })
            last_by_station[station] = xy
            previous, previous_yaw = (station, xy), yaw
            visit += 1
        episodes.append({
            "seed": seed, "K": descriptor.episode_plan.target_sugar_count,
            "N": descriptor.episode_plan.total_sugar_count,
            "navigation_visits": visit, "target_route_length_m": cumulative,
        })
    return pd.DataFrame(rows), pd.DataFrame(episodes), sampler.payload


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(np.mean(values)),
            "std": float(np.std(values)), "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def summarize(draws, episodes, payload):
    result = {
        "task": "TS01", "n_seeds": len(episodes), "seed_range": [0, len(episodes) - 1],
        "data_kind": "production_sampler_targets_not_simulated_arrivals",
        "route_scope": "first sugar-station arrival through completion-button arrival; reset pose excluded",
        "yaw_policy": "independent absolute station yaw; noise does not accumulate",
        "draw_count": len(draws), "all_inside_disks": bool(draws.inside_disk.all()),
        "all_inside_yaw_bounds": bool(draws.inside_yaw.all()),
        "count_cells": {f"k{k}_n{n}": int(len(g)) for (k, n), g in episodes.groupby(["K", "N"])},
        "route_length_by_K_m": {str(k): stats(g.target_route_length_m) for k, g in episodes.groupby("K")},
        "per_station": {},
        "per_visit": {name: {"radius_m": stats(g.radius_m), "yaw_offset_deg": stats(g.yaw_offset_deg)} for name, g in draws.groupby("visit_label")},
        "revisits_by_station_m": {station: stats(g.revisit_shift_m) for station, g in draws.groupby("station") if g.revisit_shift_m.notna().any()},
        "legs_m": {name: stats(g.leg_m) for name, g in draws[draws.leg_type != "start"].groupby("leg_type")},
        "yaw_changes_deg": stats(draws.yaw_change_deg),
    }
    for station, g in draws.groupby("station"):
        circle = payload["stations"][station]["circle"]
        radius = circle["radius_m"]
        radial_test = kstest((g.radius_m / radius) ** 2, "uniform")
        yaw_test = kstest((g.yaw_offset_deg / np.degrees(circle["yaw_half_range_rad"]) + 1) / 2, "uniform")
        result["per_station"][station] = {
            "circle": circle, "radius_m": stats(g.radius_m), "yaw_offset_deg": stats(g.yaw_offset_deg),
            "expected_radius_mean_m": 2 * radius / 3,
            "expected_revisit_mean_m": 128 * radius / (45 * np.pi),
            "radial_uniformity_KS": {"D": float(radial_test.statistic), "p": float(radial_test.pvalue), "p_bonferroni_6": min(1.0, 6 * float(radial_test.pvalue))},
            "yaw_uniformity_KS": {"D": float(yaw_test.statistic), "p": float(yaw_test.pvalue), "p_bonferroni_6": min(1.0, 6 * float(yaw_test.pvalue))},
        }
    return result


def violins(ax, groups, labels, colors):
    for index, (values, color) in enumerate(zip(groups, colors)):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        if len(values) > 1 and np.ptp(values) > 0:
            body = ax.violinplot(values, positions=[index], widths=0.75, showextrema=False)["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.55)
        lo, hi = np.percentile(values, [5, 95])
        ax.plot([index, index], [lo, hi], color="#333333", lw=3)
        ax.scatter(index, np.median(values), color="white", edgecolor="#333333", zorder=3, s=22)
    ax.set_xticks(range(len(labels)), labels)
    ax.grid(axis="y", alpha=0.15)


def plot(draws, episodes, payload, output):
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1])
    spatial = fig.add_subplot(grid[0, :])
    for _, g in draws.groupby("seed"):
        spatial.plot(g.x, g.y, color="#999999", alpha=0.06, lw=0.6)
    for station, color, label in zip(STATIONS, COLORS, ("sugar", "saucepan", "button")):
        g = draws[draws.station == station]
        circle = payload["stations"][station]["circle"]
        spatial.scatter(g.x, g.y, s=8, color=color, alpha=0.35, label=f"{label}: {len(g)} visits")
        spatial.add_patch(Circle(circle["center_xy"], circle["radius_m"], fill=False, color=color, lw=2))
        spatial.scatter(*circle["center_xy"], marker="+", color="#222222", s=65)
    spatial.set_aspect("equal", adjustable="box")
    spatial.set_ylim(-3.79, -3.59)
    spatial.set_title(f"Spatial support: {len(episodes)} seeds, {len(draws)} sampled station arrivals")
    spatial.set_xlabel("world x (m)"); spatial.set_ylabel("world y (m)")
    spatial.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    labels = [name for name in ("S1", "P1", "S2", "P2", "S3", "P3", "B") if name in set(draws.visit_label)]
    visit_colors = [COLORS[2] if label == "B" else COLORS[0 if label.startswith("S") else 1] for label in labels]
    radial = fig.add_subplot(grid[1, 0])
    violins(radial, [draws[draws.visit_label == label].radius_m * 100 for label in labels], labels, visit_colors)
    radial.axhline(5, ls="--", color="gray", label="bound: 5 cm")
    radial.axhline(10 / 3, ls=":", color="gray", label="uniform-disk mean: 3.33 cm")
    radial.set_title("Radial fill by visit"); radial.set_ylabel("radius (cm)"); radial.legend(fontsize=8)
    revisit = fig.add_subplot(grid[1, 1])
    groups = [draws[draws.station == station].revisit_shift_m * 100 for station in STATIONS[:2]]
    violins(revisit, groups, ["sugar revisits", "saucepan revisits"], COLORS[:2])
    revisit.axhline(10, ls="--", color="gray")
    revisit.axhline(128 * 5 / (45 * np.pi), ls=":", color="gray")
    revisit.set_title("Same disk, new point"); revisit.set_ylabel("consecutive revisit shift (cm)")
    legs = fig.add_subplot(grid[1, 2])
    leg_types = [f"{STATIONS[a]} -> {STATIONS[b]}" for a, b in ((0, 1), (1, 0), (1, 2))]
    violins(legs, [draws[draws.leg_type == name].leg_m for name in leg_types], ["S -> P", "P -> S", "P -> B"], COLORS)
    legs.set_title("Transport legs between targets"); legs.set_ylabel("straight-line distance (m)")
    yaw = fig.add_subplot(grid[2, 0])
    violins(yaw, [draws[draws.visit_label == label].yaw_offset_deg for label in labels], labels, visit_colors)
    yaw.axhline(5, ls="--", color="gray"); yaw.axhline(-5, ls="--", color="gray")
    yaw.set_title("Each station draw is +/-5 degrees"); yaw.set_ylabel("heading offset (deg)")
    heading = fig.add_subplot(grid[2, 1:])
    for _, g in draws[draws.K == 3].groupby("seed"):
        heading.plot(g.visit_index, g.yaw_offset_deg, color=COLORS[0], alpha=0.1, lw=0.7)
    for k, color in zip((1, 2, 3), COLORS):
        values = draws[draws.K == k].pivot(index="seed", columns="visit_index", values="yaw_offset_deg")
        if values.empty:
            continue
        x = values.columns.to_numpy()
        heading.fill_between(x, np.percentile(values, 5, axis=0), np.percentile(values, 95, axis=0), color=color, alpha=0.12)
        heading.plot(x, np.median(values, axis=0), "o-", color=color, label=f"K={k}, n={len(values)}")
    heading.axhline(5, ls="--", color="gray"); heading.axhline(-5, ls="--", color="gray")
    heading.set_ylim(-6, 6); heading.set_xticks(range(7))
    heading.set_xlabel("navigation visit index (button at 2K)")
    heading.set_ylabel("heading vs calibrated station (deg)")
    heading.set_title("Absolute station headings: bounded noise, no accumulation")
    heading.legend(frameon=False)
    fig.suptitle(f"TS01 base randomization: seeds 0-{len(episodes) - 1}", fontsize=18, fontweight="bold")
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--task-config-file", type=Path, default=ROOT / "bench/tasks/ts01/resources/task.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/analysis/ts01_station_circles")
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    draws, episodes, payload = collect(args.count, args.task_config_file)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"ts01_{args.count}seed"
    draws.to_csv(out / f"{prefix}_station_draws.csv", index=False)
    episodes.to_csv(out / f"{prefix}_episode_routes.csv", index=False)
    summary = summarize(draws, episodes, payload)
    summary["config_sha256"] = hashlib.sha256(args.task_config_file.read_bytes()).hexdigest()
    (out / "ts01_station_circle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "stations_snapshot.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "task_config_snapshot.yaml").write_bytes(args.task_config_file.read_bytes())
    (out / "meta.json").write_text(json.dumps({
        "n_seeds": args.count, "seed_range": [0, args.count - 1],
        "data_kind": summary["data_kind"], "csv": str(out / f"{prefix}_station_draws.csv"),
        "analysis_reference": "outputs/analysis/wa01_station_circles/wa01_200seed_temporal_distribution.png",
    }, indent=2), encoding="utf-8")
    plot(draws, episodes, payload, out / f"{prefix}_temporal_distribution")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
