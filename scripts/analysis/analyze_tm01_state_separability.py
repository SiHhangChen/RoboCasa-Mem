#!/usr/bin/env python3
"""State-only separability audit for TM01's remembered task route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.analyze_wa02_state_separability import (
    STATE_NAMES,
    analyze_point,
    dimension_metrics,
    perfect_linear_separation,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_seed_mapping(raw_root: Path) -> dict[int, int]:
    rows = read_jsonl(raw_root / "meta" / "seeds.jsonl")
    mapping = {int(row["seed"]): int(row["episode_index"]) for row in rows}
    if len(mapping) != len(rows) or sorted(mapping.values()) != list(range(len(rows))):
        raise ValueError("raw seed mapping is not one-to-one over episode indices")
    return mapping


def load_raw_state(
    raw_root: Path,
    seed: int,
    expected_length: int,
    seed_to_episode: dict[int, int],
) -> np.ndarray:
    episode_index = seed_to_episode.get(seed)
    if episode_index is None:
        raise ValueError(f"seed {seed} is absent from raw meta/seeds.jsonl")
    path = raw_root / "data" / "chunk-000" / f"file-{episode_index:03d}.parquet"
    table = pq.read_table(path, columns=["observation.state"])
    values = table["observation.state"].combine_chunks().values.to_numpy()
    state = np.asarray(values, dtype=np.float64).reshape(-1, 37)
    if len(state) != expected_length:
        raise ValueError(
            f"seed {seed}: raw rows {len(state)} != boundary length {expected_length}"
        )
    return state


def stage(row: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in row["stages"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"seed {row['seed']}: expected one stage {name!r}")
    return matches[0]


def first_stage_with_prefix(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    matches = [item for item in row["stages"] if item["name"].startswith(prefix)]
    if not matches:
        raise ValueError(f"seed {row['seed']}: expected stage prefix {prefix!r}")
    return matches[0]


def frame_points(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    points: dict[str, list[int]] = {"initial": [0 for _ in records]}
    definitions = {
        "after_A1_return": "return_arm_after_A1",
        "after_B1_close": "close_drawer_",
        "after_B1_return": "return_arm_after_B1",
        "after_A2_return": "return_arm_after_A2",
        "after_B2_return": "return_arm_after_B2",
        "completion_handoff": "navigate_",
    }
    for row in records:
        points.setdefault("after_A1_return", []).append(
            int(stage(row, "return_arm_after_A1")["end_frame"]) - 1
        )
        points.setdefault("after_B1_close", []).append(
            int(first_stage_with_prefix(row, "close_drawer_")["end_frame"]) - 1
        )
        points.setdefault("after_B1_return", []).append(
            int(stage(row, "return_arm_after_B1")["end_frame"]) - 1
        )
        points.setdefault("after_A2_return", []).append(
            int(stage(row, "return_arm_after_A2")["end_frame"]) - 1
        )
        points.setdefault("after_B2_return", []).append(
            int(stage(row, "return_arm_after_B2")["end_frame"]) - 1
        )
        completion = [
            item
            for item in row["stages"]
            if item["name"].startswith("navigate_")
            and item["name"].endswith("_completion_button")
        ]
        if len(completion) != 1:
            raise ValueError(f"seed {row['seed']}: expected one completion navigation stage")
        points.setdefault("completion_handoff", []).append(int(completion[0]["start_frame"]))
    points["B1_return_next_frame"] = [
        int(stage(row, "return_arm_after_B1")["end_frame"]) for row in records
    ]
    return points


def analyze_frame_set(X: np.ndarray, y: np.ndarray, n_perm: int) -> dict[str, Any]:
    return {
        "first_30": analyze_point(X[:, :30], y, n_perm),
        "full_37": analyze_point(X, y, n_perm),
    }


def diagnostic_summary(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, XX in (("first_30", X[:, :30]), ("full_37", X)):
        metrics = dimension_metrics(XX, y)
        top = max(metrics["dimensions"], key=lambda row: row["accuracy"])
        result[name] = {
            "max_single_dimension_accuracy": metrics["max_accuracy"],
            "top_dimension": {
                "index": top["index"],
                "name": top["name"],
                "accuracy": top["accuracy"],
            },
            "strictly_separable_dimensions": metrics["strictly_separable_indices"],
            "perfect_affine_linear_separation": perfect_linear_separation(XX, y),
        }
    return result


def causal_analysis(
    states: list[np.ndarray], frames: list[int], y: np.ndarray, n_perm: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected: dict[str, Any] = {}
    dense: dict[str, Any] = {}
    for offset in (0, 1, 2, 5, 10, 20, 40, 80, 120, 160):
        valid = [
            i
            for i, (state_value, frame) in enumerate(zip(states, frames))
            if frame + offset < len(state_value)
        ]
        X = np.vstack([states[i][frames[i] + offset] for i in valid])
        yy = y[valid]
        selected[str(offset)] = {
            "n": len(valid),
            "frame_offset": offset,
            "first_30": analyze_point(X[:, :30], yy, n_perm),
            "full_37": analyze_point(X, yy, n_perm),
        }
    for offset in range(0, 81):
        valid = [
            i
            for i, (state_value, frame) in enumerate(zip(states, frames))
            if frame + offset < len(state_value)
        ]
        X = np.vstack([states[i][frames[i] + offset] for i in valid])
        dense[str(offset)] = diagnostic_summary(X, y[valid])
    return selected, dense


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args()

    records = sorted(read_jsonl(args.boundaries), key=lambda row: int(row["seed"]))
    if len(records) != 200 or len({int(row["seed"]) for row in records}) != 200:
        raise ValueError(f"expected 200 unique TM01 records, found {len(records)}")
    seed_to_episode = load_seed_mapping(args.raw_root)
    states_by_seed = {
        int(row["seed"]): load_raw_state(
            args.raw_root,
            int(row["seed"]),
            int(row["total_frames"]),
            seed_to_episode,
        )
        for row in records
    }
    states = [states_by_seed[int(row["seed"])] for row in records]
    route = np.asarray([0 if row["route"] == "L1" else 1 for row in records], dtype=np.int8)
    if set(route) != {0, 1}:
        raise ValueError("expected both TM01 routes L1 and L2")
    points = frame_points(records)

    output: dict[str, Any] = {
        "dataset": str(args.raw_root),
        "n_episodes": len(records),
        "state_dim": 37,
        "state_names": STATE_NAMES,
        "label_definition": "route: L1=0 (A2 then B2), L2=1 (B2 then A2)",
        "route_counts": {"L1": int(np.sum(route == 0)), "L2": int(np.sum(route == 1))},
        "drawer_order_counts": {
            "left_then_right": int(sum(row["drawer_order"] == ["drawer_left", "drawer_right"] for row in records)),
            "right_then_left": int(sum(row["drawer_order"] == ["drawer_right", "drawer_left"] for row in records)),
        },
        "completion_button_side_counts": {
            "drawer_side": int(sum(row["completion_button_side"] == "drawer_side" for row in records)),
            "stove_side": int(sum(row["completion_button_side"] == "stove_side" for row in records)),
        },
        "raw_seed_mapping": {
            "n": len(seed_to_episode),
            "min_seed": min(seed_to_episode),
            "max_seed": max(seed_to_episode),
            "missing_in_seed_range": sorted(
                set(range(max(seed_to_episode) + 1)) - set(seed_to_episode)
            ),
        },
        "boundary_frames": {
            name: {
                "n": len(frames),
                "min": min(frames),
                "max": max(frames),
                "mean": float(np.mean(frames)),
            }
            for name, frames in points.items()
        },
        "points": {},
        "decision_handoff": {},
        "causal_offsets": {},
        "causal_dense_scan": {},
    }

    for name, frames in points.items():
        X = np.vstack([states[i][int(frame)] for i, frame in enumerate(frames)])
        output["points"][name] = analyze_frame_set(X, route, args.permutations)

    handoff = points["after_B1_return"]
    next_frame = points["B1_return_next_frame"]
    X = np.vstack([states[i][frame] for i, frame in enumerate(handoff)])
    X_next = np.vstack([states[i][frame] for i, frame in enumerate(next_frame)])
    output["decision_handoff"] = {
        "n": len(route),
        "class_counts": {"L1": int(np.sum(route == 0)), "L2": int(np.sum(route == 1))},
        "frame_definition": "return_arm_after_B1 end_frame - 1",
        "first_30": analyze_frame_set(X, route, args.permutations)["first_30"],
        "full_37": analyze_frame_set(X, route, args.permutations)["full_37"],
        "next_frame": {
            "frame_definition": "return_arm_after_B1 end_frame",
            "first_30": analyze_frame_set(X_next, route, args.permutations)["first_30"],
            "full_37": analyze_frame_set(X_next, route, args.permutations)["full_37"],
        },
    }
    selected, dense = causal_analysis(states, handoff, route, args.permutations)
    output["causal_offsets"] = selected
    output["causal_dense_scan"] = dense

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for when, metrics in (("handoff", output["decision_handoff"]), ("next_frame", output["decision_handoff"]["next_frame"])):
        for dims in ("first_30", "full_37"):
            result = metrics[dims]
            print(
                when,
                dims,
                "max",
                result["max_single_dimension_accuracy"],
                "strict",
                result["strictly_separable_dimensions"],
                "perm_median",
                result["permutation"]["median"],
                "linear",
                result["perfect_affine_linear_separation"]["separable"],
                "cv",
                result["ten_fold_standardized_logistic_probe"]["mean_accuracy"],
            )
    for dims in ("first_30", "full_37"):
        first_strict = next(
            (int(offset) for offset, row in dense.items() if row[dims]["strictly_separable_dimensions"]),
            None,
        )
        print(dims, "first_strict_offset", first_strict)


if __name__ == "__main__":
    main()
