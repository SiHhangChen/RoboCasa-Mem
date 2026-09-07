#!/usr/bin/env python3
"""State-only separability audit for TS02's drawer-search memory.

The policy-visible task is to search drawers for a red cube.  At each return
after closing a drawer, the expert must stop if the cube was found or continue
to the next drawer.  Boundary metadata locates these frames; state is loaded
directly from the raw LeRobot parquet dataset.
"""

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


def return_stages(row: dict[str, Any]) -> list[dict[str, Any]]:
    stages = [
        stage
        for stage in row["stages"]
        if stage["name"].startswith("return_after_cabinet_slot_")
    ]
    if len(stages) != int(row["search_count"]):
        raise ValueError(
            f"seed {row['seed']}: found {len(stages)} returns, expected {row['search_count']}"
        )
    return stages


def completion_stage(row: dict[str, Any]) -> dict[str, Any]:
    stages = [
        stage
        for stage in row["stages"]
        if stage["name"] == "navigate_completion_button"
    ]
    if len(stages) != 1:
        raise ValueError(f"seed {row['seed']}: expected one completion navigation stage")
    return stages[0]


def frame_points(records: list[dict[str, Any]]) -> dict[str, list[int | None]]:
    points: dict[str, list[int | None]] = {"initial": [0 for _ in records]}
    for ordinal in (1, 2, 3):
        return_last: list[int | None] = []
        return_next: list[int | None] = []
        close_last: list[int | None] = []
        for row in records:
            returns = return_stages(row)
            closes = [
                stage
                for stage in row["stages"]
                if stage["name"].startswith("close_cabinet_slot_")
            ]
            if len(closes) != int(row["search_count"]):
                raise ValueError(f"seed {row['seed']}: close/return count mismatch")
            if len(returns) < ordinal:
                return_last.append(None)
                return_next.append(None)
                close_last.append(None)
            else:
                return_last.append(int(returns[ordinal - 1]["end_frame"]) - 1)
                return_next.append(int(returns[ordinal - 1]["end_frame"]))
                close_last.append(int(closes[ordinal - 1]["end_frame"]) - 1)
        points[f"after_{ordinal}_drawer_close"] = close_last
        points[f"after_{ordinal}_drawer_return"] = return_last
        points[f"{ordinal}_drawer_return_next_frame"] = return_next
    points["completion_handoff"] = [
        int(completion_stage(row)["start_frame"]) for row in records
    ]
    return points


def analyze_frame_set(X: np.ndarray, labels: np.ndarray, n_perm: int) -> dict[str, Any]:
    return {
        "first_30": analyze_point(X[:, :30], labels, n_perm),
        "full_37": analyze_point(X, labels, n_perm),
    }


def diagnostic_summary(X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, XX in (("first_30", X[:, :30]), ("full_37", X)):
        metrics = dimension_metrics(XX, labels)
        top = max(metrics["dimensions"], key=lambda row: row["accuracy"])
        result[name] = {
            "max_single_dimension_accuracy": metrics["max_accuracy"],
            "top_dimension": {
                "index": top["index"],
                "name": top["name"],
                "accuracy": top["accuracy"],
            },
            "strictly_separable_dimensions": metrics["strictly_separable_indices"],
            "perfect_affine_linear_separation": perfect_linear_separation(XX, labels),
        }
    return result


def analyze_causal_scan(
    states: list[np.ndarray],
    frames: list[int],
    labels: np.ndarray,
    offsets: range,
    n_perm: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected: dict[str, Any] = {}
    dense: dict[str, Any] = {}
    for offset in (0, 1, 5, 10, 20, 40, 80, 120, 160):
        valid = [
            i
            for i, (state, frame) in enumerate(zip(states, frames))
            if frame + offset < len(state)
        ]
        X = np.vstack([states[i][frames[i] + offset] for i in valid])
        y = labels[valid]
        selected[str(offset)] = {
            "n": len(valid),
            "frame_offset": offset,
            "first_30": analyze_point(X[:, :30], y, n_perm),
            "full_37": analyze_point(X, y, n_perm),
        }
    for offset in offsets:
        valid = [
            i
            for i, (state, frame) in enumerate(zip(states, frames))
            if frame + offset < len(state)
        ]
        X = np.vstack([states[i][frames[i] + offset] for i in valid])
        dense[str(offset)] = diagnostic_summary(X, labels[valid])
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
        raise ValueError(f"expected 200 unique TS02 boundary records, found {len(records)}")
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
    search_count = np.asarray([int(row["search_count"]) for row in records], dtype=np.int8)
    if set(search_count) != {1, 2, 3}:
        raise ValueError(f"expected search_count 1/2/3, found {sorted(set(search_count))}")
    points = frame_points(records)

    output: dict[str, Any] = {
        "dataset": str(args.raw_root),
        "n_episodes": len(records),
        "state_dim": 37,
        "state_names": STATE_NAMES,
        "label_definition": "search_count: target drawer position; each handoff uses stop vs continue",
        "search_count_counts": {str(k): int(np.sum(search_count == k)) for k in (1, 2, 3)},
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
                "n": sum(frame is not None for frame in frames),
                "min": min(frame for frame in frames if frame is not None),
                "max": max(frame for frame in frames if frame is not None),
                "mean": float(np.mean([frame for frame in frames if frame is not None])),
            }
            for name, frames in points.items()
        },
        "points": {},
        "decision_handoffs": {},
        "causal_offsets": {},
        "causal_dense_scan": {},
    }

    for name, frames in points.items():
        if name in {"initial", "completion_handoff"} or name.startswith("after_1_") or name.startswith("1_"):
            point_labels = (search_count >= 2).astype(np.int8)
            eligible = np.ones(len(records), dtype=bool)
        elif name.startswith("after_2_") or name.startswith("2_"):
            point_labels = (search_count == 3).astype(np.int8)
            eligible = search_count >= 2
        else:
            # The third return is reached only by search_count=3 episodes and
            # has no remaining continue/stop contrast.
            continue
        valid = [
            i
            for i, frame in enumerate(frames)
            if frame is not None and eligible[i]
        ]
        if not valid:
            continue
        X = np.vstack([states[i][int(frames[i])] for i in valid])
        output["points"][name] = {
            "n": len(valid),
            "label_definition": "stop vs continue at this point",
            "first_30": analyze_frame_set(X, point_labels[valid], args.permutations)["first_30"],
            "full_37": analyze_frame_set(X, point_labels[valid], args.permutations)["full_37"],
        }

    decisions = {
        "after_first_drawer_return": {
            "frames": points["after_1_drawer_return"],
            "next_frames": points["1_drawer_return_next_frame"],
            "eligible": np.ones(len(records), dtype=bool),
            "labels": (search_count >= 2).astype(np.int8),
            "label_names": {"0": "stop (search_count=1)", "1": "continue (search_count>=2)"},
        },
        "after_second_drawer_return": {
            "frames": points["after_2_drawer_return"],
            "next_frames": points["2_drawer_return_next_frame"],
            "eligible": search_count >= 2,
            "labels": (search_count == 3).astype(np.int8),
            "label_names": {"0": "stop (search_count=2)", "1": "continue (search_count=3)"},
        },
    }
    for name, spec in decisions.items():
        eligible = np.asarray(spec["eligible"], dtype=bool)
        frames = spec["frames"]
        valid = [i for i in np.flatnonzero(eligible) if frames[i] is not None]
        X = np.vstack([states[i][int(frames[i])] for i in valid])
        y = np.asarray(spec["labels"], dtype=np.int8)[valid]
        output["decision_handoffs"][name] = {
            "n": len(valid),
            "label_names": spec["label_names"],
            "class_counts": {"stop": int(np.sum(y == 0)), "continue": int(np.sum(y == 1))},
            "frame_definition": "return stage end_frame - 1",
            "first_30": analyze_frame_set(X, y, args.permutations)["first_30"],
            "full_37": analyze_frame_set(X, y, args.permutations)["full_37"],
        }
        next_frames = spec["next_frames"]
        valid_next = [i for i in np.flatnonzero(eligible) if next_frames[i] is not None]
        X_next = np.vstack([states[i][int(next_frames[i])] for i in valid_next])
        y_next = np.asarray(spec["labels"], dtype=np.int8)[valid_next]
        output["decision_handoffs"][name]["next_frame"] = {
            "frame_definition": "return stage end_frame",
            "first_30": analyze_frame_set(X_next, y_next, args.permutations)["first_30"],
            "full_37": analyze_frame_set(X_next, y_next, args.permutations)["full_37"],
        }
        selected, dense = analyze_causal_scan(
            [states[i] for i in valid],
            [int(frames[i]) for i in valid],
            y,
            range(0, 81),
            args.permutations,
        )
        output["causal_offsets"][name] = selected
        output["causal_dense_scan"][name] = dense

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {args.output}")
    for name, result in output["decision_handoffs"].items():
        print("decision", name)
        for when, metrics in (("handoff", result), ("next_frame", result["next_frame"])):
            for dims in ("first_30", "full_37"):
                m = metrics[dims]
                print(
                    when,
                    dims,
                    "max",
                    m["max_single_dimension_accuracy"],
                    "strict",
                    m["strictly_separable_dimensions"],
                    "perm_median",
                    m["permutation"]["median"],
                    "linear",
                    m["perfect_affine_linear_separation"]["separable"],
                    "cv",
                    m["ten_fold_standardized_logistic_probe"]["mean_accuracy"],
                )
        for dims in ("first_30", "full_37"):
            first_strict = next(
                (
                    int(offset)
                    for offset, row in output["causal_dense_scan"][name].items()
                    if row[dims]["strictly_separable_dimensions"]
                ),
                None,
            )
            first_linear = next(
                (
                    int(offset)
                    for offset, row in output["causal_dense_scan"][name].items()
                    if row[dims]["perfect_affine_linear_separation"]["separable"]
                ),
                None,
            )
            print(dims, "first_strict", first_strict, "first_perfect_affine", first_linear)


if __name__ == "__main__":
    main()
