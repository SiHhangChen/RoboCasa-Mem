#!/usr/bin/env python3
"""State-only separability audit for TS04's two-drawer search memory."""

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


def stages_with_prefix(row: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    return [stage for stage in row["stages"] if stage["name"].startswith(prefix)]


def frame_points(records: list[dict[str, Any]]) -> dict[str, list[int | None]]:
    points: dict[str, list[int | None]] = {"initial": [0 for _ in records]}
    for ordinal in (1, 2):
        close_last: list[int | None] = []
        return_last: list[int | None] = []
        return_next: list[int | None] = []
        for row in records:
            closes = stages_with_prefix(row, "close_cabinet_slot_")
            returns = stages_with_prefix(row, "return_after_cabinet_slot_")
            if len(closes) != len(returns) or len(returns) != int(row["search_count"]):
                raise ValueError(f"seed {row['seed']}: close/return stage count mismatch")
            if len(returns) < ordinal:
                close_last.append(None)
                return_last.append(None)
                return_next.append(None)
            else:
                close_last.append(int(closes[ordinal - 1]["end_frame"]) - 1)
                return_last.append(int(returns[ordinal - 1]["end_frame"]) - 1)
                return_next.append(int(returns[ordinal - 1]["end_frame"]))
        points[f"after_{ordinal}_drawer_close"] = close_last
        points[f"after_{ordinal}_drawer_return"] = return_last
        points[f"{ordinal}_drawer_return_next_frame"] = return_next
    completion = []
    for row in records:
        stages = [
            stage for stage in row["stages"] if stage["name"] == "navigate_completion_button"
        ]
        if len(stages) != 1:
            raise ValueError(f"seed {row['seed']}: expected one completion navigation stage")
        completion.append(int(stages[0]["start_frame"]))
    points["completion_handoff"] = completion
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
    states: list[np.ndarray],
    frames: list[int],
    y: np.ndarray,
    n_perm: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected: dict[str, Any] = {}
    dense: dict[str, Any] = {}
    for offset in (0, 1, 2, 5, 10, 20, 40, 80, 120, 160):
        valid = [
            i
            for i, (state, frame) in enumerate(zip(states, frames))
            if frame + offset < len(state)
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
            for i, (state, frame) in enumerate(zip(states, frames))
            if frame + offset < len(state)
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
        raise ValueError(f"expected 200 unique TS04 boundary records, found {len(records)}")
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
    if set(search_count) != {1, 2}:
        raise ValueError(f"expected search_count 1/2, found {sorted(set(search_count))}")
    labels = (search_count == 2).astype(np.int8)
    points = frame_points(records)

    output: dict[str, Any] = {
        "dataset": str(args.raw_root),
        "n_episodes": len(records),
        "state_dim": 37,
        "state_names": STATE_NAMES,
        "label_definition": "search_count: 0=stop after first drawer, 1=continue to second drawer",
        "search_count_counts": {"1": int(np.sum(search_count == 1)), "2": int(np.sum(search_count == 2))},
        "target_fixture_counts": {
            str(name): int(sum(row["target_fixture"] == name for row in records))
            for name in sorted({row["target_fixture"] for row in records})
        },
        "fixture_order_counts": {
            "cabinet_slot_0_then_1": int(
                sum(row["full_fixture_order"] == ["cabinet_slot_0", "cabinet_slot_1"] for row in records)
            ),
            "cabinet_slot_1_then_0": int(
                sum(row["full_fixture_order"] == ["cabinet_slot_1", "cabinet_slot_0"] for row in records)
            ),
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
                "n": sum(frame is not None for frame in frames),
                "min": min(frame for frame in frames if frame is not None),
                "max": max(frame for frame in frames if frame is not None),
                "mean": float(np.mean([frame for frame in frames if frame is not None])),
            }
            for name, frames in points.items()
        },
        "points": {},
        "decision_handoff": {},
        "causal_offsets": {},
        "causal_dense_scan": {},
    }

    for name, frames in points.items():
        valid = [i for i, frame in enumerate(frames) if frame is not None]
        if not valid or name.startswith("after_2_") or name.startswith("2_"):
            continue
        X = np.vstack([states[i][int(frames[i])] for i in valid])
        output["points"][name] = {
            "n": len(valid),
            "first_30": analyze_frame_set(X, labels[valid], args.permutations)["first_30"],
            "full_37": analyze_frame_set(X, labels[valid], args.permutations)["full_37"],
        }

    handoff = points["after_1_drawer_return"]
    next_frame = points["1_drawer_return_next_frame"]
    valid = [i for i, frame in enumerate(handoff) if frame is not None]
    X = np.vstack([states[i][int(handoff[i])] for i in valid])
    y = labels[valid]
    X_next = np.vstack([states[i][int(next_frame[i])] for i in valid])
    output["decision_handoff"] = {
        "n": len(valid),
        "class_counts": {"stop": int(np.sum(y == 0)), "continue": int(np.sum(y == 1))},
        "frame_definition": "first return stage end_frame - 1",
        "first_30": analyze_frame_set(X, y, args.permutations)["first_30"],
        "full_37": analyze_frame_set(X, y, args.permutations)["full_37"],
        "next_frame": {
            "frame_definition": "first return stage end_frame",
            "first_30": analyze_frame_set(X_next, y, args.permutations)["first_30"],
            "full_37": analyze_frame_set(X_next, y, args.permutations)["full_37"],
        },
    }
    selected, dense = causal_analysis(states, handoff, labels, args.permutations)
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
