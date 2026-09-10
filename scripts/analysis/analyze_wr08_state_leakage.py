#!/usr/bin/env python3
"""State-only leakage audit for the recorded WR-08 campaign.

Labels and stage boundaries are read from native metadata, while every model
feature is loaded exclusively from ``trajectory.h5`` ``obs/robot_state``.
The audit treats the target slot for each transport as the answer-bearing
label and also reports source-slot, color, and transport-order diagnostics.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.runtime.data_contract import STATE_FIELDS
from scripts.annotate.planner_to_stage_boundaries import count_recorded_steps


EPISODE_RE = re.compile(r"^seed_(\d{6})_attempt_00$")
STATE_NAMES = [
    "arm_joint_qpos_0", "arm_joint_qpos_1", "arm_joint_qpos_2",
    "arm_joint_qpos_3", "arm_joint_qpos_4", "arm_joint_qpos_5",
    "arm_joint_qpos_6", "arm_joint_qvel_0", "arm_joint_qvel_1",
    "arm_joint_qvel_2", "arm_joint_qvel_3", "arm_joint_qvel_4",
    "arm_joint_qvel_5", "arm_joint_qvel_6", "gripper_qpos_0",
    "gripper_qpos_1", "gripper_qvel_0", "gripper_qvel_1",
    "base_pos_world_0", "base_pos_world_1", "base_pos_world_2",
    "base_quat_world_x", "base_quat_world_y", "base_quat_world_z",
    "base_quat_world_w", "base_joint_qvel_0", "base_joint_qvel_1",
    "base_joint_qvel_2", "torso_qpos", "torso_qvel", "eef_pos_in_base_0",
    "eef_pos_in_base_1", "eef_pos_in_base_2", "eef_quat_in_base_x",
    "eef_quat_in_base_y", "eef_quat_in_base_z", "eef_quat_in_base_w",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def frame_boundaries(episode_dir: Path) -> tuple[dict[str, Any], list[int], int]:
    episode = read_json(episode_dir / "episode.json")
    planner = read_json(episode_dir / "planner.json")
    stages = planner.get("rollout", {}).get("stages")
    require(isinstance(stages, list) and len(stages) == 17, f"{episode_dir}: expected 17 stages")
    cumulative = 0
    starts: list[int] = []
    for index, stage in enumerate(stages):
        require(isinstance(stage, Mapping), f"{episode_dir}: invalid stage {index}")
        starts.append(cumulative)
        steps, _ = count_recorded_steps(stage, path=f"{episode_dir.name}.stage[{index}]")
        cumulative += steps
    require(cumulative == int(episode["step_count"]), f"{episode_dir}: stage/frame mismatch")
    metadata = episode.get("metadata", {})
    expert = metadata.get("expert_plan", {}) if isinstance(metadata, Mapping) else {}
    labels = expert.get("metadata", {}) if isinstance(expert, Mapping) else {}
    require(isinstance(labels, Mapping), f"{episode_dir}: missing expert metadata")
    return dict(labels), starts, cumulative


def collect(root: Path) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    records: list[dict[str, Any]] = []
    states: dict[int, np.ndarray] = {}
    for episode_dir in sorted(root.glob("seed_*_attempt_00")):
        if not episode_dir.is_dir() or EPISODE_RE.fullmatch(episode_dir.name) is None:
            continue
        match = EPISODE_RE.fullmatch(episode_dir.name)
        assert match is not None
        seed = int(match.group(1))
        labels, starts, total = frame_boundaries(episode_dir)
        order = list(labels["transport_order"])
        colors = dict(labels["source_color_by_item"])
        source_slots = dict(labels["source_slot_by_item"])
        target_by_color = dict(labels["target_slot_by_color"])
        target_slots = [target_by_color[colors[item]] for item in order]
        source_index = {"source_left": 0, "source_center": 1, "source_right": 2}
        target_index = {"target_left": 0, "target_center": 1, "target_right": 2}
        target_minus_source = (target_index[target_slots[0]] - source_index[source_slots[order[0]]]) % 3
        require(target_minus_source in (1, 2), f"{episode_dir}: target layout must be cyclic")
        state_path = episode_dir / "trajectory.h5"
        with h5py.File(state_path, "r") as handle:
            state = np.asarray(handle["/data/demo_1/obs/robot_state"], dtype=np.float64)
        require(state.shape == (total, 37), f"{state_path}: expected [{total},37], got {state.shape}")
        require(bool(np.all(np.isfinite(state))), f"{state_path}: non-finite state")
        states[seed] = state
        records.append({
            "seed": seed,
            "episode_dir": str(episode_dir),
            "starts": starts,
            "total_frames": total,
            "first_item": order[0], "second_item": order[1],
            "first_source_slot": source_slots[order[0]],
            "second_source_slot": source_slots[order[1]],
            "first_color": colors[order[0]], "second_color": colors[order[1]],
            "first_target_slot": target_slots[0], "second_target_slot": target_slots[1],
            "target_layout_shift": f"target_minus_source_{target_minus_source}",
        })
    records.sort(key=lambda row: int(row["seed"]))
    require(len(records) == 200, f"expected 200 WR-08 episodes, found {len(records)}")
    require([row["seed"] for row in records] == list(range(200)), "expected contiguous seeds 0..199")
    return records, states


def binary_threshold(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    cuts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1, len(xs)]
    one = np.r_[0, np.cumsum(ys == 1)]
    zero = np.r_[0, np.cumsum(ys == 0)]
    total_one, total_zero = int(one[-1]), int(zero[-1])
    best = max(int(zero[c] + total_one - one[c]) for c in cuts)
    best = max(best, *(int(one[c] + total_zero - zero[c]) for c in cuts))
    return float(best / len(y))


def three_segment(x: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    cuts = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1, len(xs)]
    prefix = np.zeros((len(xs) + 1, n_classes), dtype=np.int64)
    for cls in range(n_classes):
        prefix[1:, cls] = np.cumsum(ys == cls)
    best = 0
    for right_index, right in enumerate(cuts):
        left = cuts[:right_index + 1]
        scores = prefix[left].max(axis=1) + (prefix[right] - prefix[left]).max(axis=1)
        best = max(best, int(scores.max() + (prefix[-1] - prefix[right]).max()))
    return float(best / len(y))


def cv_linear_probe(X: np.ndarray, y: np.ndarray, folds: int = 10) -> dict[str, Any]:
    classes = np.unique(y)
    counts = [int(np.sum(y == cls)) for cls in classes]
    n_folds = min(folds, min(counts))
    if n_folds < 2:
        return {"available": False, "reason": "class with fewer than two samples"}
    rng = np.random.default_rng(20260907)
    fold_rows: list[list[int]] = [[] for _ in range(n_folds)]
    for cls in classes:
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)
        for i, index in enumerate(indices):
            fold_rows[i % n_folds].append(int(index))
    predictions = np.full(len(y), -1, dtype=np.int64)
    scores = []
    for test_values in fold_rows:
        test = np.asarray(sorted(test_values), dtype=np.int64)
        train = np.setdiff1d(np.arange(len(y)), test)
        mean, scale = X[train].mean(axis=0), X[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        train_X = np.c_[(X[train] - mean) / scale, np.ones(len(train))]
        test_X = np.c_[(X[test] - mean) / scale, np.ones(len(test))]
        targets = np.eye(len(classes), dtype=np.float64)[y[train]]
        penalty = np.eye(train_X.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(train_X.T @ train_X + penalty, train_X.T @ targets)
        pred = np.argmax(test_X @ weights, axis=1)
        predictions[test] = pred
        scores.append(float(np.mean(pred == y[test])))
    return {"available": True, "n_folds": n_folds, "fold_accuracies": scores,
            "mean_accuracy": float(np.mean(scores)), "overall_accuracy": float(np.mean(predictions == y))}


def point_metrics(X: np.ndarray, y: np.ndarray, labels: list[str]) -> dict[str, Any]:
    classes = np.arange(len(labels), dtype=np.int64)
    per_dim = []
    for index in range(X.shape[1]):
        if len(labels) == 2:
            accuracy = binary_threshold(X[:, index], y)
        elif len(labels) == 3:
            accuracy = three_segment(X[:, index], y, 3)
        else:
            accuracy = max(binary_threshold(X[:, index], (y == cls).astype(np.int64)) for cls in classes)
        per_dim.append({"index": index, "name": STATE_NAMES[index], "accuracy": accuracy})
    top = max(per_dim, key=lambda row: row["accuracy"])
    counts = {label: int(np.sum(y == i)) for i, label in enumerate(labels)}
    strict = []
    for index in range(X.shape[1]):
        ranges = sorted((float(X[y == cls, index].min()), float(X[y == cls, index].max())) for cls in classes)
        if all(ranges[i][1] < ranges[i + 1][0] for i in range(len(ranges) - 1)):
            strict.append(index)
    return {"n": len(y), "class_counts": counts,
            "majority_baseline": max(counts.values()) / len(y),
            "max_single_dimension_accuracy": top["accuracy"], "top_dimension": top,
            "strictly_separable_dimensions": strict,
            "cross_validated_ridge_linear_probe": cv_linear_probe(X, y)}


def permutation_probe(X: np.ndarray, y: np.ndarray, labels: list[str], n: int) -> dict[str, Any]:
    actual = point_metrics(X, y, labels)
    rng = np.random.default_rng(20260907)
    max_1d = np.empty(n, dtype=np.float64)
    linear = np.empty(n, dtype=np.float64)
    for index in range(n):
        shuffled = rng.permutation(y)
        metrics = point_metrics(X, shuffled, labels)
        max_1d[index] = metrics["max_single_dimension_accuracy"]
        linear[index] = metrics["cross_validated_ridge_linear_probe"]["overall_accuracy"]
    return {
        "n": n,
        "seed": 20260907,
        "actual_max_single_dimension_accuracy": actual["max_single_dimension_accuracy"],
        "max_single_dimension_q975": float(np.percentile(max_1d, 97.5)),
        "max_single_dimension_p_ge_plus1": float((np.sum(max_1d >= actual["max_single_dimension_accuracy"]) + 1) / (n + 1)),
        "actual_cv_linear_accuracy": actual["cross_validated_ridge_linear_probe"]["overall_accuracy"],
        "cv_linear_q975": float(np.percentile(linear, 97.5)),
        "cv_linear_p_ge_plus1": float((np.sum(linear >= actual["cross_validated_ridge_linear_probe"]["overall_accuracy"]) + 1) / (n + 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-causal-offset", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    records, states = collect(args.episodes_root.resolve())

    label_specs = {
        "first_target_slot": ["target_center", "target_left", "target_right"],
        "second_target_slot": ["target_center", "target_left", "target_right"],
        "first_source_slot": ["source_center", "source_left", "source_right"],
        "second_source_slot": ["source_center", "source_left", "source_right"],
        "transport_order": ["boxed_food_a_first", "boxed_food_b_first"],
        "target_layout_shift": ["target_minus_source_1", "target_minus_source_2"],
    }
    encoded: dict[str, np.ndarray] = {}
    for name, labels in label_specs.items():
        encoded[name] = np.asarray([labels.index({
            "first_target_slot": row["first_target_slot"], "second_target_slot": row["second_target_slot"],
            "first_source_slot": row["first_source_slot"], "second_source_slot": row["second_source_slot"],
            "transport_order": f"{row['first_item']}_first",
            "target_layout_shift": row["target_layout_shift"],
        }[name]) for row in records], dtype=np.int64)

    # Stages 5 and 13 navigate to the common target-plate station.  The state
    # at stage 6/14 starts the first target-slot-specific placement action, so
    # those are the decisive pre-action handoffs for causal leakage.
    point_stages = {"initial": 0, "after_first_source_observation": 1,
                    "before_first_pick": 2, "after_first_pick": 3,
                    "before_first_target_navigation": 5,
                    "first_target_action_handoff": 6, "after_first_place": 7,
                    "before_second_source_navigation": 9,
                    "before_second_pick": 10, "after_second_pick": 11,
                    "before_second_target_navigation": 13,
                    "second_target_action_handoff": 14,
                    "after_second_place": 15, "terminal": 16}
    output: dict[str, Any] = {
        "dataset_root": str(args.episodes_root.resolve()), "n_episodes": len(records),
        "state_dim": 37, "state_fields": list(STATE_FIELDS), "state_names": STATE_NAMES,
        "model_input": "native trajectory.h5 /data/demo_1/obs/robot_state only",
        "labels": {name: {"values": labels, "counts": {label: int(np.sum(encoded[name] == i)) for i, label in enumerate(labels)}}
                   for name, labels in label_specs.items()},
        "points": {}, "causal_scans": {}, "permutation_controls": {},
        "conditional_target_probes": {},
    }
    for point_name, stage_index in point_stages.items():
        X = np.vstack([states[row["seed"]][row["starts"][stage_index]] for row in records])
        output["points"][point_name] = {name: point_metrics(X, y, label_specs[name]) for name, y in encoded.items()}

    for name, stage_index in (("first_target_slot", 6), ("second_target_slot", 14)):
        handoff = np.asarray([row["starts"][stage_index] for row in records], dtype=np.int64)
        scan: dict[str, Any] = {}
        for offset in range(args.max_causal_offset + 1):
            valid = [i for i, row in enumerate(records) if handoff[i] + offset < row["total_frames"]]
            X = np.vstack([states[records[i]["seed"]][handoff[i] + offset] for i in valid])
            y = encoded[name][valid]
            labels = label_specs[name]
            per_dim = [three_segment(X[:, index], y, len(labels)) for index in range(X.shape[1])]
            top_index = int(np.argmax(per_dim))
            scan[str(offset)] = {
                "n": len(valid),
                "max_single_dimension_accuracy": per_dim[top_index],
                "top_dimension": {"index": top_index, "name": STATE_NAMES[top_index],
                                  "accuracy": per_dim[top_index]},
                "majority_baseline": float(np.bincount(y, minlength=len(labels)).max() / len(y)),
            }
        output["causal_scans"][name] = {"handoff_stage": stage_index, "offsets": scan}

    controls = {
        "initial_first_target": ("first_target_slot", 0),
        "before_first_target_navigation": ("first_target_slot", 5),
        "first_target_action_handoff": ("first_target_slot", 6),
        "initial_second_target": ("second_target_slot", 0),
        "before_second_target_navigation": ("second_target_slot", 13),
        "second_target_action_handoff": ("second_target_slot", 14),
        "initial_layout_shift": ("target_layout_shift", 0),
        "before_first_target_layout_shift": ("target_layout_shift", 5),
        "first_target_action_layout_shift": ("target_layout_shift", 6),
        "before_second_target_layout_shift": ("target_layout_shift", 13),
        "second_target_action_layout_shift": ("target_layout_shift", 14),
    }
    for control_name, (label_name, stage_index) in controls.items():
        X = np.vstack([states[row["seed"]][row["starts"][stage_index]] for row in records])
        output["permutation_controls"][control_name] = permutation_probe(
            X, encoded[label_name], label_specs[label_name], args.permutations
        )

    # If state reveals the picked source slot or which distinct carton is held,
    # an imbalanced finite campaign can make the target statistically
    # predictable without target choice being encoded causally.  Report target
    # probes within source-slot and item strata to separate those effects.
    for label_name, stage_index, prefix in (
        ("first_target_slot", 5, "first"),
        ("first_target_slot", 6, "first"),
        ("second_target_slot", 13, "second"),
        ("second_target_slot", 14, "second"),
    ):
        point_name = f"{prefix}_target_stage_{stage_index}"
        output["conditional_target_probes"][point_name] = {}
        for condition_name, condition_key in (("source_slot", f"{prefix}_source_slot"),
                                               ("item_identity", f"{prefix}_item")):
            rows: dict[str, Any] = {}
            for value in sorted({str(row[condition_key]) for row in records}):
                indices = [i for i, row in enumerate(records) if str(row[condition_key]) == value]
                raw_y = encoded[label_name][indices]
                present = sorted(np.unique(raw_y).tolist())
                if len(present) < 2:
                    continue
                remap = {raw: new for new, raw in enumerate(present)}
                yy = np.asarray([remap[int(raw)] for raw in raw_y], dtype=np.int64)
                subgroup_labels = [label_specs[label_name][raw] for raw in present]
                X = np.vstack([states[records[i]["seed"]][records[i]["starts"][stage_index]] for i in indices])
                rows[value] = point_metrics(X, yy, subgroup_labels)
            output["conditional_target_probes"][point_name][condition_name] = rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for point, rows in output["points"].items():
        for name in ("first_target_slot", "second_target_slot", "target_layout_shift", "transport_order"):
            m = rows[name]
            cv = m["cross_validated_ridge_linear_probe"].get("overall_accuracy")
            print(point, name, "1d", round(m["max_single_dimension_accuracy"], 4),
                  "baseline", round(m["majority_baseline"], 4), "cv", None if cv is None else round(cv, 4))
    for name, result in output["causal_scans"].items():
        rows = result["offsets"]
        first = next(((int(k), v) for k, v in rows.items() if v["max_single_dimension_accuracy"] >= 1.0 - 1e-12), None)
        print("causal", name, "first_perfect_1d", first)


if __name__ == "__main__":
    main()
