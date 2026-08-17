#!/usr/bin/env python
"""Apply task subtask definitions to replayed stage boundaries -> per-frame labels.

Reads ``boundaries.jsonl`` from :mod:`replay_stage_boundaries` and a task
subtask config (configs/memer/<task>_subtasks.json), then produces a per-frame
subtask label sequence plus memory keyframes for every episode.

Inputs:
  --config       task subtask config (JSON)
  --boundaries   boundaries.jsonl from replay_stage_boundaries.py
  --episodes-root native episode directory (train/<TASK_ID>/), used to read the
                  recorded total frame count so labels cover the real episode
  --output       subtasks.jsonl (one JSON object per line)

Output record per episode:
  {"task_id", "seed", "total_frames",
   "subtask_indices": [0,0,0,...],        # per-frame subtask key
   "subtask_names":  {"0": "add one ..."}, # key -> label text
   "keyframes": [ <absolute frame>, ... ]  # one per subtask segment per rule
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    prefixes: dict[str, list[str]] = {}
    for key, plist in config["stage_prefixes"].items():
        prefixes[key] = [str(p) for p in plist]
    return {
        "task_id": str(config["task_id"]),
        "high_level_instruction": config.get("high_level_instruction", ""),
        "subtasks": config["subtasks"],
        "stage_prefixes": prefixes,
    }


def match_subtask(stage_name: str, prefixes: dict[str, list[str]]) -> str | None:
    """Longest-prefix-first match of a stage name against configured prefixes."""
    candidates: list[tuple[int, str]] = []
    for key, plist in prefixes.items():
        for prefix in plist:
            if stage_name.startswith(prefix):
                candidates.append((len(prefix), key))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def recorded_total_frames(episode_dir: Path) -> int:
    episode_path = episode_dir / "episode.json"
    if episode_path.exists():
        data = json.loads(episode_path.read_text(encoding="utf-8"))
        step_count = data.get("step_count")
        if isinstance(step_count, int):
            return step_count
    h5 = episode_dir / "trajectory.h5"
    if h5.exists():
        import h5py
        with h5py.File(h5, "r") as fp:
            demo = next(iter(fp["data"].values()))
            return len(demo["actions"])
    raise FileNotFoundError(f"no episode.json or trajectory.h5 under {episode_dir}")


def label_episode(record: dict, config: dict, recorded_total: int) -> dict:
    subtask_order: list[str] = list(config["subtasks"].keys())
    stage_subs: list[str] = []
    for stage in record["stages"]:
        sub = match_subtask(str(stage["name"]), config["stage_prefixes"])
        if sub is None:
            raise ValueError(
                f"task {record['task_id']} seed {record['seed']}: "
                f"stage '{stage['name']}' matched no subtask prefix"
            )
        stage_subs.append(sub)

    # per-frame labels from stage ranges
    labels: list[str] = []
    prev_end = 0
    last_sub = stage_subs[-1] if stage_subs else None
    for stage, sub in zip(record["stages"], stage_subs):
        end = int(stage["end_frame"])
        for _ in range(prev_end, end):
            labels.append(sub)
        prev_end = end
    # align exactly to the recorded episode length (replay may drift a few
    # frames in either direction; both truncate and extend are needed)
    if len(labels) < recorded_total and last_sub is not None:
        while len(labels) < recorded_total:
            labels.append(last_sub)
    elif len(labels) > recorded_total:
        labels = labels[:recorded_total]

    # memory keyframes: one per contiguous subtask segment, per its rule
    keyframes: list[int] = []
    if labels:
        seg_start = 0
        seg_sub = labels[0]
        for i in range(1, len(labels) + 1):
            changed = i == len(labels) or labels[i] != seg_sub
            if changed:
                rule = config["subtasks"][seg_sub].get("keyframe", "none")
                if rule == "first":
                    keyframes.append(seg_start)
                elif rule == "last":
                    keyframes.append(i - 1)
                if i < len(labels):
                    seg_start = i
                    seg_sub = labels[i]

    name_map = {str(k): config["subtasks"][k]["label"] for k in subtask_order}
    index_map = {k: i for i, k in enumerate(subtask_order)}
    return {
        "task_id": record["task_id"],
        "seed": record["seed"],
        "total_frames": len(labels),
        "subtask_indices": [index_map[lab] for lab in labels],
        "subtask_names": name_map,
        "subtask_order": [config["subtasks"][k]["label"] for k in subtask_order],
        "keyframes": keyframes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=None, help="optional comma-separated filter")
    parser.add_argument("--task-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = {int(s) for s in args.seeds.split(",")} if args.seeds else None

    records = [
        json.loads(line)
        for line in args.boundaries.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [r for r in records if r.get("task_id") == (args.task_id or config["task_id"])]
    if seeds is not None:
        records = [r for r in records if r.get("seed") in seeds]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.output.open("w", encoding="utf-8") as fp:
        for record in sorted(records, key=lambda r: r.get("seed", 0)):
            episode_dir = args.episodes_root / f"seed_{record['seed']:06d}_attempt_00"
            if not episode_dir.exists():
                print(f"WARN: missing episode dir {episode_dir}, skipping", file=sys.stderr)
                continue
            recorded_total = recorded_total_frames(episode_dir)
            out = label_episode(record, config, recorded_total)
            fp.write(json.dumps(out, sort_keys=True) + "\n")
            n_written += 1

    print(f"wrote {n_written} labeled episodes to {args.output}")
    if n_written:
        with args.output.open("r", encoding="utf-8") as fp:
            sample = json.loads(next(line for line in fp if line.strip()))
        subtask_order: list[str] = list(config["subtasks"].keys())
        idx_to_label = {i: config["subtasks"][k]["label"] for i, k in enumerate(subtask_order)}
        counts: dict[str, int] = {}
        for idx in sample["subtask_indices"]:
            label = idx_to_label[idx]
            counts[label] = counts.get(label, 0) + 1
        print("seed", sample["seed"], "total", sample["total_frames"], "keyframes", sample["keyframes"])
        print("segment frame counts:", json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
