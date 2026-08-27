#!/usr/bin/env python
"""Refresh memory/keyframe references in an existing MemER train.json."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def align(frames: list[int], start: int, end: int, stride: int) -> list[int]:
    aligned = [
        ((frame - start) // stride) * stride + start
        for frame in frames
        if start <= frame < end
    ]
    return list(dict.fromkeys(aligned))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--subtasks", type=Path, required=True)
    parser.add_argument("--frame-subsample", type=int, required=True)
    parser.add_argument("--memory-length", type=int, required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.subtasks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keyframes = {
        int(record.get("episode_index", record["seed"])): [
            int(value) for value in record["keyframes"]
        ]
        for record in records
    }
    entries = json.loads(args.train_json.read_text(encoding="utf-8"))
    missing_images: set[str] = set()
    for entry in entries:
        metadata = entry["metadata"]
        episode = int(metadata["episode_index"])
        timestep = int(metadata["timestep"])
        recent = [int(value) for value in metadata["context_indices"]]
        if not recent:
            raise ValueError(f"{entry['id']}: empty recent context")
        start_offset = timestep % args.frame_subsample
        recent_start = recent[0]
        recent_keyframes = align(
            keyframes[episode], recent_start, timestep + 1, args.frame_subsample
        )
        positions = [recent.index(frame) + 1 for frame in recent_keyframes]
        memory = align(
            keyframes[episode], start_offset, recent_start, args.frame_subsample
        )[-args.memory_length :]
        metadata["memory_indices"] = memory
        answer = dict(metadata["answer"])
        answer["keyframe_positions"] = positions
        metadata["answer"] = answer
        entry["conversations"][2]["value"] = json.dumps(answer)
        trajectory = f"episode_{episode:06d}"
        entry["image"] = [
            f"media/{trajectory}/frame_{frame:06d}.jpg"
            for frame in memory + recent
        ]
        for relative in entry["image"]:
            if not (args.train_json.parent / relative).is_file():
                missing_images.add(relative)

    if missing_images:
        sample = sorted(missing_images)[:10]
        raise FileNotFoundError(
            f"{len(missing_images)} refreshed image references are missing: {sample}"
        )
    temporary = args.train_json.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.train_json)
    print(f"refreshed {len(entries)} SFT examples in {args.train_json}")


if __name__ == "__main__":
    main()
