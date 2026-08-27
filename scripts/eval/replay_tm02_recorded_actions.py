#!/usr/bin/env python3
"""Replay TM02 native actions without expert handlers and save camera videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.recording.replay_validation import load_recorded_actions
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "outputs/debug/tm02_all_routes/train/TM02",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/debug/tm02_action_replay_videos",
    )
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Append one video frame every N action steps (actions are always replayed).",
    )
    return parser.parse_args()


def _episode_dir(input_root: Path, seed: int) -> Path:
    return input_root / f"seed_{seed:06d}_attempt_00"


def _write_frame(writers: dict[str, Any], observation: dict[str, Any]) -> None:
    for camera, writer in writers.items():
        frame = np.asarray(observation[f"{camera}_image"], dtype=np.uint8)
        writer.write(frame[:, :, ::-1])


def write_manifest(output_root: Path) -> dict[str, Any]:
    results = []
    for path in sorted(output_root.glob("seed_*/replay.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    manifest = {
        "task_id": "TM02",
        "mode": "recorded_actions_action_only_physics",
        "expert_handlers": False,
        "results": results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def replay_seed(
    *,
    seed: int,
    input_root: Path,
    output_root: Path,
    width: int,
    height: int,
    fps: int,
    frame_stride: int,
) -> dict[str, Any]:
    source_dir = _episode_dir(input_root, seed)
    trajectory = source_dir / "trajectory.h5"
    if not trajectory.is_file():
        raise FileNotFoundError(f"missing TM02 trajectory: {trajectory}")

    actions = load_recorded_actions(trajectory)
    episode = TaskCatalog.resolve(
        EpisodeRequest(
            task_id="TM02",
            seed=seed,
            split="train",
            env_split="pretrain",
            variant="memory",
            allow_partial_calibration=False,
        )
    )
    api = resolve_builtin_task_plugin("TM02").load_runtime_api()
    descriptor = api.build_descriptor(episode, True)
    runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(
        descriptor, expert=False
    )
    output_dir = output_root / f"seed_{seed:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    writers: dict[str, Any] = {}
    for camera in CAMERAS:
        writer = cv2.VideoWriter(
            str(output_dir / f"{camera}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {camera}")
        writers[camera] = writer
    consumed = 0
    success_step: int | None = None
    diagnostics: dict[str, Any] = {}
    try:
        for step, action in enumerate(actions, start=1):
            transition = runtime.step(action)
            consumed = step
            diagnostics = dict(transition.diagnostics.details or {})
            if bool(transition.diagnostics.success) and success_step is None:
                success_step = step
            if step % frame_stride == 0 or step == len(actions):
                _write_frame(writers, transition.observation)
            if step % 250 == 0:
                print(
                    f"[replay] seed={seed:06d} step={step}/{len(actions)} "
                    f"success_step={success_step}",
                    flush=True,
                )
    finally:
        for writer in writers.values():
            writer.release()
        runtime.close()

    result = {
        "task_id": "TM02",
        "seed": seed,
        "route": str(episode.facts["route"]),
        "episode_identity": episode.identity,
        "mode": "recorded_actions_action_only_physics",
        "expert_handlers": False,
        "recorded_steps": int(actions.shape[0]),
        "consumed_steps": consumed,
        "frame_stride": frame_stride,
        "success_step": success_step,
        "success": bool(diagnostics.get("success") is True),
        "failure_reason": None
        if diagnostics.get("success") is True
        else "recorded_actions_exhausted",
        "diagnostics": diagnostics,
        "videos": {
            camera: str(output_dir / f"{camera}.mp4") for camera in CAMERAS
        },
    }
    (output_dir / "replay.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seed or (0, 1))
    results = [
        replay_seed(
            seed=seed,
            input_root=args.input_root,
            output_root=args.output_root,
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            frame_stride=max(1, int(args.frame_stride)),
        )
        for seed in seeds
    ]
    manifest = write_manifest(args.output_root)
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if all(item["success"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
