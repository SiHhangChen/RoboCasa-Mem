#!/usr/bin/env python3
"""Replay native WA-01 recordings in the current RoboCasa runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.rollout import EpisodeResult
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


EVALUATOR_PROFILE = "recorded-expert-replay-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="WA-01")
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "outputs" / "datasets" / "WA01"
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=ROOT / "outputs" / "debug" / "wa01-recorded-replay",
    )
    parser.add_argument("--seeds", default=None, help="e.g. '0-3' or '0,5,10'")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-cameras",
        default="agentview_left,agentview_right,eye_in_hand",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--task-config-file", type=Path, default=None)
    parser.add_argument("--observation-profile-file", type=Path, default=None)
    return parser.parse_args()


def parse_seeds(value: str | None) -> list[int]:
    if value is None:
        return []
    result: list[int] = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(item))
    return list(dict.fromkeys(result))


def discover_episodes(
    dataset_root: Path, seed_filter: list[int], task_id: str
) -> list[dict]:
    episodes: list[dict] = []
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("items", []):
            if item.get("task_id") != task_id:
                continue
            seed = int(item["seed"])
            if seed_filter and seed not in seed_filter:
                continue
            episode_dir = Path(item["episode_dir"])
            if not (episode_dir / "trajectory.h5").is_file():
                episode_dir = (
                    dataset_root
                    / "train"
                    / task_id
                    / f"seed_{seed:06d}_attempt_{int(item.get('attempt', 0)):02d}"
                )
            episodes.append({"seed": seed, "episode_dir": episode_dir})
    if not episodes:
        for directory in sorted((dataset_root / "train" / task_id).glob("seed_*_attempt_*")):
            if not (directory / "trajectory.h5").is_file():
                continue
            seed = int(directory.name.removeprefix("seed_").split("_")[0])
            if seed_filter and seed not in seed_filter:
                continue
            episodes.append({"seed": seed, "episode_dir": directory})
    return sorted(episodes, key=lambda item: item["seed"])


def load_actions(episode_dir: Path) -> np.ndarray:
    with h5py.File(episode_dir / "trajectory.h5", "r") as handle:
        actions = np.asarray(handle["data/demo_1/actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 13 or not np.all(np.isfinite(actions)):
        raise ValueError(f"invalid WA-01 action array: {actions.shape}")
    return np.ascontiguousarray(actions)


def load_recording_metadata(episode_dir: Path) -> dict:
    path = episode_dir / "episode.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    return {
        "episode_identity": metadata.get("episode_identity"),
        "initialization": metadata.get("initialization"),
        "recorded_step_count": payload.get("step_count"),
        "recorded_termination": payload.get("termination"),
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    cameras = tuple(
        dict.fromkeys(item.strip() for item in args.video_cameras.split(",") if item.strip())
    )
    if args.record_video and not cameras:
        raise ValueError("--record-video requires at least one camera")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    task_id = str(args.task_id).strip()
    episodes = discover_episodes(dataset_root, parse_seeds(args.seeds), task_id)
    if not episodes:
        raise FileNotFoundError(f"no recorded {task_id} episodes under {dataset_root}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")

    definition = get_task_definition(task_id)
    runtime_api = resolve_builtin_task_plugin(task_id).load_runtime_api()
    if runtime_api is None:
        raise LookupError(f"{task_id} has no canonical runtime API")
    seeds = tuple(item["seed"] for item in episodes)
    writer = ArtifactWriter(output, expected_seeds=seeds)
    writer.write_run_manifest(
        {
            "schema_version": "membench.run_manifest/v1",
            "run_id": output.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": {
                "id": task_id,
                "env_name": definition.env_name,
                "env_split": "pretrain",
                "task_version": definition.task_version,
            },
            "seeds": list(seeds),
            "dataset_root": str(dataset_root),
            "replay": {
                "source_run": str(dataset_root),
                "mode": "recorded_actions_environment_rescore",
                "evaluator_profile": EVALUATOR_PROFILE,
                "action_source": "trajectory.h5[data/demo_1/actions]",
            },
            "rollout": {
                "action_repeat": 1,
                "stop_on_success": False,
                "full_horizon_terminal_validation": True,
                "max_steps": 5000,
            },
            "artifacts": {
                "record_video": bool(args.record_video),
                "video_cameras": list(cameras) if args.record_video else [],
                "video_fps": args.video_fps,
            },
        }
    )

    for index, item in enumerate(episodes, start=1):
        seed = item["seed"]
        print(f"[replay] [{index}/{len(episodes)}] seed={seed:06d}", flush=True)
        actions = load_actions(item["episode_dir"])
        recorded = load_recording_metadata(item["episode_dir"])
        episode = definition.resolve(
            EpisodeRequest(
                task_id=task_id,
                seed=seed,
                variant="memory",
                split="train",
                env_split="pretrain",
                task_config_file=args.task_config_file,
                observation_profile_file=args.observation_profile_file,
                allow_partial_calibration=False,
            )
        )
        descriptor = runtime_api.build_descriptor(episode, execute=True)
        runtime = RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
            descriptor, expert=False
        )
        session = writer.start_episode(
            seed=seed,
            record_video=args.record_video,
            video_cameras=cameras if args.record_video else (),
            video_fps=args.video_fps,
        )
        started = time.monotonic()
        success_step = None
        diagnostics: dict = {}
        max_completion_button_displacement_m = 0.0
        max_completion_button_displacement_step = None
        consumed = 0
        try:
            for step, action in enumerate(actions, start=1):
                transition = runtime.step(action)
                displacement_getter = getattr(
                    runtime.env, "completion_button_displacement", None
                )
                if callable(displacement_getter):
                    displacement = float(displacement_getter())
                    if displacement > max_completion_button_displacement_m:
                        max_completion_button_displacement_m = displacement
                        max_completion_button_displacement_step = step
                session.append_action(action)
                if args.record_video:
                    frames = {
                        alias: np.ascontiguousarray(
                            transition.observation[
                                descriptor.sensors.observation_key(alias)
                            ]
                        )
                        for alias in cameras
                    }
                    session.append_frames(frames)
                consumed = step
                diagnostics = dict(transition.diagnostics.details or {})
                if transition.diagnostics.success and success_step is None:
                    success_step = step
            success = bool(diagnostics.get("success") is True)
            used = np.asarray(actions[:consumed], dtype=np.float32)
            checks = diagnostics.get("success_checks") or {}
            if callable(getattr(runtime.env, "completion_button_displacement", None)):
                diagnostics["replay_button_motion"] = {
                    "max_displacement_m": max_completion_button_displacement_m,
                    "max_displacement_step": max_completion_button_displacement_step,
                }
            failed_checks = [name for name, passed in checks.items() if not bool(passed)]
            result = EpisodeResult(
                seed=seed,
                episode_id=f"{task_id.lower()}-seed-{seed:06d}",
                task_id=task_id,
                suite_version=EVALUATOR_PROFILE,
                status="success" if success else "failure",
                success=success,
                success_step=success_step,
                env_steps=consumed,
                policy_calls=0,
                reward_sum=0.0,
                failure_reason=None if success else (
                    ",".join(failed_checks) or "recorded_actions_exhausted"
                ),
                diagnostics=diagnostics,
                action_mean=used.mean(axis=0).tolist(),
                action_std=used.std(axis=0).tolist(),
                action_min=used.min(axis=0).tolist(),
                action_max=used.max(axis=0).tolist(),
                policy_actions_returned=0,
                policy_actions_consumed=consumed,
                duration_s=time.monotonic() - started,
                episode_metadata={
                    **(recorded.get("initialization") or {}),
                    "recorded_step_count": recorded.get("recorded_step_count"),
                    "recorded_termination": recorded.get("recorded_termination"),
                    "episode_identity": recorded.get("episode_identity"),
                    "replayed_env_steps": consumed,
                },
            )
            session.finish(asdict(result))
            writer.write_summary()
            print(
                f"[replay] seed={seed:06d} {'SUCCESS' if success else 'FAILURE'} "
                f"steps={consumed} reason={result.failure_reason}",
                flush=True,
            )
        except BaseException:
            session.abort()
            raise
        finally:
            runtime.close()

    summary = writer.write_summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"run_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
