#!/usr/bin/env python3
"""Replay native dataset trajectories through the production evaluator."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.artifacts import ArtifactWriter  # noqa: E402
from bench.evaluation.replay import (  # noqa: E402
    DATASET_REPLAY_MODE,
    ReplayEpisode,
    canonical_task_id,
    discover_recorded_episodes,
    load_recording_metadata,
    parse_seed_spec,
    replay_policy_contract,
    request_from_metadata,
    run_action_replay,
    suite_request_defaults,
    task_suite_path,
)
from bench.evaluation.suite import ArtifactConfig, load_eval_suite  # noqa: E402
from bench.recording.replay_validation import load_recorded_actions  # noqa: E402
from bench.tasks import get_task_definition  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--seeds", default=None, help="e.g. '0-9' or '0,5,10'")
    parser.add_argument(
        "--source-status",
        choices=("all", "success", "failed"),
        default="all",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-cameras",
        default="agentview_left,agentview_right,eye_in_hand",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--task-config-file", type=Path, default=None)
    parser.add_argument("--observation-profile-file", type=Path, default=None)
    parser.add_argument(
        "--require-episode-identity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    task_id = canonical_task_id(args.task_id)
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")
    suite = load_eval_suite(task_suite_path(task_id))
    discovered = discover_recorded_episodes(
        dataset_root,
        task_id,
        parse_seed_spec(args.seeds),
        source_status=args.source_status,
    )
    if not discovered:
        raise FileNotFoundError(
            f"no matching {task_id} trajectory.h5 files under {dataset_root}"
        )
    episodes: list[ReplayEpisode] = []
    for item in discovered:
        seed = int(item["seed"])
        episode_dir = Path(item["episode_dir"])
        actions = load_recorded_actions(episode_dir / "trajectory.h5")
        if args.max_steps is not None:
            actions = actions[: args.max_steps]
        metadata = load_recording_metadata(episode_dir)
        defaults = replace(suite_request_defaults(suite, seed), split="train")
        request, identity = request_from_metadata(
            task_id=task_id,
            seed=seed,
            metadata=metadata,
            defaults=defaults,
            task_config_file=args.task_config_file,
            observation_profile_file=args.observation_profile_file,
        )
        identity = identity or item.get("episode_identity")
        episodes.append(
            ReplayEpisode(
                seed=seed,
                actions=actions,
                request=request,
                expected_identity=None if identity is None else str(identity),
                source={
                    "kind": "native_dataset",
                    "dataset_root": str(dataset_root),
                    "episode_dir": str(episode_dir),
                    "attempt": int(item.get("attempt", 0)),
                    "source_status": item.get("source_status"),
                    "recorded_step_count": metadata.get("recorded_step_count"),
                    "recorded_termination": metadata.get("recorded_termination"),
                },
            )
        )

    cameras = _camera_aliases(args.video_cameras)
    if args.record_video and not cameras:
        raise ValueError("--record-video requires at least one camera")
    suite = replace(
        suite,
        seeds=tuple(episode.seed for episode in episodes),
        artifacts=ArtifactConfig(
            record_video=bool(args.record_video),
            video_cameras=cameras if args.record_video else (),
            video_fps=args.video_fps,
        ),
    )
    contract = replay_policy_contract(horizon=suite.rollout.replan_steps)
    writer = ArtifactWriter(output, expected_seeds=suite.seeds)
    writer.write_run_manifest(
        {
            "schema_version": "membench.run_manifest/v1",
            "run_id": output.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "suite": {
                "path": str(suite.source_path),
                "hash": suite.content_hash,
                "version": suite.suite_version,
            },
            "task": {
                "id": task_id,
                "env_name": get_task_definition(task_id).env_name,
                "env_split": suite.env_split,
                "options": dict(suite.task_options),
            },
            "seeds": list(suite.seeds),
            "policy_contract": contract.to_hello(),
            "replay": {
                "mode": DATASET_REPLAY_MODE,
                "dataset_root": str(dataset_root),
                "source_status": args.source_status,
                "action_source": "trajectory.h5[data/demo_1/actions]",
                "episode_identity_required": bool(args.require_episode_identity),
            },
            "rollout": {
                "max_steps": max(len(episode.actions) for episode in episodes),
                "replan_steps": suite.rollout.replan_steps,
                "action_repeat": 1,
                "stop_on_success": False,
                "ignore_done": True,
            },
            "artifacts": {
                "record_video": suite.artifacts.record_video,
                "video_cameras": list(suite.artifacts.video_cameras),
                "video_fps": suite.artifacts.video_fps,
            },
        }
    )
    run_action_replay(
        episodes,
        suite=suite,
        contract=contract,
        writer=writer,
        require_identity=bool(args.require_episode_identity),
    )
    summary = writer.write_summary()
    print(f"run_dir={writer.run_dir}")
    print(f"summary={json.dumps(summary, sort_keys=True)}")
    return 2 if summary.get("counts", {}).get("error", 0) else 0


def _camera_aliases(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )


if __name__ == "__main__":
    raise SystemExit(main())
