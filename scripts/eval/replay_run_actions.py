#!/usr/bin/env python3
"""Replay actions saved by a benchmark evaluation run.

The action file acts as an in-process policy. Environment stepping, task
evaluation, report stages, videos, and artifacts use the production evaluation
runner unchanged.
"""

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
    RUN_REPLAY_MODE,
    ReplayEpisode,
    canonical_task_id,
    load_json,
    load_npz_actions,
    load_source_suite,
    parse_seed_spec,
    replay_policy_contract,
    request_from_metadata,
    run_action_replay,
    suite_request_defaults,
)
from bench.evaluation.suite import ArtifactConfig  # noqa: E402
from bench.policy.protocol import PolicyContract  # noqa: E402
from bench.tasks import get_task_definition  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_run", type=Path)
    parser.add_argument("--task-id", default=None, help="infer from run manifest by default")
    parser.add_argument("--seeds", default=None, help="e.g. '1050-1053' or '1050,1060'")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-cameras",
        default="agentview_left,agentview_right,eye_in_hand",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--require-episode-identity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reject replay when current task resolution differs from the source run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    source = args.source_run.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")
    manifest = load_json(source / "run_manifest.json")
    task_value = args.task_id or _manifest_task_id(manifest)
    task_id = canonical_task_id(task_value)
    suite = load_source_suite(manifest, task_id)
    wanted = parse_seed_spec(args.seeds)
    episodes: list[ReplayEpisode] = []
    for action_path in sorted((source / "episodes").glob("seed_*/actions.npz")):
        seed = int(action_path.parent.name.removeprefix("seed_"))
        if wanted and seed not in wanted:
            continue
        actions = load_npz_actions(action_path)
        if args.max_steps is not None:
            actions = actions[: args.max_steps]
        result_path = action_path.parent / "result.json"
        source_result = load_json(result_path) if result_path.is_file() else {}
        metadata_value = source_result.get("episode_metadata", {})
        metadata = dict(metadata_value) if isinstance(metadata_value, dict) else {}
        defaults = suite_request_defaults(suite, seed)
        request, identity = request_from_metadata(
            task_id=task_id,
            seed=seed,
            metadata=metadata,
            defaults=defaults,
        )
        episodes.append(
            ReplayEpisode(
                seed=seed,
                actions=actions,
                request=request,
                expected_identity=identity,
                source={
                    "kind": "evaluation_run",
                    "run": str(source),
                    "run_id": manifest.get("run_id"),
                    "action_path": str(action_path),
                    "source_status": source_result.get("status"),
                },
            )
        )
    if not episodes:
        raise FileNotFoundError(f"no matching actions.npz files under {source}")

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
    contract_value = manifest.get("policy_contract")
    contract = (
        PolicyContract.from_hello(contract_value)
        if isinstance(contract_value, dict)
        else replay_policy_contract(horizon=suite.rollout.replan_steps)
    )
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
                "mode": RUN_REPLAY_MODE,
                "source_run": str(source),
                "source_run_id": manifest.get("run_id"),
                "action_source": "episodes/seed_*/actions.npz[actions]",
                "episode_identity_required": bool(args.require_episode_identity),
            },
            "rollout": {
                "max_steps": max(len(episode.actions) for episode in episodes),
                "replan_steps": min(
                    suite.rollout.replan_steps, contract.action.horizon
                ),
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


def _manifest_task_id(manifest: dict) -> str:
    task = manifest.get("task", {})
    value = task.get("id") if isinstance(task, dict) else None
    if not value:
        value = manifest.get("task_id")
    if not value:
        raise ValueError("source run manifest has no task id")
    return str(value)


def _camera_aliases(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )


if __name__ == "__main__":
    raise SystemExit(main())
