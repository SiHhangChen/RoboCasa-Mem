#!/usr/bin/env python3
"""Replay WR-06 native recordings under the current physical evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.rollout import EpisodeResult
from bench.recording.replay_validation import REPLAY_MODE, load_recorded_actions
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


TASK_ID = "WR-06"
EVALUATOR_PROFILE = "wr06-recorded-action-only-physics-replay-v2"
SOURCE_BRANCHES = {
    "success": Path("train") / TASK_ID,
    "failed": Path("debug_failed") / TASK_ID,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs" / "datasets" / "WR06",
    )
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=None,
        help="subset such as '0-9' or '0,5,10' (default: all discovered seeds)",
    )
    parser.add_argument(
        "--source-status",
        choices=("all", "success", "failed"),
        default="all",
        help="which recording outcomes to audit (default: all)",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-cameras",
        default="agentview_left,agentview_right,eye_in_hand",
        help="comma-separated observation-profile camera aliases",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--task-config-file", type=Path, default=None)
    parser.add_argument("--observation-profile-file", type=Path, default=None)
    identity_policy = parser.add_mutually_exclusive_group()
    identity_policy.add_argument(
        "--require-episode-identity",
        dest="require_episode_identity",
        action="store_true",
        default=True,
        help="reject recording/current episode identity mismatches (default)",
    )
    identity_policy.add_argument(
        "--allow-episode-identity-mismatch",
        dest="require_episode_identity",
        action="store_false",
        help="allow a diagnostic replay against different task semantics",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop on the first episode error instead of recording an error result",
    )
    return parser.parse_args()


def _parse_seeds(value: str | None) -> set[int]:
    if value is None:
        return set()
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
    return set(result)


def _episode_key(path: Path) -> tuple[int, int]:
    parts = path.name.removeprefix("seed_").split("_attempt_", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid recorded episode directory name: {path.name}")
    return int(parts[0]), int(parts[1])


def _load_manifest_items(dataset_root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    path = dataset_root / "manifest.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for item in payload.get("items", ()):
        if item.get("task_id") != TASK_ID:
            continue
        key = (int(item["seed"]), int(item.get("attempt", 0)))
        result[key] = dict(item)
    return result


def _discover_episodes(
    dataset_root: Path,
    seeds: set[int],
    *,
    source_status: str,
) -> list[dict[str, Any]]:
    """Discover both accepted and failed recordings without trusting a partial manifest."""

    manifest_items = _load_manifest_items(dataset_root)
    by_seed: dict[int, list[dict[str, Any]]] = {}
    statuses = ("success", "failed") if source_status == "all" else (source_status,)
    for status in statuses:
        task_dir = dataset_root / SOURCE_BRANCHES[status]
        for episode_dir in sorted(task_dir.glob("seed_*_attempt_*")):
            trajectory = episode_dir / "trajectory.h5"
            if not trajectory.is_file():
                continue
            seed, attempt = _episode_key(episode_dir)
            if seeds and seed not in seeds:
                continue
            manifest_item = manifest_items.get((seed, attempt), {})
            candidate = {
                "seed": seed,
                "attempt": attempt,
                "episode_dir": episode_dir,
                "source_status": status,
                "source_success": status == "success",
                "manifest_selected": bool(manifest_item),
                "episode_identity": manifest_item.get("episode_identity"),
            }
            by_seed.setdefault(seed, []).append(candidate)

    episodes: list[dict[str, Any]] = []
    for seed, candidates in by_seed.items():
        # The manifest describes the selected attempt when it is complete. For
        # incomplete/overwritten manifests, prefer an accepted recording and
        # then the newest attempt found on disk.
        selected = max(
            candidates,
            key=lambda item: (
                bool(item["manifest_selected"]),
                bool(item["source_success"]),
                int(item["attempt"]),
            ),
        )
        episodes.append(selected)
    episodes.sort(key=lambda item: int(item["seed"]))
    return episodes


def _load_recording_metadata(episode_dir: Path) -> dict[str, Any]:
    path = episode_dir / "episode.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    resolved = dict(metadata.get("resolved_episode") or {})
    return {
        "episode_identity": metadata.get("episode_identity"),
        "initialization": metadata.get("initialization"),
        "recorded_step_count": payload.get("step_count"),
        "recorded_termination": payload.get("termination"),
        "variant": resolved.get("variant", metadata.get("variant", "memory")),
        "split": resolved.get("split", metadata.get("split", "train")),
        "env_split": resolved.get("env_split", metadata.get("env_split", "pretrain")),
    }


def _video_aliases(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(item.strip() for item in str(value).split(",") if item.strip())
    )


def _action_stats(actions: np.ndarray) -> dict[str, list[float]]:
    if actions.size == 0:
        return {name: [] for name in ("mean", "std", "min", "max")}
    return {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
    }


def _failure_reason(
    *,
    success: bool,
    success_step: int | None,
    consumed: int,
    available: int,
    diagnostics: dict[str, Any] | None = None,
) -> str | None:
    if success:
        return None
    if success_step is not None:
        return "terminal_state_not_preserved"
    if consumed < available:
        return "replay_stopped_early"
    details = dict(diagnostics or {})
    physical_events = dict(details.get("physical_events") or {})
    violation = physical_events.get("violation_reason") or details.get("failure_code")
    if violation:
        return str(violation)
    phase_reasons = {
        "home": "shelf_pick_incomplete",
        "to_plate": "plate_staging_incomplete",
        "at_plate": "plate_repick_incomplete",
        "to_return": "return_placement_incomplete",
        "verify_return": "terminal_verification_incomplete",
        "invalid": "protocol_violation",
    }
    phase = physical_events.get("phase")
    if phase in phase_reasons:
        return phase_reasons[str(phase)]
    return "recorded_actions_exhausted"


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    seeds = _parse_seeds(args.seeds)
    video_cameras = _video_aliases(args.video_cameras)
    if args.record_video and not video_cameras:
        raise ValueError("--record-video requires at least one --video-cameras alias")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    episodes = _discover_episodes(
        dataset_root,
        seeds,
        source_status=args.source_status,
    )
    if not episodes:
        raise FileNotFoundError(f"no WR-06 trajectories found under {dataset_root}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")

    definition = get_task_definition(TASK_ID)
    api = resolve_builtin_task_plugin(TASK_ID).load_runtime_api()
    if api is None:
        raise LookupError("WR-06 has no canonical runtime API")

    expected_seeds = tuple(int(item["seed"]) for item in episodes)
    writer = ArtifactWriter(output, expected_seeds=expected_seeds)
    source_counts = {
        status: sum(item["source_status"] == status for item in episodes)
        for status in ("success", "failed")
    }
    writer.write_run_manifest(
        {
            "schema_version": "membench.run_manifest/v1",
            "run_id": output.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": {
                "id": TASK_ID,
                "env_name": definition.env_name,
                "task_version": definition.task_version,
            },
            "dataset_root": str(dataset_root),
            "seeds": list(expected_seeds),
            "source": {
                "status_filter": args.source_status,
                "discovered_counts": source_counts,
            },
            "replay": {
                "mode": REPLAY_MODE,
                "evaluator_profile": EVALUATOR_PROFILE,
                "action_source": "trajectory.h5[data/demo_1/actions]",
                "execution_mode": "action_only_physics",
                "episode_identity_policy": (
                    "require_match"
                    if args.require_episode_identity
                    else "allow_mismatch_for_diagnostics"
                ),
            },
            "rollout": {
                "action_repeat": 1,
                "stop_on_success": False,
                "full_horizon_terminal_validation": True,
                "max_steps": args.max_steps,
            },
            "artifacts": {
                "record_video": bool(args.record_video),
                "video_cameras": list(video_cameras) if args.record_video else [],
                "video_fps": args.video_fps,
            },
        }
    )

    for index, item in enumerate(episodes, start=1):
        seed = int(item["seed"])
        directory = Path(item["episode_dir"])
        print(
            f"[replay] [{index}/{len(episodes)}] seed={seed:06d} "
            f"source={item['source_status']} dir={directory.name}",
            flush=True,
        )
        session = writer.start_episode(
            seed=seed,
            record_video=bool(args.record_video),
            video_cameras=video_cameras if args.record_video else (),
            video_fps=args.video_fps,
        )
        runtime = None
        started = time.monotonic()
        consumed = 0
        success_step = None
        diagnostics: dict[str, Any] = {}
        actions = np.empty((0, 13), dtype=np.float32)
        recorded: dict[str, Any] = {}
        current_identity: str | None = None
        try:
            actions = load_recorded_actions(directory / "trajectory.h5")
            recorded = _load_recording_metadata(directory)
            request = EpisodeRequest(
                task_id=TASK_ID,
                seed=seed,
                variant=str(recorded.get("variant", "memory")),
                split=str(recorded.get("split", "train")),
                env_split=str(recorded.get("env_split", "pretrain")),
                task_config_file=args.task_config_file,
                observation_profile_file=args.observation_profile_file,
                allow_partial_calibration=False,
            )
            episode = definition.resolve(request)
            current_identity = episode.identity
            recorded_identity = recorded.get("episode_identity") or item.get("episode_identity")
            identity_matches = bool(
                recorded_identity is not None and recorded_identity == current_identity
            )
            if recorded_identity is not None and not identity_matches:
                message = (
                    f"recorded/current episode identity mismatch for seed={seed}: "
                    f"recorded={recorded_identity} current={current_identity}"
                )
                if args.require_episode_identity:
                    raise ValueError(message)
                print(f"[replay] WARNING {message}", flush=True)

            descriptor = api.build_descriptor(episode, execute=True)
            runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(
                descriptor,
                expert=False,
            )
            replay_actions = (
                actions
                if args.max_steps is None
                else actions[: min(len(actions), int(args.max_steps))]
            )
            for step, action in enumerate(replay_actions, start=1):
                transition = runtime.step(action)
                session.append_action(action)
                consumed = step
                diagnostics = dict(transition.diagnostics.details or {})
                if transition.diagnostics.success and success_step is None:
                    success_step = step
                if args.record_video:
                    frames = {}
                    for alias in video_cameras:
                        source_key = descriptor.sensors.observation_key(alias)
                        if source_key not in transition.observation:
                            raise KeyError(
                                f"missing camera alias {alias!r} (expected {source_key!r})"
                            )
                        frames[alias] = np.ascontiguousarray(
                            transition.observation[source_key]
                        )
                    session.append_frames(frames)

            success = bool(diagnostics.get("success") is True)
            used = np.asarray(actions[:consumed], dtype=np.float32)
            stats = _action_stats(used)
            recorded_identity = recorded.get("episode_identity") or item.get("episode_identity")
            result = EpisodeResult(
                seed=seed,
                episode_id=f"wr-06-seed-{seed:06d}",
                task_id=TASK_ID,
                suite_version=EVALUATOR_PROFILE,
                status="success" if success else "failure",
                success=success,
                success_step=success_step,
                env_steps=consumed,
                policy_calls=0,
                reward_sum=0.0,
                failure_reason=_failure_reason(
                    success=success,
                    success_step=success_step,
                    consumed=consumed,
                    available=len(actions),
                    diagnostics=diagnostics,
                ),
                diagnostics=diagnostics,
                action_mean=stats["mean"],
                action_std=stats["std"],
                action_min=stats["min"],
                action_max=stats["max"],
                policy_actions_consumed=consumed,
                duration_s=time.monotonic() - started,
                episode_metadata={
                    "source_episode_dir": str(directory),
                    "source_status": item["source_status"],
                    "source_success": bool(item["source_success"]),
                    "source_attempt": int(item["attempt"]),
                    "recorded_step_count": recorded.get("recorded_step_count"),
                    "recorded_termination": recorded.get("recorded_termination"),
                    "recorded_episode_identity": recorded_identity,
                    "current_episode_identity": current_identity,
                    "episode_identity_matches": bool(
                        recorded_identity is not None
                        and recorded_identity == current_identity
                    ),
                    "replayed_env_steps": consumed,
                    "execution_mode": "action_only_physics",
                },
            )
            session.finish(asdict(result))
            print(
                f"[replay] seed={seed:06d} "
                f"{'SUCCESS' if success else 'FAILURE'} steps={consumed} "
                f"reason={result.failure_reason}",
                flush=True,
            )
        except BaseException as exc:
            if args.fail_fast or isinstance(exc, (KeyboardInterrupt, SystemExit)):
                session.abort()
                raise
            used = np.asarray(actions[:consumed], dtype=np.float32)
            stats = _action_stats(used)
            result = EpisodeResult(
                seed=seed,
                episode_id=f"wr-06-seed-{seed:06d}",
                task_id=TASK_ID,
                suite_version=EVALUATOR_PROFILE,
                status="error",
                success=False,
                success_step=success_step,
                env_steps=consumed,
                policy_calls=0,
                reward_sum=0.0,
                failure_reason="replay_error",
                error="".join(traceback.format_exception(exc)),
                diagnostics=diagnostics,
                action_mean=stats["mean"],
                action_std=stats["std"],
                action_min=stats["min"],
                action_max=stats["max"],
                policy_actions_consumed=consumed,
                duration_s=time.monotonic() - started,
                episode_metadata={
                    "source_episode_dir": str(directory),
                    "source_status": item["source_status"],
                    "source_success": bool(item["source_success"]),
                    "source_attempt": int(item["attempt"]),
                    "recorded_step_count": recorded.get("recorded_step_count"),
                    "current_episode_identity": current_identity,
                    "replayed_env_steps": consumed,
                    "execution_mode": "action_only_physics",
                },
            )
            session.finish(asdict(result))
            print(
                f"[replay] seed={seed:06d} ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            if runtime is not None:
                runtime.close()

    summary = writer.write_summary()
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"run_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
