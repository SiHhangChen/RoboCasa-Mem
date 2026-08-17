#!/usr/bin/env python3
"""Replay recorded WR-03 expert trajectories under the current environment evaluator.

Reads the recorded dataset produced by ``scripts/record_entrypoint.py`` (native
bench format: per-episode ``trajectory.h5`` with ``demo_1/actions``), rebuilds
each episode with the exact same ``EpisodeRequest`` used at recording time, and
feeds the recorded actions back into the current RoboCasa runtime one by one.
Success is judged by the *current* task-local evaluator (``task_diagnostics`` /
``evaluate_success``), i.e. the same full-horizon semantics used by the official
WR-03 evaluation.

This is the recorded-data analogue of ``replay_wr04_actions.py``:
``replay_wr04_actions.py`` replays actions saved by a policy evaluation run,
while this script replays the original expert demonstration recordings, so it
answers "can the WR-03 task still be executed successfully with the recorded
expert actions under the current environment/task semantics?".

Usage (run inside the ``robocasa`` conda env, like the eval entrypoint):

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=. \
        conda run -n robocasa python scripts/eval/replay_wr03_recorded_actions.py \
            --dataset-root outputs/datasets/WR03 \
            --output-run outputs/eval/wr03-gr00t-recorded-replay-v1

Options:
    --dataset-root  recorded dataset root (default outputs/datasets/WR03)
    --output-run    output run directory (default outputs/eval/wr03-gr00t-recorded-replay-v1)
    --seeds         optional subset, e.g. "0-9" or "0,5,10" (default: all recorded seeds)
    --record-video  save replay RGB videos under each episode's videos/ directory
    --video-cameras comma-separated camera aliases (default: all three WR-03 cameras)
    --task-config-file / --observation-profile-file  optional overrides
"""

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

RECORDED_EVALUATOR_PROFILE = "wr03-recorded-action-only-physics-replay-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "outputs" / "datasets" / "WR03",
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=ROOT / "outputs" / "eval" / "wr03-gr00t-recorded-replay-v1",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="subset of recorded seeds, e.g. '0-9' or '0,5,10' (default: all)",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-cameras",
        default="agentview_left,agentview_right,eye_in_hand",
        help="comma-separated camera aliases used when --record-video is set",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--task-config-file", type=Path, default=None)
    parser.add_argument("--observation-profile-file", type=Path, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional per-episode replay prefix for diagnostics (default: all actions)",
    )
    parser.add_argument(
        "--require-episode-identity",
        action="store_true",
        help="fail if the current resolved episode differs from the recording",
    )
    parser.add_argument(
        "--trace-physics",
        action="store_true",
        help="include sampled can/eef/contact state in result.json",
    )
    return parser.parse_args()


def _parse_seeds(value: str | None) -> list[int]:
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


def _discover_episodes(dataset_root: Path, seed_filter: list[int]) -> list[dict]:
    """Return recorded episode descriptors, preferring the dataset manifest."""
    manifest_path = dataset_root / "manifest.json"
    episodes: list[dict] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("items", []):
            if item.get("task_id") != "WR-03":
                continue
            seed = int(item["seed"])
            if seed_filter and seed not in seed_filter:
                continue
            episode_dir = Path(item["episode_dir"])
            # manifest stores the staging-era path; fall back to the train tree.
            if not (episode_dir / "trajectory.h5").is_file():
                episode_dir = (
                    dataset_root
                    / "train"
                    / "WR-03"
                    / f"seed_{seed:06d}_attempt_{int(item.get('attempt', 0)):02d}"
                )
            episodes.append(
                {"seed": seed, "episode_dir": episode_dir, "attempt": int(item.get("attempt", 0))}
            )
    if not episodes:
        task_dir = dataset_root / "train" / "WR-03"
        for d in sorted(task_dir.glob("seed_*_attempt_*")):
            if not (d / "trajectory.h5").is_file():
                continue
            name = d.name
            seed = int(name.removeprefix("seed_").split("_")[0])
            if seed_filter and seed not in seed_filter:
                continue
            episodes.append({"seed": seed, "episode_dir": d, "attempt": 0})
    episodes.sort(key=lambda item: item["seed"])
    return episodes


def _load_episode_identity(episode_dir: Path) -> dict:
    """Load recording-time episode facts we want to preserve in the replay run."""
    path = episode_dir / "episode.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    metadata = payload.get("metadata", {}) or {}
    return {
        "episode_identity": metadata.get("episode_identity"),
        "initial_prompt": metadata.get("initial_prompt"),
        "initialization": metadata.get("initialization"),
        "recorded_step_count": payload.get("step_count"),
        "recorded_termination": payload.get("termination"),
    }


def _load_recorded_actions(episode_dir: Path) -> np.ndarray:
    trajectory = episode_dir / "trajectory.h5"
    if not trajectory.is_file():
        raise FileNotFoundError(f"missing trajectory.h5 under {episode_dir}")
    with h5py.File(trajectory, "r") as handle:
        # Native recordings nest the demo under h5 root group "data".
        actions = np.asarray(handle["data/demo_1/actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 13:
        raise ValueError(
            f"unexpected recorded action shape {actions.shape} (expected (N, 13))"
        )
    return np.ascontiguousarray(actions)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    seed_filter = _parse_seeds(args.seeds)
    video_cameras = tuple(
        dict.fromkeys(
            item.strip() for item in str(args.video_cameras).split(",") if item.strip()
        )
    )
    if args.record_video and not video_cameras:
        raise ValueError("--record-video requires at least one --video-cameras alias")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    episodes = _discover_episodes(dataset_root, seed_filter)
    if not episodes:
        raise FileNotFoundError(
            f"no recorded WR-03 episodes found under {dataset_root}"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")

    definition = get_task_definition("WR-03")
    plugin = resolve_builtin_task_plugin("WR-03")
    runtime_api = plugin.load_runtime_api()
    if runtime_api is None:
        raise LookupError("WR-03 has no canonical runtime API")

    seeds = tuple(item["seed"] for item in episodes)
    writer = ArtifactWriter(output, expected_seeds=seeds)

    replay_manifest = {
        "schema_version": "membench.run_manifest/v1",
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": {
            "id": "WR-03",
            "env_name": definition.env_name,
            "env_split": "pretrain",
            "task_version": definition.task_version,
        },
        "seeds": list(seeds),
        "dataset_root": str(dataset_root),
        "replay": {
            "source_run": str(dataset_root),
            "source_run_id": dataset_root.name,
            "mode": "recorded_actions_action_only_physics",
            "evaluator_profile": RECORDED_EVALUATOR_PROFILE,
            "action_source": "trajectory.h5[demo_1/actions]",
            "execution_mode": "action_only_physics",
        },
        "rollout": {
            "action_repeat": 1,
            "stop_on_success": False,
            "full_horizon_terminal_validation": True,
            "max_steps": args.max_steps or 15000,
        },
        "artifacts": {
            "record_video": bool(args.record_video),
            "video_cameras": list(video_cameras) if args.record_video else [],
            "video_fps": args.video_fps,
        },
    }
    writer.write_run_manifest(replay_manifest)

    summary_rows: list[dict] = []
    for index, item in enumerate(episodes, start=1):
        seed = item["seed"]
        print(
            f"[replay] [{index}/{len(episodes)}] seed={seed:06d} "
            f"dir={item['episode_dir'].name}",
            flush=True,
        )
        actions = _load_recorded_actions(item["episode_dir"])
        recorded = _load_episode_identity(item["episode_dir"])
        request = EpisodeRequest(
            task_id="WR-03",
            seed=seed,
            variant="memory",
            split="train",
            env_split="pretrain",
            task_config_file=args.task_config_file,
            observation_profile_file=args.observation_profile_file,
            allow_partial_calibration=False,
        )
        episode = definition.resolve(request)
        recorded_identity = recorded.get("episode_identity")
        identity_matches = bool(
            recorded_identity is not None and recorded_identity == episode.identity
        )
        if recorded_identity is not None and not identity_matches:
            message = (
                f"recorded/current episode identity mismatch for seed={seed}: "
                f"recorded={recorded_identity} current={episode.identity}"
            )
            if args.require_episode_identity:
                raise ValueError(message)
            print(f"[replay] WARNING {message}", flush=True)
        descriptor = runtime_api.build_descriptor(episode, execute=True)
        runtime = RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
            descriptor, expert=False
        )
        session = writer.start_episode(
            seed=seed,
            record_video=args.record_video,
            video_cameras=video_cameras if args.record_video else (),
            video_fps=args.video_fps,
        )
        started = time.monotonic()
        success_step = None
        violation_reason = None
        diagnostics: dict = {}
        physics_trace: list[dict] = []
        previous_can_contact = False
        consumed = 0
        try:
            replay_actions = (
                actions
                if args.max_steps is None
                else actions[: min(len(actions), args.max_steps)]
            )
            for step, action in enumerate(replay_actions, start=1):
                transition = runtime.step(action)
                session.append_action(action)
                if args.trace_physics:
                    sample = _physics_trace_sample(runtime.env, step)
                    can_contact = bool(sample["can_contacts"])
                    if step % 25 == 0 or can_contact != previous_can_contact:
                        physics_trace.append(sample)
                    previous_can_contact = can_contact
                if args.record_video:
                    frames = {}
                    for alias in video_cameras:
                        source_key = descriptor.sensors.observation_key(alias)
                        if source_key not in transition.observation:
                            raise KeyError(
                                f"replay observation is missing camera alias {alias!r} "
                                f"(expected key {source_key!r})"
                            )
                        frames[alias] = np.ascontiguousarray(
                            transition.observation[source_key]
                        )
                    session.append_frames(frames)
                consumed = step
                diagnostics = dict(transition.diagnostics.details or {})
                if transition.diagnostics.success and success_step is None:
                    success_step = step
                retrieval = diagnostics.get("retrieval", {})
                if isinstance(retrieval, dict) and retrieval.get("protocol_violation"):
                    violation_reason = "protocol_violation"
                    break
            # Full-horizon validation: success is a final-state property for the
            # store -> close doors -> remember -> retrieve chain, so only the
            # final diagnostics decide.  A protocol violation always fails.
            success = bool(
                diagnostics.get("success") is True and violation_reason is None
            )
            used = np.asarray(actions[:consumed], dtype=np.float32)
            checks = diagnostics.get("success_checks", {})
            if isinstance(checks, dict):
                failed_checks = [
                    name for name, ok in checks.items() if not bool(ok)
                ]
            else:
                failed_checks = []
            result = EpisodeResult(
                seed=seed,
                episode_id=f"wr-03-seed-{seed:06d}",
                task_id="WR-03",
                suite_version=RECORDED_EVALUATOR_PROFILE,
                status="success" if success else "failure",
                success=success,
                success_step=success_step,
                env_steps=consumed,
                policy_calls=0,
                reward_sum=0.0,
                failure_reason=(
                    None
                    if success
                    else str(
                        violation_reason
                        or (
                            "terminal_state_not_preserved"
                            if success_step is not None
                            else (
                                "failed_checks:" + ",".join(failed_checks[:5])
                                if failed_checks
                                else "recorded_actions_exhausted"
                            )
                        )
                    )
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
                    "current_episode_identity": episode.identity,
                    "episode_identity_matches": identity_matches,
                    "replayed_env_steps": consumed,
                    "execution_mode": "action_only_physics",
                    **({"physics_trace": physics_trace} if args.trace_physics else {}),
                },
            )
            session.finish(asdict(result))
            print(
                f"[replay] seed={seed:06d} "
                f"{'SUCCESS' if success else 'FAILURE'} steps={consumed} "
                f"reason={result.failure_reason}",
                flush=True,
            )
            summary_rows.append(
                {
                    "seed": seed,
                    "status": result.status,
                    "success": success,
                    "failure_reason": result.failure_reason,
                    "env_steps": consumed,
                    "success_step": success_step,
                }
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


def _physics_trace_sample(env, step: int) -> dict:
    """Read-only contact trace; never changes simulator state."""
    can_body_id = int(env.obj_body_id["red_can"])
    can_geom_ids = {
        int(env.sim.model.geom_name2id(name))
        for name in (env.objects["red_can"].contact_geoms or [])
    }
    contacts = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 not in can_geom_ids and geom2 not in can_geom_ids:
            continue
        other = geom2 if geom1 in can_geom_ids else geom1
        contacts.append(
            {
                "other_geom": str(env.sim.model.geom_id2name(other)),
                "distance": float(contact.dist),
                "friction": np.asarray(contact.friction, dtype=float)[:3].tolist(),
            }
        )
    eef_id = int(env.robots[0].eef_site_id["right"])
    return {
        "step": int(step),
        "can_position": np.asarray(
            env.sim.data.body_xpos[can_body_id], dtype=float
        ).tolist(),
        "eef_position": np.asarray(
            env.sim.data.site_xpos[eef_id], dtype=float
        ).tolist(),
        "can_contacts": contacts,
    }


if __name__ == "__main__":
    raise SystemExit(main())
