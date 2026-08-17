#!/usr/bin/env python3
"""Replay saved WA-01 policy actions with the current task evaluator."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.evaluation.artifacts import ArtifactWriter
from bench.evaluation.rollout import EpisodeResult
from bench.evaluation.suite import load_eval_suite
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


EVALUATOR_PROFILE = "wa01-ordered-stages-v1"
STAGES = ("sugar", "one_ready", "ready", "selection", "correct", "success")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_run", type=Path)
    parser.add_argument("--seeds", default=None, help="e.g. '1050-1053' or '1050,1060'")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel simulator processes (default: 4; use 1 for serial replay)",
    )
    parser.add_argument(
        "--render-observations",
        action="store_true",
        help="retain camera rendering during replay (not needed for action-only rescoring)",
    )
    return parser.parse_args()


def parse_seeds(value: str | None) -> set[int]:
    if value is None:
        return set()
    result: set[int] = set()
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            step = 1 if end >= start else -1
            result.update(range(start, end + step, step))
        else:
            result.add(int(item))
    return result


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _load_source_result(action_path: Path) -> dict[str, Any]:
    path = action_path.parent / "result.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _physics_only_descriptor(descriptor: Any) -> Any:
    sensors = replace(
        descriptor.sensors,
        camera_depths=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_observations=False,
    )
    return replace(descriptor, sensors=sensors)


def _failed_stage_reason(diagnostics: dict[str, Any]) -> str:
    failure_code = diagnostics.get("failure_code")
    if failure_code:
        return str(failure_code)
    stages = diagnostics.get("stages")
    if isinstance(stages, dict):
        for name in STAGES:
            values = stages.get(name)
            if isinstance(values, dict) and not bool(values.get("qualified")):
                reason = values.get("reason") or "not_qualified"
                return f"stage:{name}:{reason}"
    return "saved_actions_exhausted"


def _write_stage_summary(output: Path) -> dict[str, Any]:
    results = sorted((output / "episodes").glob("seed_*/result.json"))
    counts = {name: 0 for name in STAGES}
    valid = 0
    for path in results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") not in {"success", "failure"}:
            continue
        valid += 1
        stages = payload.get("diagnostics", {}).get("stages", {})
        for name in STAGES:
            values = stages.get(name, {}) if isinstance(stages, dict) else {}
            counts[name] += int(
                isinstance(values, dict) and bool(values.get("qualified"))
            )
    summary = {
        "schema_version": "membench.wa01_stage_summary/v1",
        "qualification_source": "diagnostics.stages.<stage>.qualified",
        "episodes": valid,
        "stages": {
            name: {
                "count": counts[name],
                "rate": counts[name] / valid if valid else 0.0,
            }
            for name in STAGES
        },
    }
    (output / "stage_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _replay_episode(
    *,
    seed: int,
    action_path: str,
    raw_options: dict[str, Any],
    task_options: dict[str, Any],
    env_split: str,
    max_steps: int | None,
    render_observations: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run one simulator replay without writing shared run artifacts."""

    task_id = "WA-01"
    source_action_path = Path(action_path)
    source_result = _load_source_result(source_action_path)
    definition = get_task_definition(task_id)
    runtime_api = resolve_builtin_task_plugin(task_id).load_runtime_api()
    if runtime_api is None:
        raise LookupError(f"{task_id} has no canonical runtime API")

    episode = definition.resolve(
        EpisodeRequest(
            task_id=task_id,
            seed=seed,
            variant=str(raw_options.get("variant", "memory")),
            split="evaluation",
            env_split=env_split,
            options=dict(task_options),
            task_config_file=_resolve_path(raw_options.get("task_config_file")),
            observation_profile_file=_resolve_path(
                raw_options.get("observation_profile_file")
            ),
            allow_partial_calibration=False,
        )
    )
    source_episode = source_result.get("episode_metadata", {}).get(
        "resolved_episode", {}
    )
    source_identity = (
        source_episode.get("identity") if isinstance(source_episode, dict) else None
    )
    identity_matches = source_identity is None or source_identity == episode.identity

    descriptor = runtime_api.build_descriptor(episode, execute=True)
    if not render_observations:
        descriptor = _physics_only_descriptor(descriptor)
    runtime = RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
        descriptor,
        expert=False,
    )
    actions = np.asarray(np.load(source_action_path)["actions"], dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[0] == 0
        or actions.shape[1] != 13
        or not np.all(np.isfinite(actions))
    ):
        runtime.close()
        raise ValueError(
            f"invalid WA-01 action array at {source_action_path}: {actions.shape}"
        )
    replay_actions = actions if max_steps is None else actions[:max_steps]

    started = time.monotonic()
    success_step = None
    diagnostics: dict[str, Any] = {}
    consumed = 0
    try:
        for step, action in enumerate(replay_actions, start=1):
            transition = runtime.step(action)
            consumed = step
            diagnostics = dict(transition.diagnostics.details or {})
            if transition.diagnostics.success and success_step is None:
                success_step = step
    finally:
        runtime.close()

    success = bool(diagnostics.get("success") is True)
    used = np.asarray(replay_actions[:consumed], dtype=np.float32)
    result = EpisodeResult(
        seed=seed,
        episode_id=f"wa-01-seed-{seed:06d}",
        task_id=task_id,
        suite_version=EVALUATOR_PROFILE,
        status="success" if success else "failure",
        success=success,
        success_step=success_step,
        env_steps=consumed,
        policy_calls=0,
        reward_sum=0.0,
        failure_reason=None if success else _failed_stage_reason(diagnostics),
        diagnostics=diagnostics,
        action_mean=used.mean(axis=0).tolist(),
        action_std=used.std(axis=0).tolist(),
        action_min=used.min(axis=0).tolist(),
        action_max=used.max(axis=0).tolist(),
        policy_actions_returned=0,
        policy_actions_consumed=consumed,
        duration_s=time.monotonic() - started,
        episode_metadata={
            **dict(source_result.get("episode_metadata") or {}),
            "source_episode_identity": source_identity,
            "current_episode_identity": episode.identity,
            "episode_identity_matches": identity_matches,
            "replayed_env_steps": consumed,
            "execution_mode": "saved_actions_action_only_physics",
        },
    )
    return asdict(result), used


def _commit_episode(
    writer: ArtifactWriter,
    *,
    result: dict[str, Any],
    actions: np.ndarray,
) -> None:
    session = writer.start_episode(
        seed=int(result["seed"]),
        record_video=False,
        video_cameras=(),
        video_fps=20,
    )
    try:
        for action in actions:
            session.append_action(action)
        session.finish(result)
    except BaseException:
        session.abort()
        raise


def main() -> int:
    args = parse_args()
    source = args.source_run.expanduser().resolve()
    output = args.output_run.expanduser().resolve()
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task", {}).get("id") != "WA-01":
        raise ValueError("replay_wa01_actions.py only accepts WA-01 runs")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    seed_filter = parse_seeds(args.seeds)
    action_files = sorted((source / "episodes").glob("seed_*/actions.npz"))
    if seed_filter:
        action_files = [
            path
            for path in action_files
            if int(path.parent.name.removeprefix("seed_")) in seed_filter
        ]
    seeds = tuple(int(path.parent.name.removeprefix("seed_")) for path in action_files)
    if not seeds:
        raise FileNotFoundError(f"no matching saved actions under {source / 'episodes'}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output exists and is non-empty: {output}")

    suite_path = _resolve_path(manifest.get("suite", {}).get("path"))
    if suite_path is None:
        raise ValueError("source manifest has no suite.path")
    suite = load_eval_suite(suite_path)
    raw_options = dict(suite.task_options)
    task_options = raw_options.get("task_options", {})
    if not isinstance(task_options, dict):
        raise ValueError("task.options.task_options must be a map")

    task_id = "WA-01"
    writer = ArtifactWriter(output, expected_seeds=seeds)

    replay_manifest = dict(manifest)
    replay_manifest.update(
        {
            "run_id": output.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seeds": list(seeds),
            "replay": {
                "source_run": str(source),
                "source_run_id": manifest.get("run_id"),
                "mode": "saved_actions_environment_rescore",
                "action_source": "episodes/seed_*/actions.npz[actions]",
                "evaluator_profile": EVALUATOR_PROFILE,
                "physics_only_observations": not args.render_observations,
                "workers": min(args.workers, len(seeds)),
            },
            "rollout": {
                **dict(manifest.get("rollout", {})),
                "stop_on_success": False,
                "max_steps": args.max_steps,
            },
            "artifacts": {
                "record_video": False,
                "video_cameras": [],
                "video_fps": 20,
            },
        }
    )
    writer.write_run_manifest(replay_manifest)

    jobs = [
        {
            "seed": seed,
            "action_path": str(action_path),
            "raw_options": raw_options,
            "task_options": task_options,
            "env_split": suite.env_split,
            "max_steps": args.max_steps,
            "render_observations": args.render_observations,
        }
        for seed, action_path in zip(seeds, action_files)
    ]
    worker_count = min(args.workers, len(jobs))
    print(
        f"[replay] episodes={len(jobs)} workers={worker_count} output={output}",
        flush=True,
    )

    def commit_and_report(index: int, result: dict[str, Any], actions: np.ndarray) -> None:
        _commit_episode(writer, result=result, actions=actions)
        writer.write_summary()
        _write_stage_summary(output)
        diagnostics = dict(result.get("diagnostics") or {})
        print(
            f"[replay] [{index}/{len(jobs)}] seed={int(result['seed']):06d} "
            f"{result['status'].upper()} steps={result['env_steps']} "
            f"qualified={diagnostics.get('stage_checks', {})} "
            f"identity_match={result['episode_metadata']['episode_identity_matches']}",
            flush=True,
        )

    if worker_count == 1:
        for index, job in enumerate(jobs, start=1):
            result, actions = _replay_episode(**job)
            commit_and_report(index, result, actions)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = [executor.submit(_replay_episode, **job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                result, actions = future.result()
                commit_and_report(index, result, actions)

    summary = writer.write_summary()
    stages = _write_stage_summary(output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(stages, indent=2, sort_keys=True))
    print(f"run_dir={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
