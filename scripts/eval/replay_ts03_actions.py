#!/usr/bin/env python3
"""Replay saved TS03 policy actions through the ordered task evaluator.

The source policy is not invoked.  Each saved 13-D action is applied to a
freshly resolved TS03 environment, then saucepan containment, release state,
and completion-button joint travel are rescored by
``bench.tasks.ts03.evaluation.TS03Evaluator``.
"""

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
from bench.tasks.ts03.domain.contracts import profile_for_counts
from bench.tasks.ts03.evaluation import (
    STAGE_ORDER,
    TS03EvaluationSnapshot,
    TS03Evaluator,
)


EVALUATOR_PROFILE = "ts03-ordered-stages-action-replay-v1"


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
        help="parallel simulator processes (default: 4)",
    )
    parser.add_argument(
        "--render-observations",
        action="store_true",
        help="retain camera rendering (unnecessary for action-only replay)",
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


def _snapshot(
    env: Any,
    *,
    profile_id: str,
    target_sugar_count: int,
    total_sugar_count: int,
    button_pressed_threshold_m: float,
) -> TS03EvaluationSnapshot:
    physical = env._physical_state()
    confirmed_count = sum(bool(value) for value in physical.sugars_in_saucepan.values())
    button_displacement = float(env.completion_button_displacement())
    return TS03EvaluationSnapshot(
        step_id=int(getattr(env, "timestep", 0)),
        profile_id=profile_id,
        target_sugar_count=target_sugar_count,
        total_sugar_count=total_sugar_count,
        confirmed_sugar_count=confirmed_count,
        objects_released=physical.released,
        completion_button_pressed=(
            button_displacement >= float(button_pressed_threshold_m)
        ),
        details={
            "button_detection": {
                "current_displacement_m": button_displacement,
                "pressed_threshold_m": float(button_pressed_threshold_m),
            },
            "sugars_in_saucepan": {
                str(name): bool(value)
                for name, value in physical.sugars_in_saucepan.items()
            },
        },
    )


def _failed_stage_reason(diagnostics: dict[str, Any]) -> str:
    if diagnostics.get("failure_code"):
        return str(diagnostics["failure_code"])
    evaluation = diagnostics.get("evaluation", {})
    stage_order = evaluation.get("stage_order", STAGE_ORDER)
    stages = diagnostics.get("stages", {})
    for name in stage_order:
        values = stages.get(name, {}) if isinstance(stages, dict) else {}
        if isinstance(values, dict) and not bool(values.get("qualified")):
            return f"stage:{name}:{values.get('reason') or 'not_qualified'}"
    return "saved_actions_exhausted"


def _write_stage_summary(output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "episodes").glob("seed_*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") in {"success", "failure"}:
            rows.append(payload)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        raw = {name: 0 for name in STAGE_ORDER}
        effective = {name: 0 for name in STAGE_ORDER}
        successes = 0
        for payload in items:
            successes += int(bool(payload.get("success")))
            stages = payload.get("diagnostics", {}).get("stages", {})
            for name in STAGE_ORDER:
                values = stages.get(name, {}) if isinstance(stages, dict) else {}
                if not isinstance(values, dict):
                    continue
                raw[name] += int(bool(values.get("qualified")))
                effective[name] += int(bool(values.get("effective_success")))
        count = len(items)
        return {
            "episodes": count,
            "success_count": successes,
            "success_rate": successes / count if count else 0.0,
            "stages": {
                name: {
                    "qualified_count": raw[name],
                    "qualified_rate": raw[name] / count if count else 0.0,
                    "effective_count": effective[name],
                    "effective_rate": effective[name] / count if count else 0.0,
                }
                for name in STAGE_ORDER
            },
        }

    profile_ids = sorted(
        {
            str(payload.get("diagnostics", {}).get("profile_id"))
            for payload in rows
            if payload.get("diagnostics", {}).get("profile_id")
        }
    )
    summary = {
        "schema_version": "membench.ts03_stage_summary/v1",
        "evaluator_profile": EVALUATOR_PROFILE,
        "qualification_source": "diagnostics.stages.<stage>.qualified",
        "effective_source": "diagnostics.stages.<stage>.effective_success",
        "overall": aggregate(rows),
        "profiles": {
            profile_id: aggregate(
                [
                    payload
                    for payload in rows
                    if payload.get("diagnostics", {}).get("profile_id") == profile_id
                ]
            )
            for profile_id in profile_ids
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
    task_id = "TS03"
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
    with np.load(source_action_path) as archive:
        actions = np.asarray(archive["actions"], dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[0] == 0
        or actions.shape[1] != 13
        or not np.all(np.isfinite(actions))
    ):
        runtime.close()
        raise ValueError(f"invalid TS03 action array at {source_action_path}: {actions.shape}")
    replay_actions = actions if max_steps is None else actions[:max_steps]

    plan = descriptor.episode_plan
    profile = profile_for_counts(plan.target_sugar_count, plan.total_sugar_count)
    evaluator = TS03Evaluator(
        profile_id=profile.profile_id,
        target_sugar_count=profile.target_sugar_count,
        total_sugar_count=profile.total_sugar_count,
    )
    completion = dict(descriptor.actions.completion_button_metadata)
    button_threshold = float(completion.get("minimum_press_displacement_m", 0.004))
    evaluator.reset(
        _snapshot(
            runtime.env,
            profile_id=profile.profile_id,
            target_sugar_count=profile.target_sugar_count,
            total_sugar_count=profile.total_sugar_count,
            button_pressed_threshold_m=button_threshold,
        )
    )

    started = time.monotonic()
    success_step = None
    consumed = 0
    eval_result = evaluator.result()
    physical_diagnostics: dict[str, Any] = {}
    try:
        for step, action in enumerate(replay_actions, start=1):
            runtime.step(action)
            consumed = step
            eval_result = evaluator.observe(
                _snapshot(
                    runtime.env,
                    profile_id=profile.profile_id,
                    target_sugar_count=profile.target_sugar_count,
                    total_sugar_count=profile.total_sugar_count,
                    button_pressed_threshold_m=button_threshold,
                )
            )
            if eval_result.success and success_step is None:
                success_step = step
            if eval_result.terminal:
                break
        physical_diagnostics = dict(runtime.env.sugar_diagnostics())
    finally:
        runtime.close()

    diagnostics = eval_result.to_diagnostics()
    diagnostics["physical"] = physical_diagnostics
    diagnostics["source_action_steps"] = int(actions.shape[0])
    diagnostics["replayed_action_steps"] = int(consumed)
    success = bool(eval_result.success)
    used = np.asarray(replay_actions[:consumed], dtype=np.float32)
    result = EpisodeResult(
        seed=seed,
        episode_id=f"ts03-seed-{seed:06d}",
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
            "profile_id": profile.profile_id,
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
    if manifest.get("task", {}).get("id") != "TS03":
        raise ValueError("replay_ts03_actions.py only accepts TS03 runs")
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
                "button_signal": "joint_displacement_threshold",
                "sugar_signal": "current_saucepan_containment_count",
                "physics_only_observations": not args.render_observations,
                "workers": min(args.workers, len(seeds)),
            },
            "rollout": {
                **dict(manifest.get("rollout", {})),
                "stop_on_success": True,
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

    def commit_and_report(
        index: int,
        result: dict[str, Any],
        actions: np.ndarray,
    ) -> None:
        _commit_episode(writer, result=result, actions=actions)
        writer.write_summary()
        _write_stage_summary(output)
        diagnostics = dict(result.get("diagnostics") or {})
        print(
            f"[replay] [{index}/{len(jobs)}] seed={int(result['seed']):06d} "
            f"profile={diagnostics.get('profile_id')} {result['status'].upper()} "
            f"steps={result['env_steps']} "
            f"effective={diagnostics.get('effective_stage_checks', {})} "
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
