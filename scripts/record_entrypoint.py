#!/usr/bin/env python
"""Unified canonical expert recording CLI."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.recording.plan import DatasetPlan
from bench.recording.runner import RecordingRunner, build_recording_config
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import TaskCatalog, get_task_definition, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs


def _seeds(value: str) -> list[int]:
    values: list[int] = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(item))
    return list(dict.fromkeys(values))


def _options(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        key, separator, value = str(item).partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--task-option must be KEY=VALUE, got {item!r}")
        result[key.strip()] = value
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record MobileMemBench expert episodes")
    parser.add_argument("--dataset-id", default="membench_v2")
    parser.add_argument("--task-id", action="append", default=None)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--variant", default="memory")
    parser.add_argument("--split", default="train")
    parser.add_argument("--env-split", default="pretrain")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/datasets"))
    parser.add_argument("--camera", action="append", default=None)
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--save-navigation-snapshots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--accept-failed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-partial-calibration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--task-config-file", type=Path, default=None)
    parser.add_argument("--observation-profile-file", type=Path, default=None)
    parser.add_argument("--task-option", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested = args.task_id or list(TaskCatalog.ids())
    task_ids = [TaskCatalog.canonical_id(value) for value in requested]
    plan = DatasetPlan.expand(
        task_ids,
        _seeds(args.seeds),
        variants=(args.variant,),
        attempts=args.attempts,
    )
    if args.dry_run:
        for item in plan.items:
            print(item.work_id, item.task_id, item.seed, item.variant, item.attempt)
        return 0

    dataset_root = args.output_root / args.dataset_id
    summaries: list[dict[str, Any]] = []
    for item in plan.items:
        summary = _record_item(item, args=args, dataset_root=dataset_root)
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "membench.dataset/v2",
                "dataset_id": args.dataset_id,
                "items": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _record_item(item: Any, *, args: argparse.Namespace, dataset_root: Path) -> dict[str, Any]:
    definition = get_task_definition(item.task_id)
    plugin = resolve_builtin_task_plugin(item.task_id)
    request = EpisodeRequest(
        task_id=item.task_id,
        seed=item.seed,
        variant=item.variant,
        split=args.split,
        env_split=args.env_split,
        options=_options(args.task_option),
        task_config_file=args.task_config_file,
        observation_profile_file=args.observation_profile_file,
        allow_partial_calibration=bool(args.allow_partial_calibration),
    )
    episode = definition.resolve(request)
    episode_name = f"seed_{item.seed:06d}_attempt_{item.attempt:02d}"
    accepted_dir = dataset_root / args.split / item.task_id / episode_name
    failed_dir = dataset_root / "debug_failed" / item.task_id / episode_name
    staging_dir = dataset_root / "_staging" / item.task_id / episode_name
    for path in (accepted_dir, failed_dir, staging_dir):
        if path.exists():
            if not args.overwrite:
                raise FileExistsError(f"output directory already exists: {path}")
            shutil.rmtree(path)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    runtime_api = plugin.load_runtime_api()
    descriptor = runtime_api.build_descriptor(episode, bool(args.execute))
    plan = definition.build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
    runtime = RoboCasaRuntimeFactory(runtime_api).create_from_descriptor(
        descriptor,
        expert=True,
    )
    try:
        runner = runtime.initialized.runner
        if runner is None:
            raise RuntimeError(f"{item.task_id} did not build a canonical expert runner")
        config = build_recording_config(
            runtime,
            output_dir=staging_dir,
            cameras=args.camera,
            fps=args.fps,
            save_navigation_snapshots=args.save_navigation_snapshots,
        )
        result, _ = RecordingRunner(runtime=runtime, runner=runner, config=config).run(plan)
        destination = accepted_dir if result.success or args.accept_failed else failed_dir
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_dir), str(destination))
        return {
            "work_id": item.work_id,
            "task_id": item.task_id,
            "seed": item.seed,
            "variant": item.variant,
            "attempt": item.attempt,
            "episode_identity": episode.identity,
            "success": result.success,
            "stage_count": len(result.stages),
            "episode_dir": str(destination),
        }
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
