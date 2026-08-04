#!/usr/bin/env python3
"""Run a station, expert stage, operation, or EEF probe in RoboCasa.

This developer tool is intentionally built on the production canonical path:
``EpisodeRequest -> ResolvedEpisode -> RuntimeDescriptor -> EpisodeRuntime``.
It builds the canonical runtime and ExpertPlan directly.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.execution.contracts import NavigateStage, OperateStage
from bench.execution.navigation.contracts import StationTarget
from bench.execution.probe import (
    build_probe_task,
    inject_station_pose,
    load_eef_pose_file,
    load_stage_file,
    make_move_eef_stage,
    reset_base_to_station,
    resolve_registered_task_id,
    resolve_station_pose,
    select_workflow_stages,
    stage_catalog,
    station_catalog,
)
from bench.recording.runner import RecordingRunner, build_recording_config
from bench.recording.writers import jsonable
from bench.runtime import RoboCasaRuntimeFactory
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a canonical RoboCasa task runtime")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="debug")
    parser.add_argument("--env-split", default="pretrain")
    parser.add_argument("--variant", default="memory")
    parser.add_argument("--task-option", action="append", default=[])
    parser.add_argument("--camera", action="append", default=None)
    parser.add_argument("--station", default=None)
    parser.add_argument("--station-file", type=Path, default=None)
    parser.add_argument("--station-pose", type=float, nargs=3, default=None)
    parser.add_argument("--station-sample-index", type=int, default=None)
    parser.add_argument("--station-purpose", default=None)
    parser.add_argument("--station-mode", choices=("reset", "navigate"), default="reset")
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--operation", action="append", nargs=2, default=[])
    parser.add_argument("--stages-file", type=Path, default=None)
    parser.add_argument("--eef-pose-file", type=Path, default=None)
    parser.add_argument("--eef-pose", type=float, nargs=7, default=None)
    parser.add_argument("--eef-frame", choices=("auto", "world", "base"), default="auto")
    parser.add_argument("--eef-stage-name", default="move_eef_probe")
    parser.add_argument("--gripper-action", type=float, default=-1.0)
    parser.add_argument("--holding-object", default=None)
    parser.add_argument("--settle-steps", type=int, default=None)
    parser.add_argument("--record-trajectory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-partial-calibration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task_probe"))
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--list-stations", action="store_true")
    return parser.parse_args(argv)


def _options(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--task-option must be KEY=VALUE, got {item!r}")
        result[key.strip()] = value
    return result


def _build_runtime(args: argparse.Namespace, task_id: str):
    episode = TaskCatalog.resolve(
        EpisodeRequest(
            task_id=task_id,
            seed=int(args.seed),
            split=str(args.split),
            env_split=str(args.env_split),
            variant=str(args.variant),
            options=_options(args.task_option),
            task_config_file=args.task_config,
            allow_partial_calibration=bool(args.allow_partial_calibration),
        )
    )
    api = resolve_builtin_task_plugin(task_id).load_runtime_api()
    descriptor = api.build_descriptor(episode, bool(args.execute))
    plan = TaskCatalog.get(task_id).build_expert_plan(
        ExpertPlanInputs.from_descriptor(descriptor)
    )
    runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(descriptor, expert=True)
    return runtime, plan


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "stage_count": len(result.stages),
        "stages": [
            {
                "stage_index": int(stage.index),
                "stage_kind": stage.kind.value,
                "success": bool(stage.success),
                "failure_code": stage.failure_code,
                "details": dict(stage.details),
            }
            for stage in result.stages
        ],
    }


def _validate_sources(args: argparse.Namespace) -> None:
    sources = [args.stage, args.operation, args.stages_file, args.eef_pose_file, args.eef_pose]
    if sum(bool(value) for value in sources) > 1:
        raise ValueError("choose exactly one action source")
    if args.station_file is not None and args.station_pose is not None:
        raise ValueError("--station-file and --station-pose are mutually exclusive")
    if (args.station_file is not None or args.station_pose is not None) and not args.station:
        raise ValueError("--station is required with --station-file or --station-pose")
    if not -1.0 <= float(args.gripper_action) <= 1.0:
        raise ValueError("--gripper-action must be within [-1, 1]")


def _inline_pose(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Any]]:
    values = list(args.eef_pose)
    frame = "world" if args.eef_frame == "auto" else args.eef_frame
    return {"p": values[:3], "q_xyzw": values[3:]}, frame, {"source": "inline_cli", "frame": frame}


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_sources(args)
    task_id = resolve_registered_task_id(args.task)
    if args.record and args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"recording output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    runtime, production_plan = _build_runtime(args, task_id)
    runner = runtime.initialized.runner
    if runner is None:
        runtime.close()
        raise RuntimeError(f"{task_id} did not expose a canonical expert runner")
    try:
        descriptor = runtime.initialized.descriptor
        report: dict[str, Any] = {
            "task_id": task_id,
            "episode_identity": runtime.episode.identity,
            "scene": {
                "env_name": descriptor.environment.env_name,
                "layout_id": descriptor.environment.layout_id,
                "style_id": descriptor.environment.style_id,
                "scene_seed": descriptor.environment.scene_seed,
            },
            "execute": bool(args.execute),
        }
        if args.list_stages:
            report["available_stages"] = stage_catalog(production_plan)
        if args.list_stations:
            report["available_stations"] = station_catalog(runtime)

        station = None
        if args.station:
            station = resolve_station_pose(
                runtime,
                station_id=args.station,
                station_file=args.station_file,
                station_xy_yaw=args.station_pose,
                sample_index=args.station_sample_index,
                purpose=args.station_purpose,
            )
            report["station"] = {
                "station_id": station.station_id,
                "xy": list(station.xy),
                "yaw": station.yaw,
                "source": station.source,
                "mode": args.station_mode,
            }
            if args.station_mode == "reset":
                report["station_reset"] = reset_base_to_station(runtime, station)
            elif station.source != "task_station_sampler":
                inject_station_pose(runtime, station)

        stages: list[Any] = []
        if args.stage:
            stages = select_workflow_stages(production_plan, args.stage)
        elif args.operation:
            stages = [OperateStage(operation=str(op), target=str(target), name=f"{op}_{target}_probe_{i}") for i, (op, target) in enumerate(args.operation)]
        elif args.stages_file is not None:
            stages = load_stage_file(args.stages_file)
        elif args.eef_pose_file is not None:
            pose, frame, info = load_eef_pose_file(args.eef_pose_file, env=runtime.env, frame_override=args.eef_frame)
            report["eef_pose"] = info
            stages = [make_move_eef_stage(target_pose=pose, frame=frame, name=args.eef_stage_name, gripper_action=args.gripper_action, holding_object=args.holding_object, record_trajectory=args.record_trajectory, settle_steps=args.settle_steps)]
        elif args.eef_pose is not None:
            pose, frame, info = _inline_pose(args)
            report["eef_pose"] = info
            stages = [make_move_eef_stage(target_pose=pose, frame=frame, name=args.eef_stage_name, gripper_action=args.gripper_action, holding_object=args.holding_object, record_trajectory=args.record_trajectory, settle_steps=args.settle_steps)]

        if station is not None and args.station_mode == "navigate":
            stages.insert(0, NavigateStage(target=StationTarget(station_id=station.station_id, purpose=args.station_purpose or "probe", metadata={"use_manual_path": station.source != "task_station_sampler"}), name=f"navigate_to_{station.station_id}_for_probe"))

        if not stages:
            if args.list_stages or args.list_stations or station is not None:
                report["result"] = {"success": True, "stage_count": 0, "stages": []}
                return report
            raise ValueError("nothing to do: specify a station, action source, or list option")

        probe_plan = build_probe_task(production_plan, stages, metadata={"probe_station": report.get("station"), "probe_pose": report.get("eef_pose")})
        if args.record:
            config = build_recording_config(runtime, output_dir=args.output_dir, cameras=args.camera)
            result, output_path = RecordingRunner(runtime=runtime, runner=runner, config=config).run(probe_plan)
            report["output_dir"] = str(output_path)
        else:
            result = runner.run(probe_plan)
        report["result"] = _result_payload(result)
        return report
    finally:
        runtime.close()


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    return 0 if bool(report.get("result", {}).get("success", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
