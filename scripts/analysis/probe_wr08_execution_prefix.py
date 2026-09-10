#!/usr/bin/env python3
"""Replay a verified WR-08 prefix, then test expert changes without cameras.

Counterfactual diagnostics only. These results do not qualify a dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.execution.contracts import NavigateStage
from bench.recording.replay_validation import load_recorded_actions
from bench.runtime import RoboCasaRuntimeFactory, RobotStateExtractor
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
from scripts.annotate.planner_to_stage_boundaries import count_recorded_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-stage", type=int, required=True)
    parser.add_argument("--end-stage", type=int, required=True)
    parser.add_argument("--shelf-collisions", action="store_true")
    parser.add_argument("--shelf-point-cloud", action="store_true")
    parser.add_argument("--grasp-clearance", type=float)
    parser.add_argument("--pregrasp-height", type=float)
    parser.add_argument("--shelf-inflate", type=float, default=0.05)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    recorded = json.loads((args.episode_dir / "episode.json").read_text())
    metadata = recorded["metadata"]
    resolved = metadata["resolved_episode"]
    episode = TaskCatalog.resolve(EpisodeRequest(
        task_id="WR-08", seed=int(resolved["seed"]),
        variant=resolved["variant"], split=resolved["split"],
        env_split=resolved["env_split"], options=dict(resolved["options"]),
        task_config_file=Path(resolved["task_config_file"]),
        observation_profile_file=Path(resolved["observation_profile_file"]),
        allow_partial_calibration=bool(resolved["allow_partial_calibration"]),
    ))
    if episode.identity != metadata["episode_identity"]:
        raise ValueError("Source recording and current task configuration differ")
    stages = json.loads((args.episode_dir / "planner.json").read_text())["rollout"]["stages"]
    if not 0 <= args.start_stage <= args.end_stage < 17:
        raise ValueError("Expected 0 <= start-stage <= end-stage < 17")
    prefix = stages[:args.start_stage]
    if len(prefix) != args.start_stage or not all(s["success"] for s in prefix):
        raise ValueError("The source prefix must contain only completed successful stages")
    prefix_steps = sum(count_recorded_steps(s, path=f"stage[{i}]")[0] for i, s in enumerate(prefix))
    api = resolve_builtin_task_plugin("WR-08").load_runtime_api()
    descriptor = api.build_descriptor(episode, execute=True)
    descriptor = replace(descriptor, sensors=replace(
        descriptor.sensors, use_camera_observations=False,
        has_renderer=False, has_offscreen_renderer=False,
    ))
    runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(descriptor, expert=True)
    try:
        actions = load_recorded_actions(args.episode_dir / "trajectory.h5")
        if prefix_steps >= len(actions):
            raise ValueError("Prefix leaves no source state for verification")
        runner = runtime.initialized.runner
        plan = TaskCatalog.get("WR-08").build_expert_plan(ExpertPlanInputs.from_descriptor(descriptor))
        # Replay station choices too, so later sampling uses the recorded stream.
        for stage in plan.stages[:args.start_stage]:
            if isinstance(stage, NavigateStage):
                runner.navigation.sample_station_pose(stage.target.station_id, purpose=stage.target.purpose)
        for action in actions[:prefix_steps]:
            runtime.env.step(action)
        obs = runtime.env._get_observations(force_update=True)
        with h5py.File(args.episode_dir / "trajectory.h5", "r") as handle:
            expected = np.asarray(handle["data/demo_1/obs/robot_state"][prefix_steps])
        state_error = float(np.max(np.abs(RobotStateExtractor().extract(runtime.env, obs)[:30] - expected[:30])))
        if state_error > 0.02:
            raise ValueError(f"Replayed prefix differs from recording: {state_error}")
        controller = runner.manipulation
        controller.refresh_obs()
        if args.shelf_collisions or args.shelf_point_cloud:
            options = {
                **controller.runtime_config.static_scene_options,
                "enabled": args.shelf_point_cloud,
                "validate_collisions": args.shelf_collisions,
                "fixture_keywords": "opencabinet",
                "inflate": args.shelf_inflate,
            }
            controller.runtime_config = replace(controller.runtime_config, static_scene_options=options)
            controller.stage_executor.config = controller.runtime_config
        selected = list(plan.stages[args.start_stage:args.end_stage + 1])
        pick_overrides = {}
        if args.grasp_clearance is not None:
            pick_overrides["grasp_clearance"] = args.grasp_clearance
        if args.pregrasp_height is not None:
            pick_overrides["pregrasp_height"] = args.pregrasp_height
        if pick_overrides:
            selected = [
                replace(stage, metadata={**stage.metadata, **pick_overrides})
                if getattr(stage, "operation", None) == "pick" else stage
                for stage in selected
            ]

        def report(index, stage, result):
            print(json.dumps({"stage": args.start_stage + index, "name": stage.name,
                              "success": result.success, "failure": result.failure_code}), flush=True)

        result = None if args.inspect_only else runner.run(replace(plan, stages=tuple(selected)), on_stage_end=report)
        from bench.execution.planning.collision_monitor import make_collision_monitor
        monitor = make_collision_monitor(runtime.env, enabled=True, fixture_keywords="opencabinet")
        fixture_geoms = {}
        model = runtime.env.sim.model
        for name, fixture in runtime.env.fixtures.items():
            if "opencabinet" in str(name).lower():
                fixture_geoms[name] = {
                    "root_body": fixture.root_body,
                    "declared": list(fixture.contact_geoms),
                    "compiled": [
                        {"name": model.geom_id2name(i), "group": int(model.geom_group[i]),
                         "contype": int(model.geom_contype[i]), "conaffinity": int(model.geom_conaffinity[i]),
                         "position": runtime.env.sim.data.geom_xpos[i].tolist(),
                         "size": model.geom_size[i].tolist()}
                        for i in range(model.ngeom) if str(model.geom_id2name(i)).startswith(str(name))
                    ],
                }
        payload = {
            "diagnostic_only": True, "seed": episode.seed,
            "episode_dir": str(args.episode_dir), "episode_identity": episode.identity,
            "prefix_steps": prefix_steps, "prefix_state30_max_abs_error": state_error,
            "options": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "result": None if result is None else asdict(result),
            "shelf_monitor": monitor, "fixture_geoms": fixture_geoms,
            "diagnostics": runtime.diagnostics().details,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "success": None if result is None else result.success}), flush=True)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
