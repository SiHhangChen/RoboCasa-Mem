#!/usr/bin/env python3
"""Inspect WR-08 recorded actions with camera rendering disabled."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.recording.replay_validation import load_recorded_actions
from bench.runtime import RoboCasaRuntimeFactory, RobotStateExtractor
from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = json.loads((args.episode_dir / "episode.json").read_text())["metadata"]
    resolved = metadata["resolved_episode"]
    episode = TaskCatalog.resolve(EpisodeRequest(
        task_id="WR-08", seed=int(metadata["seed"]),
        variant=metadata["variant"], split=metadata["split"],
        env_split=metadata["env_split"],
        options=dict(resolved["options"]),
        task_config_file=Path(resolved["task_config_file"]),
        observation_profile_file=Path(resolved["observation_profile_file"]),
        allow_partial_calibration=bool(resolved["allow_partial_calibration"]),
    ))
    if episode.identity != metadata["episode_identity"]:
        raise ValueError("Use the source recording's task configuration for this probe")
    api = resolve_builtin_task_plugin("WR-08").load_runtime_api()
    descriptor = api.build_descriptor(episode, execute=True)
    descriptor = replace(descriptor, sensors=replace(
        descriptor.sensors, use_camera_observations=False,
        has_renderer=False, has_offscreen_renderer=False,
    ))
    runtime = RoboCasaRuntimeFactory(api).create_from_descriptor(descriptor, expert=False)
    import h5py
    import mujoco

    try:
        env = runtime.env
        model, data = env.sim.model, env.sim.data
        robot = env.robots[0]
        controller = robot.composite_controller.part_controllers["right"]
        arm_start, arm_end = robot.composite_controller._action_split_indexes["right"]
        actions = load_recorded_actions(args.episode_dir / "trajectory.h5")
        with h5py.File(args.episode_dir / "trajectory.h5", "r") as handle:
            recorded_states = np.asarray(handle["data/demo_1/obs/robot_state"])
        extractor = RobotStateExtractor()
        geom_names = [model.geom_id2name(index) or str(index) for index in range(model.ngeom)]
        rows = []
        for step, action in enumerate(actions, start=1):
            obs = env._get_observations(force_update=True)
            state_error = float(np.max(np.abs(
                extractor.extract(env, obs)[:30] - recorded_states[step - 1, :30]
            )))
            obs = env.step(action)[0]
            contacts = []
            for index in range(data.ncon):
                contact = data.contact[index]
                pair = [geom_names[int(contact.geom1)], geom_names[int(contact.geom2)]]
                if not any(name.startswith(("robot", "gripper")) for name in pair):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model._model, data._data, index, force)
                if force[0] < 0.1:
                    continue
                contacts.append({
                    "geoms": pair, "distance_m": float(contact.dist),
                    "normal_force_n": float(force[0]),
                })
            evidence = {item: env._item_physical_grasp_evidence(item) for item in env.item_names}
            qpos = np.asarray(data.qpos[controller.qpos_index])
            rows.append({
                "step": step,
                "pre_step_state30_max_abs_replay_error": state_error,
                "action": action.tolist(),
                "arm_qpos": qpos.tolist(),
                "arm_error": (qpos - action[arm_start:arm_end]).tolist(),
                "base_pos": np.asarray(obs["robot0_base_pos"]).tolist(),
                "base_quat": np.asarray(obs["robot0_base_quat"]).tolist(),
                "eef_pos": np.asarray(data.site_xpos[robot.eef_site_id["right"]]).tolist(),
                "items": {
                    item: {
                        "position": env._obj_pos(item).tolist(),
                        "velocity": np.asarray(data.get_joint_qvel(env.objects[item].joints[0])).tolist(),
                        "held": bool(value["hold"]["held"]),
                        "gripper_qpos": value["gripper_qpos"],
                        "eef_distance": value["hold"]["end_effector_distance"],
                    }
                    for item, value in evidence.items()
                },
                "contacts": contacts,
            })
        payload = {
            "seed": episode.seed, "episode_dir": str(args.episode_dir),
            "episode_identity": episode.identity,
            "diagnostic_only": True, "camera_rendering": False,
            "state30_max_abs_replay_error": max(
                row["pre_step_state30_max_abs_replay_error"] for row in rows
            ),
            "step_count": len(rows), "final_diagnostics": runtime.diagnostics().details,
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"seed": episode.seed, "steps": len(rows), "output": str(args.output)}))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
