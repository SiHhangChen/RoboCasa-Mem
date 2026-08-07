#!/usr/bin/env python3
"""Instantiate TM04 and report the combined layout-15 scene geometry."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.env.robocasa_runtime import (  # noqa: E402
    bootstrap_local_robocasa_sources,
    make_joint_position_controller_config,
    make_robocasa_env,
    reset_env_with_initial_arm_qpos,
)
from bench.tasks.tm04.config import load_task_config, resolve_observer_config  # noqa: E402
from bench.tasks.tm04.domain.episode import resolve_episode_plan, resolve_scene_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-stride", type=int, default=4)
    parser.add_argument("--run-dry-expert", action="store_true")
    parser.add_argument("--run-expert", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.video_dir is not None and not args.run_expert:
        raise ValueError("--video-dir requires --run-expert")
    bootstrap_local_robocasa_sources()
    from bench.tasks.tm04.environment import (  # noqa: PLC0415
        TM04FillPotRetrieveCanAndAddTwoSugars,
    )
    from bench.tasks.tm02.environment import (  # noqa: PLC0415
        DIAGNOSTIC_OVERVIEW_CAMERA,
        DIAGNOSTIC_TOP_CAMERA,
    )

    del TM04FillPotRetrieveCanAndAddTwoSugars
    config = load_task_config()
    scene = resolve_scene_config(env_split="pretrain", **dict(config.scene))
    plan = resolve_episode_plan(seed=args.seed, **dict(config.episode))
    observer = resolve_observer_config(config)
    env = make_robocasa_env(
        env_name=scene.env_name,
        split=scene.env_split,
        seed=scene.scene_seed,
        controller_config=make_joint_position_controller_config(),
        env_kwargs={
            "water_button_world_xyz": scene.water_button_world_xyz,
            "completion_button_world_xyz": scene.completion_button_world_xyz,
            "cabinet_world_xyz": scene.cabinet_world_xyz,
            "cabinet_yaw_deg": scene.cabinet_yaw_deg,
            "can_place_region_world_xyz": scene.can_place_region_world_xyz,
            "can_place_region_size_xy": scene.can_place_region_size_xy,
            "water_pot_name": scene.water_pot_name,
            "can_name": scene.can_name,
            "sugar_plate_name": scene.sugar_plate_name,
            "hot_chocolate_name": scene.hot_chocolate_name,
            "sugar_names": scene.sugar_names,
            "sugar_plate_world_xyz": scene.sugar_plate_world_xyz,
            "hot_chocolate_world_xyz": scene.hot_chocolate_world_xyz,
            "target_sugar_count": plan.target_sugar_count,
            "total_sugar_count": plan.total_sugar_count,
            "selected_sugar_names": plan.selected_sugar_names,
            "observer_config": asdict(observer),
        },
        camera_names=(
            "robot0_agentview_right",
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ),
        camera_widths=int(args.width),
        camera_heights=int(args.height),
        layout_id=scene.layout_id,
        style_id=scene.style_id,
        has_renderer=False,
        has_offscreen_renderer=(
            args.image_dir is not None or args.video_dir is not None
        ),
        use_camera_obs=False,
        camera_depths=False,
    )
    video_recorder: _ExpertVideoRecorder | None = None
    try:
        observation = reset_env_with_initial_arm_qpos(env)
        for _ in range(max(0, int(args.settle_steps))):
            env.sim.step()
        env.sim.forward()
        diagnostics = env.task_diagnostics()
        report = {
            "environment_class": type(env).__name__,
            "layout_id": int(env.layout_id),
            "style_id": int(env.style_id),
            "route_id": plan.route_id,
            "selected_sugar_names": list(plan.selected_sugar_names),
            "objects": {
                name: env.object_world_position(name).tolist()
                for name in (
                    scene.water_pot_name,
                    scene.can_name,
                    scene.sugar_plate_name,
                    *scene.sugar_names,
                    scene.hot_chocolate_name,
                )
            },
            "sugars_in_hot_chocolate": env.sugars_in_hot_chocolate(),
            "hot_chocolate_stable": env.hot_chocolate_stable(),
            "observer_initial_state_valid": not bool(
                env.physical_observer.invalid_environment_labels
            ),
            "invalid_environment_labels": list(
                env.physical_observer.invalid_environment_labels
            ),
            "task_success_at_reset": bool(diagnostics["success"]),
            "diagnostics": diagnostics,
        }
        if args.image_dir is not None:
            report["rendered_images"] = _render_cameras(
                env,
                output_dir=args.image_dir,
                width=int(args.width),
                height=int(args.height),
                cameras=(
                    DIAGNOSTIC_OVERVIEW_CAMERA,
                    DIAGNOSTIC_TOP_CAMERA,
                    "robot0_agentview_right",
                    "robot0_agentview_left",
                ),
            )
        if args.run_dry_expert and args.run_expert:
            raise ValueError("choose at most one of --run-dry-expert and --run-expert")
        if args.run_dry_expert or args.run_expert:
            key = "expert" if args.run_expert else "dry_expert"
            if args.video_dir is not None:
                video_recorder = _ExpertVideoRecorder(
                    env,
                    output_dir=args.video_dir,
                    cameras=(
                        DIAGNOSTIC_TOP_CAMERA,
                        "robot0_agentview_left",
                        "robot0_agentview_right",
                    ),
                    width=int(args.video_width),
                    height=int(args.video_height),
                    fps=int(args.video_fps),
                    stride=int(args.video_stride),
                )
                video_recorder.install()
                video_recorder.capture(repeat=max(1, int(args.video_fps) // 2))
            try:
                report[key] = _run_expert(
                    env,
                    observation=dict(observation),
                    seed=int(args.seed),
                    execute=bool(args.run_expert),
                )
            finally:
                if video_recorder is not None:
                    video_recorder.capture(repeat=max(1, int(args.video_fps)))
                    report["video_recording"] = video_recorder.close()
            report["diagnostics"] = env.task_diagnostics()
        report["contract_pass"] = bool(
            report["environment_class"]
            == "TM04FillPotRetrieveCanAndAddTwoSugars"
            and report["layout_id"] == 15
            and report["style_id"] == 34
            and not any(report["sugars_in_hot_chocolate"].values())
            and report["hot_chocolate_stable"]
            and report["observer_initial_state_valid"]
            and not report["task_success_at_reset"]
            and all(
                bool(report[key]["success"])
                for key in ("dry_expert", "expert")
                if key in report
            )
        )
    finally:
        if video_recorder is not None and video_recorder.installed:
            video_recorder.close()
        close = getattr(env, "close", None)
        if callable(close):
            close()

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["contract_pass"] else 1


class _ExpertVideoRecorder:
    """Record action-aligned TM04 expert views without changing task control."""

    def __init__(
        self,
        env: Any,
        *,
        output_dir: Path,
        cameras: tuple[str, str, str],
        width: int,
        height: int,
        fps: int,
        stride: int,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("video width and height must be positive even integers")
        if fps <= 0 or stride <= 0:
            raise ValueError("video fps and stride must be positive")
        self.env = env
        self.output_dir = Path(output_dir)
        self.cameras = cameras
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.stride = int(stride)
        self.env_step_count = 0
        self.frame_count = 0
        self._original_step: Any | None = None
        self._writers: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}

    @property
    def installed(self) -> bool:
        return self._original_step is not None

    def install(self) -> None:
        if self.installed:
            return
        import imageio.v2 as imageio

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stream_names = {
            self.cameras[0]: "tm04_top",
            self.cameras[1]: "tm04_agentview_left",
            self.cameras[2]: "tm04_agentview_right",
            "multiview": "tm04_multiview",
        }
        for stream, stem in stream_names.items():
            path = (self.output_dir / f"{stem}.mp4").resolve()
            self._paths[stream] = path
            self._writers[stream] = imageio.get_writer(
                path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=None,
                ffmpeg_log_level="error",
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )

        self._original_step = self.env.step

        def wrapped_step(action: Any):
            result = self._original_step(action)
            self.env_step_count += 1
            if self.env_step_count % self.stride == 0:
                self.capture()
            return result

        self.env.step = wrapped_step

    def capture(self, *, repeat: int = 1) -> None:
        if not self.installed:
            raise RuntimeError("video recorder must be installed before capture")
        import cv2
        import numpy as np

        frames: dict[str, np.ndarray] = {}
        for camera in self.cameras:
            frames[camera] = np.asarray(
                self.env.sim.render(
                    camera_name=camera,
                    width=self.width,
                    height=self.height,
                    depth=False,
                ),
                dtype=np.uint8,
            )[::-1].copy()
        multiview = _compose_multiview(
            frames[self.cameras[0]],
            frames[self.cameras[1]],
            frames[self.cameras[2]],
            cv2=cv2,
        )
        resolved_repeat = max(1, int(repeat))
        for _ in range(resolved_repeat):
            for camera in self.cameras:
                self._writers[camera].append_data(frames[camera])
            self._writers["multiview"].append_data(multiview)
        self.frame_count += resolved_repeat

    def close(self) -> dict[str, Any]:
        if self._original_step is not None:
            self.env.step = self._original_step
            self._original_step = None
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        return {
            "codec": "h264",
            "fps": self.fps,
            "capture_stride": self.stride,
            "frame_count": self.frame_count,
            "env_step_count": self.env_step_count,
            "duration_seconds": self.frame_count / self.fps,
            "individual_resolution": [self.width, self.height],
            "multiview_resolution": [self.width + self.width // 2, self.height],
            "videos": {
                stream: str(path) for stream, path in self._paths.items()
            },
        }


def _compose_multiview(top, left, right, *, cv2):
    import numpy as np

    height, width = top.shape[:2]
    side_size = (width // 2, height // 2)
    left_small = cv2.resize(left, side_size, interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, side_size, interpolation=cv2.INTER_AREA)
    top_labeled = _label_frame(top.copy(), "TOP", cv2=cv2)
    left_labeled = _label_frame(left_small, "AGENT LEFT", cv2=cv2)
    right_labeled = _label_frame(right_small, "AGENT RIGHT", cv2=cv2)
    side = np.concatenate((left_labeled, right_labeled), axis=0)
    return np.concatenate((top_labeled, side), axis=1)


def _label_frame(frame, label: str, *, cv2):
    cv2.rectangle(frame, (8, 8), (166, 38), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        label,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _run_expert(env, *, observation: dict, seed: int, execute: bool) -> dict:
    from bench.tasks import resolve_builtin_task_plugin
    from bench.tasks.contracts import EpisodeRequest, ExpertPlanInputs
    from bench.tasks.tm04.episode import resolve_episode
    from bench.tasks.tm04.expert.spec import build_expert_plan

    plugin = resolve_builtin_task_plugin("TM04")
    api = plugin.load_runtime_api()
    episode = resolve_episode(EpisodeRequest(task_id="TM04", seed=int(seed)))
    descriptor = api.build_descriptor(episode, execute=bool(execute))
    runtime = api.initialize_runtime(descriptor, env, observation, {})
    expert_runtime = api.build_expert_runtime(
        descriptor,
        env,
        dict(runtime.observation),
    )
    plan = build_expert_plan(
        ExpertPlanInputs(
            episode=episode,
            scene=descriptor.scene,
            episode_plan=descriptor.episode_plan,
            action_config=descriptor.actions,
            execute=bool(execute),
        )
    )
    result = expert_runtime["runner"].run(plan)
    return {
        "success": bool(result.success),
        "execute": bool(execute),
        "route": list(plan.metadata["route"]),
        "planned_stage_count": len(plan.stages),
        "completed_stage_count": len(result.stages),
        "failed_stage": (
            None
            if result.success or not result.stages
            else {
                "index": int(result.stages[-1].index),
                "failure_code": result.stages[-1].failure_code,
                "details": dict(result.stages[-1].details),
            }
        ),
    }


def _render_cameras(
    env,
    *,
    output_dir: Path,
    width: int,
    height: int,
    cameras: tuple[str, ...],
) -> dict[str, str]:
    import imageio.v2 as imageio
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}
    for camera in cameras:
        rgb = np.asarray(
            env.sim.render(
                camera_name=camera,
                width=int(width),
                height=int(height),
                depth=False,
            ),
            dtype=np.uint8,
        )[::-1]
        filename = (
            "tm04_scene_overview.png"
            if camera == "bench_tm02_overview"
            else f"tm04_{camera}.png"
        )
        path = (output_dir / filename).resolve()
        imageio.imwrite(path, rgb)
        result[camera] = str(path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
