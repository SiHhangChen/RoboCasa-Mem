"""Native MobileMemBench v2 episode writer.

The writer receives only policy-safe structured samples.  It never serializes
the complete RoboCasa observation mapping, so task truth cannot accidentally
become a policy feature.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from bench.execution.planning.model_alignment import mat_to_quat_xyzw
from bench.runtime.data_contract import (
    NATIVE_EPISODE_SCHEMA,
    QUATERNION_ORDER,
    QUATERNION_SIGN_POLICY,
    action_schema_metadata,
    contract_metadata,
    validate_action,
)
from bench.runtime.robot_state import RobotStateExtractor


CAMERA_TRAJECTORY_FIELD_SUFFIXES = (
    "intrinsic",
    "extrinsic_c2w",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")


def _safe_filename_part(value: Any) -> str:
    text = str(value or "").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "camera"


def _is_depth_key(key: str, value: Any | None = None) -> bool:
    if not str(key).endswith("_depth"):
        return False
    if value is None:
        return True
    array = np.asarray(value)
    return array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 1)


class EpisodeRecorder:
    """Capture action-aligned policy frames around an expert rollout."""

    def __init__(
        self,
        env: Any,
        *,
        output_dir: Path,
        cameras: list[str] | tuple[str, ...],
        camera_width: int,
        camera_height: int,
        fps: int = 20,
        record_depth: bool = True,
        depth_dtype: str = "uint16",
        depth_scale: float = 0.001,
        env_meta: dict[str, Any] | None = None,
        env_name: str | None = None,
    ) -> None:
        self.env = env
        self.output_dir = Path(output_dir)
        self.cameras = tuple(str(camera) for camera in cameras)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.fps = int(fps)
        self.record_depth = bool(record_depth)
        self.depth_dtype = str(depth_dtype).lower()
        self.depth_scale = float(depth_scale)
        self.env_meta = dict(env_meta or {})
        self.env_name = env_name
        if self.depth_dtype != "uint16":
            raise ValueError("native v2 depth dtype must be uint16")
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0.0:
            raise ValueError("depth_scale must be positive and finite")
        if len(self.cameras) != 3:
            raise ValueError("native v2 recording requires exactly three cameras")
        self.records: list[dict[str, Any]] = []
        self.frame_counts = {camera: 0 for camera in self.cameras}
        self.started_at = time.time()
        self._original_step: Any | None = None
        self._writers: dict[str, Any] = {}
        self._current_obs: dict[str, Any] | None = None
        self._state_extractor = RobotStateExtractor()
        self._termination: dict[str, Any] = {
            "terminated": False,
            "truncated": False,
            "success": False,
            "reason": "not_finalized",
        }
        self._final_diagnostics: dict[str, Any] = {}
        self._final_oracle: dict[str, Any] = {}

    @property
    def installed(self) -> bool:
        return self._original_step is not None

    def install(self) -> None:
        if self.installed:
            return
        import imageio.v2 as imageio

        video_dir = self.output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        for camera in self.cameras:
            self._writers[camera] = imageio.get_writer(
                video_dir / f"{camera}.mp4", fps=self.fps
            )

        self._original_step = self.env.step

        def wrapped_step(action: Any):
            pre_obs = self._current_obs or self._fresh_observation()
            if pre_obs is None:
                raise RuntimeError("cannot record action without a pre-step observation")
            action_value = validate_action(action)
            pre_frame = self._capture_frame(pre_obs)
            result = self._original_step(action_value)
            obs, reward, terminated, truncated, info, five_tuple = self._normalize_step(result)
            hook = getattr(self.env, "_bench_post_step_observation_hook", None)
            if callable(hook):
                updated = hook(obs)
                if updated is not None:
                    obs = updated
                self.env._bench_post_step_observation_hook_consumed = True
            self.record_step(
                action=action_value,
                frame=pre_frame,
                reward=reward,
                info=info,
                raw_terminated=terminated,
                raw_truncated=truncated,
                raw_done=bool(terminated or truncated),
                next_obs=obs,
            )
            if five_tuple:
                return obs, reward, terminated, truncated, info
            return obs, reward, bool(terminated or truncated), info

        self.env.step = wrapped_step

    def close(self) -> None:
        if self._original_step is not None:
            self.env.step = self._original_step
            self._original_step = None
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def record_initial_observation(self, obs: Mapping[str, Any]) -> None:
        self._current_obs = dict(obs or {})

    def record_stage_observation(self, obs: Mapping[str, Any]) -> None:
        # Stage observations are useful for diagnostics/snapshots but must not
        # add duplicate frames to an action-aligned policy video.
        self._current_obs = dict(obs or {})

    def record_step(
        self,
        *,
        action: Any,
        frame: Mapping[str, Any] | None = None,
        obs: Mapping[str, Any] | None = None,
        next_obs: Mapping[str, Any] | None = None,
        reward: float = 0.0,
        info: Mapping[str, Any] | None = None,
        raw_terminated: bool = False,
        raw_truncated: bool = False,
        raw_done: bool = False,
        done: bool | None = None,
        terminated: bool | None = None,
        truncated: bool | None = None,
        success: bool = False,
    ) -> None:
        if frame is None:
            if obs is None:
                raise ValueError("record_step requires frame or obs")
            frame = self._capture_frame(dict(obs))
        value = validate_action(action)
        record = {
            "step": len(self.records),
            "action": value,
            "reward": float(reward),
            "frame": dict(frame),
            "info": jsonable(info or {}),
            "raw_terminal": {
                "terminated": bool(raw_terminated),
                "truncated": bool(raw_truncated),
                "done": bool(raw_done if done is None else done),
            },
            "terminated": bool(False if terminated is None else terminated),
            "truncated": bool(False if truncated is None else truncated),
            "success": bool(success),
            "video_frame_indices": {},
        }
        self.records.append(record)
        record["video_frame_indices"] = self._write_video_frames(frame)
        stored_frame = record["frame"]
        stored_frame.pop("images", None)
        if not self.record_depth:
            stored_frame.pop("depth", None)
        self._current_obs = dict(next_obs or obs or {})

    def finalize_terminal(
        self,
        *,
        terminated: bool,
        truncated: bool,
        success: bool,
        reason: str,
    ) -> None:
        if bool(terminated) and bool(truncated):
            raise ValueError("terminal transition cannot be both terminated and truncated")
        if not self.records:
            self._termination = {
                "terminated": False,
                "truncated": False,
                "success": False,
                "reason": str(reason),
            }
            return
        for record in self.records:
            record["terminated"] = False
            record["truncated"] = False
            record["success"] = False
        self.records[-1]["terminated"] = bool(terminated)
        self.records[-1]["truncated"] = bool(truncated)
        self.records[-1]["success"] = bool(success)
        self._termination = {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(success),
            "reason": str(reason),
        }

    def set_final_diagnostics(self, diagnostics: Mapping[str, Any]) -> None:
        """Store task truth outside policy observations.

        Task diagnostics may contain an explicit ``oracle`` subtree.  It is
        persisted under the quarantined oracle group; all remaining fields are
        diagnostics.  Neither channel is read by the LeRobot exporter.
        """

        value = jsonable(diagnostics or {})
        if not isinstance(value, dict):
            value = {"value": value}
        self._final_oracle = dict(value.pop("oracle", {}) or {})
        self._final_diagnostics = value

    def write_stage_snapshot(
        self,
        *,
        stage_index: int,
        stage_kind: str,
        label: str,
        obs: Mapping[str, Any] | None = None,
        subdir: str = "navigation",
    ) -> dict[str, str]:
        if obs is None:
            obs = self._fresh_observation() or {}
        if not self.cameras:
            return {}
        import imageio.v2 as imageio

        root = self.output_dir / "snapshots" / _safe_filename_part(subdir)
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for camera in self.cameras:
            key = f"{camera}_image"
            if key not in obs:
                continue
            path = root / (
                f"stage_{int(stage_index):03d}_{_safe_filename_part(stage_kind)}_"
                f"{_safe_filename_part(label)}_{_safe_filename_part(camera)}.png"
            )
            imageio.imwrite(path, np.asarray(obs[key]))
            paths[camera] = str(path.relative_to(self.output_dir))
        return paths

    def write_episode(
        self,
        *,
        metadata: dict[str, Any],
        planner_summary: dict[str, Any],
        accepted: bool,
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        planner_path = self.output_dir / "planner.json"
        episode_path = self.output_dir / "episode.json"
        events_path = self.output_dir / "events.jsonl"
        trajectory_path = self.output_dir / "trajectory.h5"
        write_json(planner_path, planner_summary)
        self._write_events(events_path, planner_summary)
        h5_summary = self._write_hdf5(trajectory_path, metadata=metadata)
        episode = {
            "schema_version": NATIVE_EPISODE_SCHEMA,
            "metadata": metadata,
            "accepted": bool(accepted),
            "planner_pass": bool(planner_summary.get("pass", False)),
            "step_count": len(self.records),
            "started_at_unix": self.started_at,
            "elapsed_s": time.time() - self.started_at,
            "contract": contract_metadata(),
            "termination": dict(self._termination),
            "files": {
                "episode": str(episode_path),
                "trajectory": str(trajectory_path),
                "planner": str(planner_path),
                "events": str(events_path),
                "videos": {
                    camera: str(self.output_dir / "videos" / f"{camera}.mp4")
                    for camera in self.cameras
                },
            },
            "video": {
                "cameras": list(self.cameras),
                "fps": self.fps,
                "frame_stride": 1,
                "frame_counts": dict(self.frame_counts),
            },
            "quaternion": {
                "order": QUATERNION_ORDER,
                "sign_policy": QUATERNION_SIGN_POLICY,
                "scope": "per_episode_per_state_field",
            },
            "trajectory_h5": h5_summary,
        }
        write_json(episode_path, episode)
        return episode

    def _capture_frame(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        state = self._state_extractor.extract(self.env, obs)
        images: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        for camera in self.cameras:
            image_key = f"{camera}_image"
            depth_key = f"{camera}_depth"
            if image_key not in obs:
                raise KeyError(f"recording requires RGB observation {image_key!r}")
            image = np.asarray(obs[image_key])
            if image.shape != (self.camera_height, self.camera_width, 3):
                raise ValueError(
                    f"{image_key} expected {(self.camera_height, self.camera_width, 3)}, "
                    f"got {image.shape}"
                )
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            images[camera] = np.ascontiguousarray(image)
            if self.record_depth:
                if depth_key not in obs:
                    raise KeyError(f"recording requires depth observation {depth_key!r}")
                depths[camera] = self._normalized_depth_to_storage(obs[depth_key])
        calibration = self._camera_calibration(obs)
        timestamp = self._simulation_time()
        return {
            "state": state,
            "images": images,
            "depth": depths,
            "calibration": calibration,
            "sim_time": timestamp,
        }

    def _camera_calibration(self, obs: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
        sim = getattr(self.env, "sim", None)
        if sim is None:
            raise RuntimeError("camera calibration requires env.sim")
        try:
            from robosuite.utils.camera_utils import (
                get_camera_extrinsic_matrix,
                get_camera_intrinsic_matrix,
            )
        except Exception as exc:
            raise RuntimeError("RoboSuite camera calibration utilities unavailable") from exc
        result: dict[str, dict[str, np.ndarray]] = {}
        for camera in self.cameras:
            extrinsic = np.asarray(get_camera_extrinsic_matrix(sim, camera), dtype=np.float32)
            if extrinsic.shape != (4, 4):
                raise ValueError(f"camera {camera} extrinsic must be [4,4]")
            intrinsic = np.asarray(
                get_camera_intrinsic_matrix(
                    sim,
                    camera,
                    camera_height=self.camera_height,
                    camera_width=self.camera_width,
                ),
                dtype=np.float32,
            )
            if intrinsic.shape != (3, 3):
                raise ValueError(f"camera {camera} intrinsic must be [3,3]")
            result[camera] = {
                "intrinsic": np.ascontiguousarray(intrinsic),
                "extrinsic_c2w": np.ascontiguousarray(extrinsic),
                "quat_xyzw_world": np.asarray(mat_to_quat_xyzw(extrinsic[:3, :3]), dtype=np.float32),
            }
        return result

    def _normalized_depth_to_storage(self, value: Any) -> np.ndarray:
        depth = np.asarray(value, dtype=np.float64)
        if depth.ndim == 2:
            depth = depth[..., None]
        if depth.shape != (self.camera_height, self.camera_width, 1):
            raise ValueError(f"depth expected {(self.camera_height, self.camera_width, 1)}, got {depth.shape}")
        finite = np.isfinite(depth)
        if finite.any() and (float(np.nanmin(depth)) < -1e-6 or float(np.nanmax(depth)) > 1.0 + 1e-6):
            raise ValueError("RoboSuite depth must be normalized to [0,1]")
        sim = getattr(self.env, "sim", None)
        model = getattr(sim, "model", None)
        if model is None:
            raise RuntimeError("metric depth conversion requires env.sim.model")
        extent = float(model.stat.extent)
        near = float(model.vis.map.znear) * extent
        far = float(model.vis.map.zfar) * extent
        if not (np.isfinite(near) and np.isfinite(far) and 0.0 < near < far):
            raise ValueError(f"invalid depth range near={near}, far={far}")
        normalized = np.clip(np.where(finite, depth, 1.0), 0.0, 1.0)
        metric = near / (1.0 - normalized * (1.0 - near / far))
        valid = finite & (normalized < 1.0 - 1e-7) & (metric > 0.0)
        output = np.zeros(depth.shape, dtype=np.uint16)
        if valid.any():
            scaled = np.clip(np.rint(metric[valid] / self.depth_scale), 1, np.iinfo(np.uint16).max)
            output[valid] = scaled.astype(np.uint16)
        return np.ascontiguousarray(output)

    def _write_video_frames(self, frame: Mapping[str, Any]) -> dict[str, int]:
        indexes: dict[str, int] = {}
        images = frame["images"]
        for camera, writer in self._writers.items():
            indexes[camera] = self.frame_counts[camera]
            writer.append_data(np.asarray(images[camera]))
            self.frame_counts[camera] += 1
        return indexes

    def _simulation_time(self) -> float:
        try:
            return float(self.env.sim.data.time)
        except Exception:
            return float(len(self.records)) / float(self.fps)

    def _fresh_observation(self) -> dict[str, Any] | None:
        getter = getattr(self.env, "_get_observations", None)
        if not callable(getter):
            return None
        try:
            value = getter(force_update=True)
        except TypeError:
            value = getter()
        return dict(value or {})

    @staticmethod
    def _normalize_step(result: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any], bool]:
        if not isinstance(result, tuple):
            raise TypeError("env.step() must return a tuple")
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            five_tuple = True
        elif len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
            five_tuple = False
        else:
            raise ValueError(f"unsupported env.step() result length: {len(result)}")
        if not isinstance(obs, Mapping):
            raise TypeError("env.step() observation must be a mapping")
        return dict(obs), float(reward), bool(terminated), bool(truncated), dict(info or {}), five_tuple

    @staticmethod
    def _write_events(path: Path, planner_summary: Mapping[str, Any]) -> None:
        events: list[dict[str, Any]] = []
        for source, key in (("rule_policy.operation", "rule_policy"), ("rollout.stage", "rollout")):
            payload = planner_summary.get(key) or {}
            values = payload.get("operations" if key == "rule_policy" else "stages", [])
            events.extend({"source": source, "payload": value} for value in values or [])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(jsonable(event), sort_keys=True) + "\n")

    def _write_hdf5(self, path: Path, *, metadata: Mapping[str, Any]) -> dict[str, Any]:
        try:
            import h5py
        except Exception as exc:
            return {"written": False, "path": str(path), "reason": repr(exc)}
        if not self.records:
            raise ValueError("cannot write an empty native v2 episode")
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            data.attrs["schema_version"] = NATIVE_EPISODE_SCHEMA
            data.attrs["contract_json"] = json.dumps(contract_metadata(), sort_keys=True)
            data.attrs["action_schema_json"] = json.dumps(action_schema_metadata(), sort_keys=True)
            data.attrs["camera_names"] = json.dumps(list(self.cameras))
            data.attrs["control_frequency_hz"] = 20
            data.attrs["quaternion_order"] = QUATERNION_ORDER
            data.attrs["quaternion_sign_policy"] = QUATERNION_SIGN_POLICY
            data.attrs["image_origin"] = "top_left"
            data.attrs["camera_convention"] = "opencv"
            data.attrs["env_args"] = json.dumps(jsonable(self.env_meta), sort_keys=True)
            demo = data.create_group("demo_1")
            demo.attrs["ep_meta"] = json.dumps(jsonable(metadata), sort_keys=True)
            actions = np.stack([record["action"] for record in self.records], axis=0).astype(np.float32)
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("rewards", data=np.asarray([record["reward"] for record in self.records], dtype=np.float32))
            demo.create_dataset("terminated", data=np.asarray([record["terminated"] for record in self.records], dtype=bool))
            demo.create_dataset("truncated", data=np.asarray([record["truncated"] for record in self.records], dtype=bool))
            demo.create_dataset("dones", data=np.asarray([record["terminated"] or record["truncated"] for record in self.records], dtype=bool))
            demo.create_dataset("success", data=np.asarray([record["success"] for record in self.records], dtype=bool))
            obs_group = demo.create_group("obs")
            obs_group.create_dataset("robot_state", data=np.stack([record["frame"]["state"] for record in self.records], axis=0).astype(np.float32))
            depth_group = obs_group.create_group("depth")
            calibration_group = demo.create_group("calibration")
            timestamp_group = demo.create_group("timestamps")
            timestamp_group.create_dataset("sim_time", data=np.asarray([record["frame"]["sim_time"] for record in self.records], dtype=np.float64))
            index_group = demo.create_group("video_frame_indices")
            for camera in self.cameras:
                if self.record_depth:
                    depth = depth_group.create_dataset(camera, data=np.stack([record["frame"]["depth"][camera] for record in self.records], axis=0), compression="gzip", shuffle=True)
                    depth.attrs["unit"] = "millimeter"
                    depth.attrs["scale_meters_per_unit"] = self.depth_scale
                    depth.attrs["invalid_value"] = 0
                    depth.attrs["image_origin"] = "top_left"
                camera_group = calibration_group.create_group(camera)
                intrinsic = camera_group.create_dataset("intrinsic", data=np.stack([record["frame"]["calibration"][camera]["intrinsic"] for record in self.records], axis=0).astype(np.float32))
                extrinsic = camera_group.create_dataset("extrinsic_c2w", data=np.stack([record["frame"]["calibration"][camera]["extrinsic_c2w"] for record in self.records], axis=0).astype(np.float32))
                intrinsic.attrs["camera_convention"] = "opencv"
                intrinsic.attrs["image_origin"] = "top_left"
                extrinsic.attrs["camera_convention"] = "opencv"
                extrinsic.attrs["transform_direction"] = "camera_to_world"
                index_group.create_dataset(camera, data=np.asarray([record["video_frame_indices"][camera] for record in self.records], dtype=np.int64))
            diagnostic_group = demo.create_group("diagnostics")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            diagnostic_group.create_dataset("frame_json", data=np.asarray([json.dumps(record["info"], sort_keys=True) for record in self.records], dtype=object), dtype=string_dtype)
            diagnostic_group.create_dataset("final_json", data=json.dumps(self._final_diagnostics, sort_keys=True), dtype=string_dtype)
            oracle_group = demo.create_group("oracle")
            oracle_group.create_dataset("frame_json", data=np.asarray(["{}"] * len(self.records), dtype=object), dtype=string_dtype)
            oracle_group.create_dataset("final_json", data=json.dumps(self._final_oracle, sort_keys=True), dtype=string_dtype)
            demo.attrs["num_samples"] = len(self.records)
            demo.attrs["termination_reason"] = self._termination["reason"]
        return {
            "written": True,
            "path": str(path),
            "schema": NATIVE_EPISODE_SCHEMA,
            "step_count": len(self.records),
            "actions": list(actions.shape),
            "cameras": list(self.cameras),
            "terminal": dict(self._termination),
        }


__all__ = ["EpisodeRecorder", "jsonable", "write_json"]
