"""Convert recorded bench episodes into a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[2]
LEROBOT_SRC = ROOT / "policy" / "lerobot" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

import cv2
import h5py
import numpy as np

from bench.runtime.data_contract import (
    ACTION_AXES,
    ACTION_DIM,
    ACTION_SCHEMA,
    NATIVE_EPISODE_SCHEMA,
    STATE_DIM,
    STATE_SCHEMA,
    action_schema_metadata,
    contract_metadata,
    state_schema_metadata,
)

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


STATE_NAMES = tuple(
    [f"arm_joint_qpos_{i}" for i in range(7)]
    + [f"arm_joint_qvel_{i}" for i in range(7)]
    + [f"gripper_qpos_{i}" for i in range(2)]
    + [f"gripper_qvel_{i}" for i in range(2)]
    + [f"base_pos_world_{i}" for i in range(3)]
    + ["base_quat_world_x", "base_quat_world_y", "base_quat_world_z", "base_quat_world_w"]
    + [f"base_joint_qvel_{i}" for i in range(3)]
    + ["torso_qpos", "torso_qvel"]
    + [f"eef_pos_in_base_{i}" for i in range(3)]
    + ["eef_quat_in_base_x", "eef_quat_in_base_y", "eef_quat_in_base_z", "eef_quat_in_base_w"]
)

AUTO_FEATURE_KEYS = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}
CAMERA_GEOMETRY_FEATURE_PREFIXES = (
    "observation.depth.",
    "observation.camera_intrinsics.",
    "observation.camera_extrinsics_c2w.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--robot-type", default="mobile_manipulator")
    parser.add_argument("--write-mode", choices=["create", "append"], default="create")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def camera_feature_name(camera: str) -> str:
    return f"observation.images.{camera}"


def camera_depth_feature_name(camera: str) -> str:
    return f"observation.depth.{camera}"


def camera_intrinsic_feature_name(camera: str) -> str:
    return f"observation.camera_intrinsics.{camera}"


def camera_extrinsic_c2w_feature_name(camera: str) -> str:
    return f"observation.camera_extrinsics_c2w.{camera}"


def _safe_camera_name(camera: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(camera)
    )
    return safe.strip("_") or "camera"


def _camera_geometry_dataset_paths(cameras: list[str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for camera in cameras:
        prefix = f"camera_{_safe_camera_name(camera)}"
        paths[camera_depth_feature_name(camera)] = f"obs/depth/{camera}"
        paths[camera_intrinsic_feature_name(camera)] = f"calibration/{camera}/intrinsic"
        paths[camera_extrinsic_c2w_feature_name(camera)] = (
            f"calibration/{camera}/extrinsic_c2w"
        )
    return paths


def _is_camera_geometry_feature(key: str) -> bool:
    return key.startswith(CAMERA_GEOMETRY_FEATURE_PREFIXES)


def episode_metadata(episode_dir: Path) -> dict[str, Any]:
    path = episode_dir / "episode.json"
    if not path.is_file():
        raise FileNotFoundError(f"episode metadata not found: {path}")
    return json.loads(path.read_text())


def episode_task_text(episode_dir: Path, metadata: dict[str, Any]) -> str:
    task_metadata = metadata.get("metadata", {})
    fallback = str(
        task_metadata.get("policy_prompt")
        or task_metadata.get("initial_prompt")
        or task_metadata.get("lang")
        or task_metadata.get("language_instruction")
        or task_metadata.get("instruction")
        or task_metadata.get("display_name")
        or task_metadata["task_id"]
    )
    trajectory_path = episode_dir / "trajectory.h5"
    if not trajectory_path.is_file():
        return fallback
    with h5py.File(trajectory_path, "r") as trajectory:
        demo = _native_demo(trajectory)
        if demo is None:
            return fallback
        raw_ep_meta = demo.attrs.get("ep_meta")
    if raw_ep_meta is None:
        return fallback
    try:
        ep_meta = json.loads(
            raw_ep_meta.decode("utf-8") if isinstance(raw_ep_meta, bytes) else raw_ep_meta
        )
    except (TypeError, json.JSONDecodeError):
        return fallback
    lang = ep_meta.get("lang")
    return str(lang).strip() if isinstance(lang, str) and lang.strip() else fallback


def camera_shape(video_path: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"could not read first video frame: {video_path}")
    return int(frame.shape[0]), int(frame.shape[1]), int(frame.shape[2])


def _native_demo(trajectory: h5py.File) -> h5py.Group | None:
    data = trajectory.get("data")
    if data is None:
        return None
    demos = sorted(key for key in data.keys() if key.startswith("demo_"))
    if not demos:
        return None
    return data[demos[0]]


def _camera_shape_from_episode(episode_dir: Path, camera: str) -> tuple[int, int, int]:
    video_path = episode_dir / "videos" / f"{camera}.mp4"
    if video_path.is_file():
        return camera_shape(video_path)

    trajectory_path = episode_dir / "trajectory.h5"
    with h5py.File(trajectory_path, "r") as trajectory:
        demo = _native_demo(trajectory)
        if demo is None:
            raise FileNotFoundError(
                f"camera video not found and no native demo found for {camera}: "
                f"{video_path}"
            )
        raise FileNotFoundError(f"camera video missing for native episode: {video_path}")


def _attribute_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def camera_geometry_features(
    episode_dir: Path,
    cameras: list[str],
    *,
    image_shapes: dict[str, tuple[int, int, int]],
) -> dict[str, dict[str, Any]]:
    """Describe lossless RGB-D calibration fields present in one native episode."""

    metadata = episode_metadata(episode_dir)
    depth_requested = bool(metadata.get("depth", {}).get("recorded", False)) or bool(
        metadata.get("metadata", {}).get("depth_recorded", False)
    )
    calibration_requested = bool(
        metadata.get("camera_trajectory", {}).get("recorded", False)
    ) or bool(metadata.get("metadata", {}).get("camera_trajectory_recorded", False))
    trajectory_path = episode_dir / "trajectory.h5"
    paths = _camera_geometry_dataset_paths(cameras)
    features: dict[str, dict[str, Any]] = {}

    with h5py.File(trajectory_path, "r") as trajectory:
        demo = _native_demo(trajectory)
        if demo is None:
            if depth_requested:
                raise ValueError(
                    f"RGB-D metadata is enabled but no native demo exists in {trajectory_path}"
                )
            return features

        action_count = int(demo["actions"].shape[0])
        for camera in cameras:
            depth_feature = camera_depth_feature_name(camera)
            intrinsic_feature = camera_intrinsic_feature_name(camera)
            extrinsic_feature = camera_extrinsic_c2w_feature_name(camera)
            depth_path = paths[depth_feature]
            intrinsic_path = paths[intrinsic_feature]
            extrinsic_path = paths[extrinsic_feature]
            present = {
                depth_feature: depth_path in demo,
                intrinsic_feature: intrinsic_path in demo,
                extrinsic_feature: extrinsic_path in demo,
            }

            if depth_requested and not present[depth_feature]:
                raise KeyError(f"required depth dataset missing: {depth_path}")
            if (depth_requested or calibration_requested or any(present.values())) and not (
                present[intrinsic_feature] and present[extrinsic_feature]
            ):
                raise KeyError(
                    "camera calibration must contain both intrinsic and camera_to_world: "
                    f"camera={camera}, present={present}"
                )
            if present[depth_feature] and not (
                present[intrinsic_feature] and present[extrinsic_feature]
            ):
                raise KeyError(
                    f"depth for {camera} cannot be exported without K and camera_to_world"
                )

            if present[depth_feature]:
                depth = demo[depth_path]
                expected_image_shape = image_shapes[camera]
                expected_depth_shape = (
                    action_count,
                    int(expected_image_shape[0]),
                    int(expected_image_shape[1]),
                    1,
                )
                if tuple(depth.shape) != expected_depth_shape:
                    raise ValueError(
                        f"depth/RGB/action shape mismatch for {camera}: "
                        f"depth={tuple(depth.shape)}, expected={expected_depth_shape}"
                    )
                if depth.dtype != np.dtype("uint16"):
                    raise ValueError(
                        f"expected uint16 depth for {camera}, got {depth.dtype}"
                    )
                features[depth_feature] = {
                    "dtype": "uint16",
                    "shape": tuple(depth.shape[1:]),
                    "names": ["height", "width", "channel"],
                    "info": {
                        "is_depth_map": True,
                        "unit": str(_attribute_value(depth.attrs.get("unit", "millimeter"))),
                        "scale_meters_per_unit": float(
                            depth.attrs.get(
                                "scale_meters_per_unit",
                                trajectory["data"].attrs.get(
                                    "depth_scale_meters_per_unit", 0.001
                                ),
                            )
                        ),
                        "invalid_value": int(depth.attrs.get("invalid_value", 0)),
                        "image_origin": str(
                            _attribute_value(depth.attrs.get("image_origin", "top_left"))
                        ),
                    },
                }

            if present[intrinsic_feature]:
                intrinsic = demo[intrinsic_path]
                if tuple(intrinsic.shape) != (action_count, 3, 3):
                    raise ValueError(
                        f"expected intrinsic shape {(action_count, 3, 3)} for {camera}, "
                        f"got {tuple(intrinsic.shape)}"
                    )
                if intrinsic.dtype != np.dtype("float32"):
                    raise ValueError(
                        f"expected float32 intrinsic for {camera}, got {intrinsic.dtype}"
                    )
                features[intrinsic_feature] = {
                    "dtype": "float32",
                    "shape": (3, 3),
                    "names": ["row", "column"],
                    "info": {
                        "camera_convention": str(
                            _attribute_value(
                                intrinsic.attrs.get("camera_convention", "opencv")
                            )
                        ),
                        "image_origin": str(
                            _attribute_value(
                                intrinsic.attrs.get("image_origin", "top_left")
                            )
                        ),
                    },
                }

            if present[extrinsic_feature]:
                extrinsic = demo[extrinsic_path]
                if tuple(extrinsic.shape) != (action_count, 4, 4):
                    raise ValueError(
                        f"expected c2w shape {(action_count, 4, 4)} for {camera}, "
                        f"got {tuple(extrinsic.shape)}"
                    )
                if extrinsic.dtype != np.dtype("float32"):
                    raise ValueError(
                        f"expected float32 c2w for {camera}, got {extrinsic.dtype}"
                    )
                features[extrinsic_feature] = {
                    "dtype": "float32",
                    "shape": (4, 4),
                    "names": ["row", "column"],
                    "info": {
                        "camera_convention": str(
                            _attribute_value(
                                extrinsic.attrs.get("camera_convention", "opencv")
                            )
                        ),
                        "transform_direction": str(
                            _attribute_value(
                                extrinsic.attrs.get(
                                    "transform_direction", "camera_to_world"
                                )
                            )
                        ),
                    },
                }
    return features


def dataset_features(
    episode_dir: Path,
    cameras: list[str],
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": {"axes": list(STATE_NAMES)},
            "info": state_schema_metadata(),
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": {"axes": [axis.name for axis in ACTION_AXES]},
            # LeRobot stores this as static feature metadata.  Keep the
            # action as one vector while preserving every axis slice and
            # controller semantic required to decode it.
            "info": action_schema_metadata(),
        },
        "next.terminated": {"dtype": "bool", "shape": (1,), "names": ["value"]},
        "next.truncated": {"dtype": "bool", "shape": (1,), "names": ["value"]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["value"]},
        "next.success": {"dtype": "bool", "shape": (1,), "names": ["value"]},
    }
    image_shapes = {
        camera: _camera_shape_from_episode(episode_dir, camera) for camera in cameras
    }
    for camera, shape in image_shapes.items():
        features[camera_feature_name(camera)] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
    features.update(
        camera_geometry_features(
            episode_dir,
            cameras,
            image_shapes=image_shapes,
        )
    )
    return features


class CameraFrames:
    def __init__(self, paths: dict[str, Path]):
        self._captures = {
            camera: cv2.VideoCapture(str(path)) for camera, path in paths.items()
        }
        self._positions = {camera: -1 for camera in paths}

    def read(self, frame_indices: dict[str, int]) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for camera, target_index in frame_indices.items():
            if target_index <= self._positions[camera]:
                raise ValueError(
                    f"video frame indices must be strictly increasing for {camera}: "
                    f"{target_index} <= {self._positions[camera]}"
                )
            frame = None
            while self._positions[camera] < target_index:
                ok, frame = self._captures[camera].read()
                if not ok or frame is None:
                    raise ValueError(
                        f"video ended before frame {target_index} for {camera}"
                    )
                self._positions[camera] += 1
            if frame is None:
                raise RuntimeError(f"no frame decoded for {camera} at {target_index}")
            frames[camera] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frames

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()


class Hdf5CameraGeometry:
    """Read depth and calibration lazily so conversion does not duplicate full arrays."""

    def __init__(
        self,
        trajectory_path: Path,
        cameras: list[str],
        feature_names: set[str],
    ):
        self._handle = h5py.File(trajectory_path, "r")
        demo = _native_demo(self._handle)
        if demo is None:
            self._handle.close()
            raise ValueError(f"no native demo found in {trajectory_path}")
        paths = _camera_geometry_dataset_paths(cameras)
        self._datasets = {
            feature_name: demo[dataset_path]
            for feature_name, dataset_path in paths.items()
            if feature_name in feature_names
        }
        missing = feature_names - set(self._datasets)
        if missing:
            self._handle.close()
            raise KeyError(f"required camera geometry features are missing: {sorted(missing)}")

    def read(self, index: int) -> dict[str, np.ndarray]:
        return {
            feature_name: np.asarray(dataset[int(index)])
            for feature_name, dataset in self._datasets.items()
        }

    def close(self) -> None:
        self._handle.close()


def load_episode_arrays(
    episode_dir: Path,
    cameras: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    trajectory_path = episode_dir / "trajectory.h5"
    with h5py.File(trajectory_path, "r") as trajectory:
        demo = _native_demo(trajectory)
        if demo is None or str(trajectory["data"].attrs.get("schema_version", "")) != NATIVE_EPISODE_SCHEMA:
            raise ValueError(f"{trajectory_path} is not a native {NATIVE_EPISODE_SCHEMA} episode")
        actions = np.asarray(demo["actions"], dtype=np.float32)
        states = np.asarray(demo["obs/robot_state"], dtype=np.float32)
        count = int(actions.shape[0])
        if states.shape != (count, STATE_DIM):
            raise ValueError(f"expected state shape [{count},{STATE_DIM}], got {states.shape}")
        if actions.shape != (count, ACTION_DIM):
            raise ValueError(f"expected action shape [{count},{ACTION_DIM}], got {actions.shape}")
        terminated = np.asarray(demo["terminated"], dtype=bool).reshape(count, 1)
        truncated = np.asarray(demo["truncated"], dtype=bool).reshape(count, 1)
        done = np.asarray(demo["dones"], dtype=bool).reshape(count, 1)
        success = np.asarray(demo["success"], dtype=bool).reshape(count, 1)
        indices = np.arange(count, dtype=np.int64)
    selected_video_indices = {camera: indices for camera in cameras}
    return states, actions, selected_video_indices, {
        "terminated": terminated,
        "truncated": truncated,
        "done": done,
        "success": success,
    }


def add_episode(
    dataset: "LeRobotDataset",
    episode_dir: Path,
    cameras: list[str],
) -> dict[str, Any]:
    metadata = episode_metadata(episode_dir)
    task_metadata = metadata.get("metadata", {})
    task = episode_task_text(episode_dir, metadata)
    seed = int(task_metadata["seed"])
    task_id = str(task_metadata["task_id"])
    states, actions, video_indices, terminal = load_episode_arrays(episode_dir, cameras)
    trajectory_path = episode_dir / "trajectory.h5"
    videos = CameraFrames(
        {camera: episode_dir / "videos" / f"{camera}.mp4" for camera in cameras}
    )
    geometry_feature_names = {
        key for key in dataset.features if _is_camera_geometry_feature(key)
    }
    geometry = (
        Hdf5CameraGeometry(trajectory_path, cameras, geometry_feature_names)
        if geometry_feature_names
        else None
    )
    try:
        for index in range(states.shape[0]):
            images = videos.read(
                {camera: int(video_indices[camera][index]) for camera in cameras}
            )
            frame: dict[str, Any] = {
                "observation.state": states[index],
                "action": actions[index],
                "task": task,
                "next.terminated": terminal["terminated"][index],
                "next.truncated": terminal["truncated"][index],
                "next.done": terminal["done"][index],
                "next.success": terminal["success"][index],
            }
            for camera, image in images.items():
                frame[camera_feature_name(camera)] = image
            if geometry is not None:
                frame.update(geometry.read(index))
            dataset.add_frame(frame)
    finally:
        videos.close()
        if geometry is not None:
            geometry.close()
    dataset.save_episode()
    return {
        "seed": seed,
        "task_id": task_id,
        "task": task,
        "num_frames": int(states.shape[0]),
        "state_schema": STATE_SCHEMA,
        "action_schema": ACTION_SCHEMA,
    }


def validate_existing_schema(
    root: Path,
    *,
    episode_dir: Path,
    cameras: list[str],
    fps: int,
) -> None:
    info = json.loads((root / "meta" / "info.json").read_text())
    if int(info["fps"]) != int(fps):
        raise ValueError(f"FPS mismatch: existing={info['fps']} requested={fps}")
    expected = dataset_features(episode_dir, cameras)
    existing_user_features = set(info["features"]) - AUTO_FEATURE_KEYS
    if existing_user_features != set(expected):
        missing = sorted(set(expected) - existing_user_features)
        extra = sorted(existing_user_features - set(expected))
        raise ValueError(
            "LeRobot feature set mismatch: "
            f"missing_in_existing={missing}, extra_in_existing={extra}"
        )
    for key, feature in expected.items():
        existing = info["features"][key]
        if existing["dtype"] != feature["dtype"] or tuple(existing["shape"]) != tuple(
            feature["shape"]
        ):
            raise ValueError(
                f"feature mismatch for {key}: existing={existing} requested={feature}"
            )
        if key in {"observation.state", "action"} and existing.get("info") != feature.get("info"):
            raise ValueError(
                f"{key} metadata mismatch: "
                f"existing={existing.get('info')} requested={feature.get('info')}"
            )
        if _is_camera_geometry_feature(key) and existing.get("info") != feature.get(
            "info"
        ):
            raise ValueError(
                f"camera geometry metadata mismatch for {key}: "
                f"existing={existing.get('info')} requested={feature.get('info')}"
            )


def validate_episode_schema(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    episode_dir: Path,
) -> None:
    if set(reference) != set(candidate):
        raise ValueError(
            f"episode feature set mismatch in {episode_dir}: "
            f"reference={sorted(reference)}, candidate={sorted(candidate)}"
        )
    for key, expected in reference.items():
        actual = candidate[key]
        if actual["dtype"] != expected["dtype"] or tuple(actual["shape"]) != tuple(
            expected["shape"]
        ):
            raise ValueError(
                f"episode feature mismatch for {key} in {episode_dir}: "
                f"reference={expected}, candidate={actual}"
            )
        if key in {"observation.state", "action"} and actual.get("info") != expected.get("info"):
            raise ValueError(
                f"episode {key} metadata mismatch for {episode_dir}: "
                f"reference={expected.get('info')}, candidate={actual.get('info')}"
            )
        if _is_camera_geometry_feature(key) and actual.get("info") != expected.get(
            "info"
        ):
            raise ValueError(
                f"episode camera geometry metadata mismatch for {key} in "
                f"{episode_dir}: reference={expected.get('info')}, "
                f"candidate={actual.get('info')}"
            )


def write_episodes(
    *,
    root: Path,
    repo_id: str,
    episode_dirs: list[Path],
    cameras: list[str],
    fps: int,
    robot_type: str,
    registry: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from scripts.record.lerobot_seed_dataset import make_seed_record, write_seed_registry

    if not hasattr(LeRobotDataset, "resume"):
        raise RuntimeError(
            "This converter requires the LeRobot v3 writer API (0.5.2 or newer). "
            "LeRobot 0.3.3 in the robocasa environment is not compatible; run "
            "the converter in the dedicated lerobot environment described in README.md."
        )

    frame_count = 0
    for episode_dir in episode_dirs:
        episode_index = len(registry)
        if episode_index == 0:
            create_kwargs = dict(
                repo_id=repo_id,
                root=root,
                fps=int(fps),
                robot_type=robot_type,
                features=dataset_features(episode_dir, cameras),
                use_videos=True,
                image_writer_threads=4,
                metadata_buffer_size=1,
            )
            # These constructor options were added after LeRobot 0.5.2. The
            # metadata values below provide the same chunking behavior when
            # running against the README-pinned 0.5.2 commit.
            create_parameters = inspect.signature(LeRobotDataset.create).parameters
            if "data_files_size_in_mb" in create_parameters:
                create_kwargs["data_files_size_in_mb"] = 1
            if "video_files_size_in_mb" in create_parameters:
                create_kwargs["video_files_size_in_mb"] = 1
            dataset = LeRobotDataset.create(**create_kwargs)
        else:
            dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=root,
                image_writer_threads=4,
            )
        # Each episode is written through a fresh writer. Data and episode metadata
        # therefore advance to a new file; this threshold forces video to do the same.
        if isinstance(dataset.meta.info, dict):
            dataset.meta.info["video_files_size_in_mb"] = 1e-9
        else:
            dataset.meta.info.video_files_size_in_mb = 1e-9
        try:
            result = add_episode(dataset, episode_dir, cameras)
        finally:
            dataset.finalize()

        info_path = root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["data_files_size_in_mb"] = 1
        info["video_files_size_in_mb"] = 1
        temporary_info = info_path.with_name(info_path.name + ".tmp")
        temporary_info.write_text(json.dumps(info, indent=4) + "\n")
        temporary_info.replace(info_path)

        registry.append(
            make_seed_record(
                seed=int(result["seed"]),
                episode_index=episode_index,
                task_id=str(result["task_id"]),
                task=str(result["task"]),
                num_frames=int(result["num_frames"]),
            )
        )
        write_seed_registry(root, registry)
        frame_count += int(result["num_frames"])
    return registry, frame_count


def main() -> int:
    from scripts.record.lerobot_seed_dataset import (
        canonicalize_seed_order,
        clone_with_hardlinks,
        dataset_lock,
        install_staged_dataset,
        load_seed_registry,
        make_staging_root,
        validate_seed_dataset,
    )

    args = parse_args()
    episode_dirs = [path.resolve() for path in args.episode_dir]
    first_metadata = episode_metadata(episode_dirs[0])
    cameras = list(first_metadata.get("video", {}).get("cameras", []))
    if not cameras:
        raise ValueError("the first episode has no recorded cameras")
    requested_seeds = [
        int(episode_metadata(path).get("metadata", {})["seed"])
        for path in episode_dirs
    ]
    if len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError(f"duplicate seeds in input episodes: {requested_seeds}")
    for episode_dir in episode_dirs:
        metadata = episode_metadata(episode_dir)
        episode_cameras = list(metadata.get("video", {}).get("cameras", []))
        if episode_cameras != cameras:
            raise ValueError(
                f"camera mismatch in {episode_dir}: {episode_cameras} != {cameras}"
            )

    reference_features = dataset_features(episode_dirs[0], cameras)
    for episode_dir in episode_dirs[1:]:
        validate_episode_schema(
            reference_features,
            dataset_features(episode_dir, cameras),
            episode_dir=episode_dir,
        )

    output_root = args.output_root.resolve()
    with dataset_lock(output_root):
        if args.write_mode == "create":
            if output_root.exists() and not args.overwrite:
                raise FileExistsError(f"LeRobot output already exists: {output_root}")
            registry: list[dict[str, Any]] = []
        else:
            if not output_root.is_dir():
                raise FileNotFoundError(f"LeRobot dataset does not exist: {output_root}")
            validate_seed_dataset(output_root)
            registry = load_seed_registry(output_root)
            existing_seeds = {int(item["seed"]) for item in registry}
            duplicates = sorted(existing_seeds.intersection(requested_seeds))
            if duplicates:
                raise ValueError(f"seeds already exist in dataset: {duplicates}")
            validate_existing_schema(
                output_root,
                episode_dir=episode_dirs[0],
                cameras=cameras,
                fps=int(args.fps),
            )

        container, staged_root = make_staging_root(output_root)
        try:
            if args.write_mode == "append":
                clone_with_hardlinks(output_root, staged_root)
            registry, frame_count = write_episodes(
                root=staged_root,
                repo_id=args.repo_id,
                episode_dirs=episode_dirs,
                cameras=cameras,
                fps=int(args.fps),
                robot_type=args.robot_type,
                registry=registry,
            )
            canonicalize_seed_order(staged_root, registry)
            summary = validate_seed_dataset(staged_root)
            install_staged_dataset(
                staged_root,
                output_root,
                keep_backup=False,
            )
            summary["root"] = str(output_root)
        finally:
            shutil.rmtree(container, ignore_errors=True)

    print(
        json.dumps(
            {**summary, "repo_id": args.repo_id, "added_frames": frame_count},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
