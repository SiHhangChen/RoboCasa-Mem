"""Convert native MemBench episodes to LeRobot 0.6.1 with depth videos.

This entry point intentionally lives beside, rather than replacing, the legacy
converter.  Source-side validation and frame reading are shared with the
legacy entry point, while all LeRobot writer configuration in this file is
strictly pinned to 0.6.1's depth-video implementation.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEROBOT_SRC = ROOT / "policy" / "lerobot-0.6.1" / "src"
LEGACY_LEROBOT_SRC = ROOT / "policy" / "lerobot" / "src"
EXPECTED_LEROBOT_VERSION = Version("0.6.1")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Put the requested 0.6.1 tree first.  Appending the legacy tree prevents the
# imported source-side helper module from moving that tree to the front.
_initial_lerobot_src = Path(
    os.environ.get("MEMBENCH_LEROBOT_SRC", str(DEFAULT_LEROBOT_SRC))
).expanduser()
if _initial_lerobot_src.is_dir() and str(_initial_lerobot_src.resolve()) not in sys.path:
    sys.path.insert(0, str(_initial_lerobot_src.resolve()))
if LEGACY_LEROBOT_SRC.is_dir() and str(LEGACY_LEROBOT_SRC.resolve()) not in sys.path:
    sys.path.append(str(LEGACY_LEROBOT_SRC.resolve()))

from scripts.record import convert_bench_episodes_to_lerobot as native_converter  # noqa: E402


DEPTH_FEATURE_PREFIX = "observation.depth."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert native MemBench episodes to a LeRobot v3 dataset using "
            "LeRobot 0.6.1 depth-video compression."
        )
    )
    parser.add_argument("--episode-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--robot-type", default="mobile_manipulator")
    parser.add_argument("--write-mode", choices=["create", "append"], default="create")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Convert recorded depth maps to depth-video features. "
            "Disabled by default; pass --depth to enable."
        ),
    )
    parser.add_argument(
        "--lerobot-src",
        type=Path,
        default=_initial_lerobot_src,
        help=(
            "Path to the LeRobot 0.6.1 src directory. Defaults to "
            "MEMBENCH_LEROBOT_SRC or policy/lerobot-0.6.1/src."
        ),
    )
    parser.add_argument("--depth-min-m", type=float, default=0.01)
    parser.add_argument("--depth-max-m", type=float, default=10.0)
    parser.add_argument("--depth-shift-m", type=float, default=3.5)
    parser.add_argument(
        "--depth-use-log",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--depth-gop", type=int, default=16)
    parser.add_argument("--encoder-threads", type=int, default=4)
    parser.add_argument("--image-writer-threads", type=int, default=4)
    return parser.parse_args()


def _activate_lerobot_source(path: Path) -> Path:
    source = path.expanduser().resolve()
    if not (source / "lerobot" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"LeRobot source directory must contain lerobot/__init__.py: {source}"
        )
    source_text = str(source)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    return source


def require_lerobot_061(source: Path) -> str:
    try:
        installed = Version(version("lerobot"))
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "LeRobot is not installed in the active environment. Install "
            "policy/lerobot-0.6.1 with the dataset extra first."
        ) from exc
    if installed != EXPECTED_LEROBOT_VERSION:
        raise RuntimeError(
            f"This converter requires LeRobot 0.6.1 exactly, got {installed}. "
            "Activate the lerobot061 environment."
        )

    import lerobot

    imported_file = Path(lerobot.__file__).resolve()
    expected_package = (source / "lerobot").resolve()
    if expected_package not in imported_file.parents:
        raise RuntimeError(
            "LeRobot was imported from an unexpected source tree: "
            f"imported={imported_file}, requested={expected_package}"
        )
    return str(installed)


def _depth_video_feature(feature_name: str, feature: dict[str, Any]) -> dict[str, Any]:
    if feature.get("dtype") != "uint16":
        raise ValueError(
            f"native depth feature must be uint16 before video encoding: "
            f"{feature_name}={feature}"
        )
    shape = tuple(int(value) for value in feature.get("shape", ()))
    if len(shape) != 3 or shape[-1] != 1:
        raise ValueError(
            f"native depth feature must have shape [height,width,1]: "
            f"{feature_name}={shape}"
        )

    info = dict(feature.get("info", {}))
    scale = float(info.get("scale_meters_per_unit", 0.001))
    unit = str(info.get("unit", "millimeter")).strip().lower()
    if unit not in {"mm", "millimeter", "millimetre"} or abs(scale - 0.001) > 1e-12:
        raise ValueError(
            "LeRobot 0.6.1 integer depth input is interpreted as millimeters; "
            f"got unit={unit!r}, scale_meters_per_unit={scale} for {feature_name}"
        )
    info["is_depth_map"] = True
    info["depth_unit"] = "mm"
    return {
        **feature,
        "dtype": "video",
        "shape": shape,
        "names": ["height", "width", "channels"],
        "info": info,
    }


def dataset_features(
    episode_dir: Path,
    cameras: list[str],
    *,
    include_depth: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build a LeRobot 0.6.1 feature schema from one native episode."""

    features = native_converter.dataset_features(
        episode_dir, cameras, include_depth=include_depth
    )
    if not include_depth:
        return features
    depth_keys = [key for key in features if key.startswith(DEPTH_FEATURE_PREFIX)]
    if not depth_keys:
        raise ValueError(
            f"native episode has no recorded depth features: {episode_dir}"
        )
    for key in depth_keys:
        features[key] = _depth_video_feature(key, features[key])
    return features


def _expected_info_is_subset(
    existing: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    return all(existing.get(key) == value for key, value in expected.items())


def validate_existing_schema(
    root: Path,
    *,
    episode_dir: Path,
    cameras: list[str],
    fps: int,
    depth_encoder: Any,
    include_depth: bool = False,
) -> None:
    from lerobot.configs import DepthEncoderConfig

    info = json.loads((root / "meta" / "info.json").read_text())
    if int(info["fps"]) != int(fps):
        raise ValueError(f"FPS mismatch: existing={info['fps']} requested={fps}")

    expected = dataset_features(episode_dir, cameras, include_depth=include_depth)
    existing_user_features = set(info["features"]) - native_converter.AUTO_FEATURE_KEYS
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
        if key in {"observation.state", "action"} and existing.get("info") != feature.get(
            "info"
        ):
            raise ValueError(f"{key} metadata mismatch")
        if native_converter._is_camera_geometry_feature(key):
            if not _expected_info_is_subset(
                dict(existing.get("info", {})),
                dict(feature.get("info", {})),
            ):
                raise ValueError(f"camera geometry metadata mismatch for {key}")
        if key.startswith(DEPTH_FEATURE_PREFIX):
            persisted_encoder = DepthEncoderConfig.from_video_info(existing.get("info"))
            if persisted_encoder != depth_encoder:
                raise ValueError(
                    f"depth encoder mismatch for {key}: "
                    f"existing={persisted_encoder}, requested={depth_encoder}"
                )


def make_depth_encoder(args: argparse.Namespace) -> Any:
    from lerobot.configs import DepthEncoderConfig

    if not args.depth_min_m < args.depth_max_m:
        raise ValueError(
            f"depth range must satisfy min < max, got "
            f"{args.depth_min_m} >= {args.depth_max_m}"
        )
    if args.depth_use_log and args.depth_min_m + args.depth_shift_m <= 0:
        raise ValueError(
            "log depth encoding requires depth_min_m + depth_shift_m > 0"
        )
    if args.depth_gop <= 0:
        raise ValueError(f"depth GOP must be positive, got {args.depth_gop}")
    if args.encoder_threads <= 0:
        raise ValueError(
            f"encoder threads must be positive, got {args.encoder_threads}"
        )
    if args.image_writer_threads <= 0:
        raise ValueError(
            f"image writer threads must be positive, got {args.image_writer_threads}"
        )

    return DepthEncoderConfig(
        vcodec="hevc",
        pix_fmt="gray12le",
        g=int(args.depth_gop),
        crf=30,
        extra_options={"x265-params": "lossless=1"},
        depth_min=float(args.depth_min_m),
        depth_max=float(args.depth_max_m),
        shift=float(args.depth_shift_m),
        use_log=bool(args.depth_use_log),
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
    depth_encoder: Any,
    encoder_threads: int,
    image_writer_threads: int,
    include_depth: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from scripts.record.lerobot_seed_dataset import make_seed_record, write_seed_registry

    create_parameters = inspect.signature(LeRobotDataset.create).parameters
    required_parameters = {"depth_encoder", "encoder_threads"}
    missing_parameters = sorted(required_parameters - set(create_parameters))
    if missing_parameters:
        raise RuntimeError(
            "The active LeRobot writer does not expose the 0.6.1 depth API: "
            f"missing create parameters {missing_parameters}"
        )

    frame_count = 0
    for episode_dir in episode_dirs:
        episode_index = len(registry)
        if episode_index == 0:
            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=root,
                fps=int(fps),
                robot_type=robot_type,
                features=dataset_features(episode_dir, cameras, include_depth=include_depth),
                use_videos=True,
                image_writer_threads=int(image_writer_threads),
                metadata_buffer_size=1,
                data_files_size_in_mb=1,
                video_files_size_in_mb=1,
                depth_encoder=depth_encoder,
                encoder_threads=int(encoder_threads),
                streaming_encoding=True,
            )
        else:
            dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=root,
                image_writer_threads=int(image_writer_threads),
                depth_encoder=depth_encoder,
                encoder_threads=int(encoder_threads),
                streaming_encoding=True,
            )
            # LeRobot 0.6.1 aggregates the persisted global stats while saving
            # an appended episode.  MemBench deliberately rewrites those stats
            # from all canonical episode rows after the batch is complete, and
            # the intermediate aggregate is both unnecessary and incompatible
            # with stats written by the seed canonicalizer.  Clearing it here
            # avoids that transient shape mismatch; the staged transaction is
            # never installed until canonicalize_seed_order has rebuilt it.
            dataset.meta.stats = None

        # The MemBench seed registry requires one data, metadata, and video file
        # per episode so files can be reordered without decoding or re-encoding.
        if isinstance(dataset.meta.info, dict):
            dataset.meta.info["video_files_size_in_mb"] = 1e-9
        else:
            dataset.meta.info.video_files_size_in_mb = 1e-9
        try:
            result = native_converter.add_episode(dataset, episode_dir, cameras)
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
    args = parse_args()
    lerobot_source = _activate_lerobot_source(args.lerobot_src)
    lerobot_version = require_lerobot_061(lerobot_source)
    depth_encoder = make_depth_encoder(args) if args.depth else None

    from scripts.record.lerobot_seed_dataset import (
        canonicalize_seed_order,
        clone_with_hardlinks,
        dataset_lock,
        install_staged_dataset,
        load_seed_registry,
        make_staging_root,
        validate_seed_dataset,
    )

    episode_dirs = [path.resolve() for path in args.episode_dir]
    first_metadata = native_converter.episode_metadata(episode_dirs[0])
    cameras = list(first_metadata.get("video", {}).get("cameras", []))
    if not cameras:
        raise ValueError("the first episode has no recorded cameras")

    requested_seeds = [
        int(native_converter.episode_metadata(path).get("metadata", {})["seed"])
        for path in episode_dirs
    ]
    if len(requested_seeds) != len(set(requested_seeds)):
        raise ValueError(f"duplicate seeds in input episodes: {requested_seeds}")

    for episode_dir in episode_dirs:
        metadata = native_converter.episode_metadata(episode_dir)
        episode_cameras = list(metadata.get("video", {}).get("cameras", []))
        if episode_cameras != cameras:
            raise ValueError(
                f"camera mismatch in {episode_dir}: {episode_cameras} != {cameras}"
            )

    reference_features = dataset_features(episode_dirs[0], cameras, include_depth=args.depth)
    for episode_dir in episode_dirs[1:]:
        native_converter.validate_episode_schema(
            reference_features,
            dataset_features(episode_dir, cameras, include_depth=args.depth),
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
                depth_encoder=depth_encoder,
                include_depth=args.depth,
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
                depth_encoder=depth_encoder,
                encoder_threads=int(args.encoder_threads),
                image_writer_threads=int(args.image_writer_threads),
                include_depth=args.depth,
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
            {
                **summary,
                "repo_id": args.repo_id,
                "added_frames": frame_count,
                "lerobot_version": lerobot_version,
                "lerobot_source": str(lerobot_source),
                "depth_encoder": (
                    dataclasses.asdict(depth_encoder) if depth_encoder is not None else None
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
