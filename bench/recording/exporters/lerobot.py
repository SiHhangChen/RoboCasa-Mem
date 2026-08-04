"""Boundary and validation helpers for LeRobot dataset exports."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class LeRobotExporter(Protocol):
    def export(self, native_episode: Path, output_dir: Path) -> Path: ...


def validate_append_schema(
    root: Path,
    *,
    fps: int,
    cameras: Sequence[str] = (),
    image_size: tuple[int, int],
) -> None:
    """Reject an incompatible append before recording a native episode.

    The exporter owns this check because the native recorder must not know the
    layout of a LeRobot dataset. ``image_size`` is ``(height, width)``.
    """

    info_path = Path(root) / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if int(info["fps"]) != int(fps):
        raise ValueError(
            f"LeRobot append FPS mismatch: dataset={info['fps']} requested={fps}"
        )

    requested_cameras = [str(value) for value in cameras]
    if not requested_cameras:
        requested_cameras = [
            key.removeprefix("observation.images.")
            for key, feature in info["features"].items()
            if feature["dtype"] == "video"
        ]
    expected_shape = [int(image_size[0]), int(image_size[1]), 3]
    for camera in requested_cameras:
        feature_key = f"observation.images.{camera}"
        feature = info["features"].get(feature_key)
        if feature is None:
            raise ValueError(
                f"LeRobot append camera mismatch: {feature_key} is not in {info_path}"
            )
        existing_shape = [int(value) for value in feature["shape"]]
        if existing_shape != expected_shape:
            raise ValueError(
                "LeRobot append image shape mismatch before recording: "
                f"dataset {feature_key}={existing_shape}, requested={expected_shape}. "
                f"Use --camera-width {existing_shape[1]} "
                f"--camera-height {existing_shape[0]}."
            )


__all__ = ["LeRobotExporter", "validate_append_schema"]
