"""Configuration for native MobileMemBench v2 recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RecordingConfig:
    output_dir: Path
    cameras: list[str] = field(default_factory=list)
    camera_width: int = 224
    camera_height: int = 224
    fps: int = 20
    record_depth: bool = True
    depth_dtype: str = "uint16"
    depth_scale: float = 0.001
    save_navigation_snapshots: bool = False
    env_meta: dict | None = None
    env_name: str | None = None

    def with_output_dir(self, path: Path) -> "RecordingConfig":
        return RecordingConfig(
            output_dir=Path(path),
            cameras=list(self.cameras),
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            fps=self.fps,
            record_depth=self.record_depth,
            depth_dtype=self.depth_dtype,
            depth_scale=self.depth_scale,
            save_navigation_snapshots=self.save_navigation_snapshots,
            env_meta=self.env_meta,
            env_name=self.env_name,
        )


__all__ = ["RecordingConfig"]
