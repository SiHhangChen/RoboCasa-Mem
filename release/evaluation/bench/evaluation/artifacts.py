"""Atomic, resumable evaluation artifact writing."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping
import uuid

import numpy as np

from bench.evaluation.summary import aggregate_episode_results
from bench.evaluation.schemas import validate_episode_result


class ArtifactWriter:
    def __init__(
        self,
        run_dir: Path,
        *,
        expected_seeds: Iterable[int] | None = None,
    ) -> None:
        self.run_dir = run_dir.expanduser().resolve()
        self.episodes_dir = self.run_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.expected_seeds = (
            None if expected_seeds is None else tuple(int(seed) for seed in expected_seeds)
        )
        if self.expected_seeds is not None and len(set(self.expected_seeds)) != len(
            self.expected_seeds
        ):
            raise ValueError("expected_seeds must be unique")

    def write_run_manifest(self, manifest: Mapping[str, Any]) -> None:
        path = self.run_dir / "run_manifest.json"
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if _manifest_identity(existing) != _manifest_identity(manifest):
                raise ValueError(
                    f"Run manifest identity mismatch for resume directory {self.run_dir}"
                )
            return
        _atomic_json(path, manifest)

    def start_episode(
        self,
        *,
        seed: int,
        record_video: bool,
        video_cameras: tuple[str, ...],
        video_fps: int,
    ) -> "EpisodeArtifactSession":
        return EpisodeArtifactSession(
            writer=self,
            seed=seed,
            record_video=record_video,
            video_cameras=video_cameras,
            video_fps=video_fps,
        )

    def completed_seeds(self, *, retry_errors: bool = False) -> set[int]:
        completed: set[int] = set()
        for path in self.episodes_dir.glob("seed_*/result.json"):
            try:
                seed = _seed_from_result_path(path)
                result = self._load_valid_result(path, expected_seed=seed)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if retry_errors and result["status"] == "error":
                continue
            completed.add(seed)
        return completed

    def load_episode_results(self) -> list[dict[str, Any]]:
        results = []
        for path in sorted(self.episodes_dir.glob("seed_*/result.json")):
            try:
                seed = _seed_from_result_path(path)
                payload = self._load_valid_result(path, expected_seed=seed)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            results.append(payload)
        return results

    def invalid_episode_artifacts(self) -> dict[str, str]:
        """Return invalid committed episode paths and their validation errors."""

        invalid: dict[str, str] = {}
        for path in sorted(self.episodes_dir.glob("seed_*/result.json")):
            try:
                seed = _seed_from_result_path(path)
                self._load_valid_result(path, expected_seed=seed)
            except Exception as exc:
                invalid[str(path.relative_to(self.run_dir))] = (
                    f"{type(exc).__name__}: {exc}"
                )
        return invalid

    def write_summary(self) -> dict[str, Any]:
        summary = aggregate_episode_results(
            self.load_episode_results(),
            expected_seeds=self.expected_seeds,
        )
        summary["invalid_episode_artifacts"] = self.invalid_episode_artifacts()
        if summary["invalid_episode_artifacts"]:
            summary["publishable"] = False
        _atomic_json(self.run_dir / "summary.json", summary)
        return summary

    @staticmethod
    def _load_valid_result(path: Path, *, expected_seed: int) -> dict[str, Any]:
        for required_name in ("actions.npz", "request_metrics.jsonl"):
            required_path = path.parent / required_name
            if not required_path.is_file():
                raise ValueError(f"missing required artifact: {required_name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return validate_episode_result(payload, expected_seed=expected_seed)


class EpisodeArtifactSession:
    def __init__(
        self,
        *,
        writer: ArtifactWriter,
        seed: int,
        record_video: bool,
        video_cameras: tuple[str, ...],
        video_fps: int,
    ) -> None:
        self.writer = writer
        self.seed = int(seed)
        self.final_dir = writer.episodes_dir / f"seed_{seed:06d}"
        self.temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".seed_{seed:06d}.",
                dir=writer.episodes_dir,
            )
        )
        self.record_video = record_video
        self.video_cameras = video_cameras
        self.video_fps = int(video_fps)
        self.actions: list[np.ndarray] = []
        self._metrics = (self.temp_dir / "request_metrics.jsonl").open(
            "w",
            encoding="utf-8",
        )
        self._video_writers: dict[str, Any] = {}
        self._video_frame_counts = {camera: 0 for camera in video_cameras}

    def append_action(self, action: np.ndarray) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (13,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Recorded action must be finite float32[13], got {value.shape}")
        self.actions.append(value.copy())

    def append_request_metric(self, metric: Mapping[str, Any]) -> None:
        self._metrics.write(json.dumps(_jsonable(metric), sort_keys=True) + "\n")
        self._metrics.flush()

    def append_frames(self, images: Mapping[str, np.ndarray]) -> None:
        if not self.record_video:
            return
        missing = [camera for camera in self.video_cameras if camera not in images]
        if missing:
            raise KeyError(f"Missing requested artifact video cameras: {missing}")
        for camera in self.video_cameras:
            self._append_frame(camera, np.asarray(images[camera]))
            self._video_frame_counts[camera] += 1

    def finish(self, result: Mapping[str, Any] | Any) -> None:
        self._close_streams()
        actions = (
            np.stack(self.actions).astype(np.float32)
            if self.actions
            else np.empty((0, 13), dtype=np.float32)
        )
        with (self.temp_dir / "actions.npz").open("wb") as handle:
            np.savez_compressed(handle, actions=actions)
            handle.flush()
            os.fsync(handle.fileno())
        result_payload = validate_episode_result(
            _result_mapping(result),
            expected_seed=self.seed,
        )
        result_payload["artifacts"] = {
            "action_count": int(actions.shape[0]),
            "video_frame_counts": dict(self._video_frame_counts),
        }
        _atomic_json(self.temp_dir / "result.json", result_payload)
        backup_dir: Path | None = None
        if self.final_dir.exists():
            backup_dir = self.final_dir.with_name(
                f".{self.final_dir.name}.backup.{uuid.uuid4().hex}"
            )
            os.replace(self.final_dir, backup_dir)
        try:
            os.replace(self.temp_dir, self.final_dir)
        except Exception:
            if backup_dir is not None and not self.final_dir.exists():
                os.replace(backup_dir, self.final_dir)
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)

    def abort(self) -> None:
        self._close_streams()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _append_frame(self, camera: str, image: np.ndarray) -> None:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Video frame {camera} must be uint8 HWC RGB")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required when record_video=true") from exc
        writer = self._video_writers.get(camera)
        if writer is None:
            videos_dir = self.temp_dir / "videos"
            videos_dir.mkdir(exist_ok=True)
            path = videos_dir / f"{camera}.mp4"
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {path}")
            self._video_writers[camera] = writer
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def _close_streams(self) -> None:
        if not self._metrics.closed:
            self._metrics.flush()
            os.fsync(self._metrics.fileno())
            self._metrics.close()
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _result_mapping(result: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, Mapping):
        return result
    raise TypeError("Episode result must be a dataclass or mapping")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    suite = manifest.get("suite", {})
    return {
        "run_id": manifest.get("run_id"),
        "suite": {
            "hash": suite.get("hash"),
            "version": suite.get("version"),
        }
        if isinstance(suite, Mapping)
        else None,
        "task": manifest.get("task"),
        "seeds": manifest.get("seeds"),
        "policy_contract": manifest.get("policy_contract"),
        "rollout": manifest.get("rollout"),
        "artifacts": manifest.get("artifacts"),
    }


def _seed_from_result_path(path: Path) -> int:
    directory = path.parent.name
    if not directory.startswith("seed_"):
        raise ValueError(f"invalid episode directory name: {directory}")
    try:
        return int(directory.removeprefix("seed_"))
    except ValueError as exc:
        raise ValueError(f"invalid episode directory name: {directory}") from exc
