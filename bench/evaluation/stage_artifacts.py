"""Atomic artifacts for stage-conditioned evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import shutil
from statistics import median
import tempfile
from typing import Any, Mapping
import uuid

import numpy as np

from bench.evaluation.stage_rollout import (
    FAILURE,
    INVALID_PREFIX,
    SUCCESS,
    StageEvaluationResult,
    wilson_interval,
)


class StageArtifactWriter:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, payload: Mapping[str, Any]) -> None:
        path = self.run_dir / "run_manifest.json"
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if _manifest_identity(existing) != _manifest_identity(payload):
                raise ValueError(f"stage run manifest mismatch for {self.run_dir}")
            return
        _atomic_json(path, payload)

    def completed(self, target_stage: str, seed: int) -> bool:
        return self._result_path(target_stage, seed).is_file()

    def start_episode(
        self,
        *,
        target_stage: str,
        seed: int,
        task: Any,
        record_video: bool,
        video_cameras: tuple[str, ...],
        video_fps: int,
    ) -> "StageEpisodeArtifactSession":
        return StageEpisodeArtifactSession(
            writer=self,
            target_stage=target_stage,
            seed=seed,
            task=task,
            record_video=record_video,
            video_cameras=video_cameras,
            video_fps=video_fps,
        )

    def write_result(
        self,
        result: StageEvaluationResult,
        *,
        source: str,
        request_metrics: list[Mapping[str, Any]] | None = None,
        expert_prefix_manifest: Mapping[str, Any] | None = None,
        evaluator_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Commit a stage result that does not need a live video session."""

        session = self.start_episode(
            target_stage=result.target_stage,
            seed=result.seed,
            task=None,
            record_video=False,
            video_cameras=(),
            video_fps=1,
        )
        try:
            session.finish(
                result,
                source=source,
                request_metrics=list(request_metrics or ()),
                expert_prefix_manifest=expert_prefix_manifest,
                evaluator_metadata=evaluator_metadata,
            )
        except BaseException:
            session.abort()
            raise

    def write_summary(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        expected_stages = tuple(str(value) for value in manifest.get("targets", ()))
        expected_seeds = tuple(int(value) for value in manifest.get("seeds", ()))
        target_sources = {
            str(key): str(value)
            for key, value in dict(manifest.get("target_sources") or {}).items()
        }
        stage_payloads: dict[str, Any] = {}
        stage_ids = set(expected_stages)
        stages_root = self.run_dir / "stages"
        if stages_root.is_dir():
            stage_ids.update(path.name for path in stages_root.iterdir() if path.is_dir())
        invalid_artifacts: dict[str, str] = {}
        for stage_id in sorted(stage_ids):
            stage_dir = stages_root / stage_id
            results = []
            for path in sorted(stage_dir.glob("episodes/seed_*/result.json")):
                try:
                    results.append(_load_stage_result(path, target_stage=stage_id))
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    invalid_artifacts[str(path.relative_to(self.run_dir))] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            successes = sum(item.get("status") == SUCCESS for item in results)
            failures = sum(item.get("status") == FAILURE for item in results)
            invalid = sum(item.get("status") == INVALID_PREFIX for item in results)
            valid = successes + failures
            interval = wilson_interval(successes, valid)
            seeds = [int(item["seed"]) for item in results if "seed" in item]
            expected_seed_set = set(expected_seeds)
            recorded_seed_set = set(seeds)
            duplicate_seeds = sorted(
                seed for seed in recorded_seed_set if seeds.count(seed) > 1
            )
            unexpected_seeds = sorted(recorded_seed_set - expected_seed_set)
            pending_seeds = sorted(expected_seed_set - recorded_seed_set)
            model_steps = [
                int(item["model_control_steps"])
                for item in results
                if item.get("status") in {SUCCESS, FAILURE}
            ]
            source_values = {
                str(item.get("source")) for item in results if item.get("source")
            }
            source = target_sources.get(stage_id)
            if source is None and len(source_values) == 1:
                source = next(iter(source_values))
            planned = len(expected_seeds) if expected_seeds else len(recorded_seed_set)
            recorded = len(recorded_seed_set)
            complete = (
                planned > 0
                and not pending_seeds
                and not unexpected_seeds
                and not duplicate_seeds
                and recorded == planned
            )
            stage_payloads[stage_id] = {
                "source": source,
                "planned": planned,
                "recorded": recorded,
                "pending": len(pending_seeds),
                "pending_seeds": pending_seeds,
                "unexpected_seeds": unexpected_seeds,
                "duplicate_seeds": duplicate_seeds,
                "complete": complete,
                "attempted": len(results),
                "successes": successes,
                "failures": failures,
                "invalid_prefixes": invalid,
                "valid_tests": valid,
                "stage_sr": None if valid == 0 else successes / valid,
                "prefix_validity": (
                    None
                    if source == "main_rollout" or not results
                    else valid / len(results)
                ),
                "confidence_interval": None if interval is None else list(interval),
                "model_control_steps_mean": (
                    None if not model_steps else sum(model_steps) / len(model_steps)
                ),
                "model_control_steps_median": (
                    None if not model_steps else float(median(model_steps))
                ),
            }
        stage_rates = [
            payload["stage_sr"]
            for stage_id, payload in stage_payloads.items()
            if stage_id in expected_stages and payload["stage_sr"] is not None
        ]
        complete = bool(expected_stages) and all(
            stage_payloads.get(stage_id, {}).get("complete", False)
            for stage_id in expected_stages
        )
        summary = {
            "schema_version": "membench.stage_run_summary/v1",
            "task_id": manifest.get("task_id"),
            "stage_order": list(expected_stages),
            "stages": stage_payloads,
            "macro_stage_sr": (
                None
                if len(stage_rates) != len(expected_stages)
                else sum(stage_rates) / len(stage_rates)
            ),
            "complete": complete,
            "publishable": complete and not invalid_artifacts,
            "invalid_episode_artifacts": invalid_artifacts,
        }
        _atomic_json(self.run_dir / "summary.json", summary)
        return summary

    def _load_manifest(self) -> dict[str, Any]:
        path = self.run_dir / "run_manifest.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"stage run manifest must be a JSON object: {path}")
        return payload

    def _result_path(self, target_stage: str, seed: int) -> Path:
        return (
            self.run_dir
            / "stages"
            / str(target_stage)
            / "episodes"
            / f"seed_{int(seed):06d}"
            / "result.json"
        )


class StageEpisodeArtifactSession:
    def __init__(
        self,
        *,
        writer: StageArtifactWriter,
        target_stage: str,
        seed: int,
        task: Any,
        record_video: bool,
        video_cameras: tuple[str, ...],
        video_fps: int,
    ) -> None:
        self.task = task
        self.target_stage = str(target_stage)
        self.seed = int(seed)
        episodes = writer.run_dir / "stages" / self.target_stage / "episodes"
        episodes.mkdir(parents=True, exist_ok=True)
        self.final_dir = episodes / f"seed_{self.seed:06d}"
        self.temp_dir = Path(
            tempfile.mkdtemp(prefix=f".seed_{self.seed:06d}.", dir=episodes)
        )
        self.record_video = bool(record_video)
        self.video_cameras = tuple(video_cameras)
        self.video_fps = int(video_fps)
        self.actions: list[np.ndarray] = []
        self._steps = (self.temp_dir / "steps.jsonl").open("w", encoding="utf-8")
        self._writers: dict[str, Any] = {}
        self._frame_counts = {camera: 0 for camera in self.video_cameras}

    def record(self, payload: Mapping[str, Any]) -> None:
        self._steps.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
        self._steps.flush()
        action = np.asarray(payload["executed_action"], dtype=np.float32)
        if action.shape != (13,) or not np.all(np.isfinite(action)):
            raise ValueError(
                f"stage executed action must be finite float32[13], got {action.shape}"
            )
        self.actions.append(action.copy())
        if self.record_video:
            self._append_frames(self.task.artifact_frames(self.video_cameras))

    def finish(
        self,
        result: StageEvaluationResult,
        *,
        source: str,
        request_metrics: list[Mapping[str, Any]],
        expert_prefix_manifest: Mapping[str, Any] | None = None,
        evaluator_metadata: Mapping[str, Any] | None = None,
    ) -> None:
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
        with (self.temp_dir / "request_metrics.jsonl").open("w", encoding="utf-8") as handle:
            for metric in request_metrics:
                handle.write(json.dumps(_jsonable(metric), sort_keys=True) + "\n")
        payload = result.to_dict()
        payload.pop("logs", None)
        payload.update(
            {
                "schema_version": "membench.stage_episode_result/v1",
                "source": str(source),
                "logs_file": "steps.jsonl",
                "expert_prefix": _jsonable(expert_prefix_manifest),
                "evaluator_metadata": _jsonable(evaluator_metadata),
                "artifacts": {
                    "action_count": int(actions.shape[0]),
                    "video_frame_counts": dict(self._frame_counts),
                },
            }
        )
        _atomic_json(self.temp_dir / "result.json", payload)
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

    def _append_frames(self, images: Mapping[str, np.ndarray]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required when stage record_video=true") from exc
        videos = self.temp_dir / "videos"
        videos.mkdir(exist_ok=True)
        for camera in self.video_cameras:
            image = np.asarray(images[camera])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"stage video frame {camera} must be uint8 HWC RGB")
            writer = self._writers.get(camera)
            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(videos / f"{camera}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.video_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not open stage video writer for {camera}")
                self._writers[camera] = writer
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self._frame_counts[camera] += 1

    def _close_streams(self) -> None:
        if not self._steps.closed:
            self._steps.flush()
            os.fsync(self._steps.fileno())
            self._steps.close()
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()


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


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _manifest_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "run_id": payload.get("run_id"),
        "stage_suite": payload.get("stage_suite"),
        "base_suite": payload.get("base_suite"),
        "task_id": payload.get("task_id"),
        "targets": payload.get("targets"),
        "target_sources": payload.get("target_sources"),
        "seeds": payload.get("seeds"),
        "expert_prefix_root": payload.get("expert_prefix_root"),
        "main_rollout_run": payload.get("main_rollout_run"),
        "policy_contract": payload.get("policy_contract"),
    }


def _load_stage_result(path: Path, *, target_stage: str) -> dict[str, Any]:
    for required_name in ("actions.npz", "steps.jsonl", "request_metrics.jsonl"):
        if not (path.parent / required_name).is_file():
            raise ValueError(f"missing required stage artifact: {required_name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stage result must be a JSON object")
    if payload.get("schema_version") != "membench.stage_episode_result/v1":
        raise ValueError(f"unsupported stage result schema: {payload.get('schema_version')!r}")
    if str(payload.get("target_stage")) != str(target_stage):
        raise ValueError("stage result target does not match its directory")
    directory = path.parent.name
    if not directory.startswith("seed_"):
        raise ValueError(f"invalid stage episode directory: {directory}")
    try:
        directory_seed = int(directory.removeprefix("seed_"))
    except ValueError as exc:
        raise ValueError(f"invalid stage episode directory: {directory}") from exc
    if payload.get("seed") != directory_seed:
        raise ValueError("stage result seed does not match its directory")
    status = payload.get("status")
    if status not in {SUCCESS, FAILURE, INVALID_PREFIX}:
        raise ValueError(f"unsupported stage result status: {status!r}")
    if bool(payload.get("success")) != (status == SUCCESS):
        raise ValueError("stage result status and success disagree")
    for field in (
        "expert_prefix_steps",
        "model_control_steps",
        "total_model_inference_calls",
        "environment_steps",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"stage result {field} must be a non-negative integer")
    return payload


__all__ = ["StageArtifactWriter", "StageEpisodeArtifactSession"]
