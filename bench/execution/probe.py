"""Task-agnostic helpers for one-off probes over a canonical EpisodeRuntime."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bench.execution.contracts import ExpertPlan, NavigateStage, OperateStage
from bench.execution.navigation.base_motion import (
    qpos_for_controller,
    set_base_pose_kinematic,
    step_base_and_hold_arm,
)
from bench.execution.navigation.contracts import Pose, StationSpec, StationTarget
from bench.execution.navigation.station_poses import (
    quat_xyzw_to_yaw,
    station_sample_pose,
    station_sample_torso_height,
)
from bench.execution.planning import matrix_from_json_pose, pose_from_matrix
from bench.tasks import TaskCatalog


Stage = NavigateStage | OperateStage


@dataclass(frozen=True)
class ProbeStationPose:
    """One resolved mobile-base + torso pose used by a probe session."""

    station_id: str
    xy: tuple[float, float]
    yaw: float
    source: str
    sample: Mapping[str, Any]
    torso_height: float | None = None

    def as_pose(self) -> Pose:
        half = 0.5 * float(self.yaw)
        return Pose(
            position=(float(self.xy[0]), float(self.xy[1]), 0.0),
            quaternion=(0.0, 0.0, float(np.sin(half)), float(np.cos(half))),
            torso_height=self.torso_height,
        )


def _normalized_task_id(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def resolve_registered_task_id(requested: str) -> str:
    """Resolve a user-supplied task spelling through the task catalog."""

    return TaskCatalog.canonical_id(requested)


def stage_label(stage: Stage, index: int | None = None) -> str:
    name = getattr(stage, "name", None)
    if name:
        return str(name)
    if isinstance(stage, NavigateStage):
        return f"navigate:{stage.target.station_id}"
    operation = str(stage.operation)
    label = f"{operation}:{stage.target}"
    return label if index is None else f"{index}:{label}"


def stage_catalog(task: ExpertPlan) -> list[dict[str, Any]]:
    """Return stable names and key fields for every production stage."""

    rows: list[dict[str, Any]] = []
    for index, stage in enumerate(task.stages):
        row: dict[str, Any] = {
            "index": int(index),
            "name": stage_label(stage),
            "kind": "navigate" if isinstance(stage, NavigateStage) else "operate",
        }
        if isinstance(stage, NavigateStage):
            row.update(
                {
                    "station": stage.target.station_id,
                    "purpose": stage.target.purpose,
                    "holding": stage.target.holding,
                }
            )
        else:
            row.update(
                {
                    "operation": str(stage.operation),
                    "target": stage.target,
                }
            )
        rows.append(row)
    return rows


def select_workflow_stages(
    task: ExpertPlan,
    selectors: Sequence[str],
) -> list[Stage]:
    """Select production stages by index, explicit name, or generated label.

    The returned order follows ``selectors`` rather than the production
    workflow, allowing a caller to compose a deliberate debug sequence.
    """

    rows = stage_catalog(task)
    chosen: list[Stage] = []
    for raw_selector in selectors:
        selector = str(raw_selector).strip()
        if not selector:
            raise ValueError("stage selector must be non-empty")
        index: int | None = None
        numeric = selector[1:] if selector.startswith("#") else selector
        if numeric.isdigit():
            index = int(numeric)
            if index < 0 or index >= len(task.stages):
                raise IndexError(
                    f"stage index {index} is outside [0, {len(task.stages) - 1}]"
                )
        else:
            matches = [
                int(row["index"])
                for row in rows
                if selector
                in {
                    str(row["name"]),
                    str(getattr(task.stages[int(row["index"])], "name", "")),
                }
            ]
            if not matches:
                raise KeyError(
                    f"stage {selector!r} not found; "
                    f"available={[row['name'] for row in rows]}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"stage selector {selector!r} is ambiguous; indices={matches}"
                )
            index = matches[0]
        chosen.append(copy.deepcopy(task.stages[index]))
    return chosen


def load_stage_file(path: str | Path) -> list[Stage]:
    """Load an ordered NavigateStage / OperateStage list from JSON or YAML."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw_stages = payload.get("stages") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_stages, list):
        raise ValueError(f"stage file must contain a list or stages list: {source}")
    return [
        _stage_from_mapping(value, source=source, index=index)
        for index, value in enumerate(raw_stages)
    ]


def _stage_from_mapping(value: Any, *, source: Path, index: int) -> Stage:
    if not isinstance(value, Mapping):
        raise ValueError(f"stage {index} in {source} must be an object")
    raw = dict(value)
    kind = str(raw.pop("kind", raw.pop("type", ""))).strip().lower()
    metadata = raw.pop("metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"stage {index} metadata in {source} must be an object")
    if kind in {"navigate", "navigation"}:
        station = str(raw.pop("station", "")).strip()
        if not station:
            raise ValueError(f"navigate stage {index} in {source} needs station")
        stage = NavigateStage(
            target=StationTarget(
                station_id=station,
                holding=_optional_string(raw.pop("holding", None)),
                purpose=_optional_string(raw.pop("purpose", None)) or "",
                metadata=dict(metadata),
            ),
            name=_optional_string(raw.pop("name", None)),
            skip_if_reachable=bool(raw.pop("skip_if_reachable", False)),
            fallback_if_unreachable=bool(raw.pop("fallback_if_unreachable", True)),
            metadata=dict(metadata),
        )
    elif kind in {"operate", "operation"}:
        operation = str(raw.pop("operation", "")).strip()
        target = str(raw.pop("target", "")).strip()
        if not operation or not target:
            raise ValueError(
                f"operate stage {index} in {source} needs operation and target"
            )
        target_pos = raw.pop("target_pos", None)
        receptacle = _optional_string(raw.pop("receptacle", None))
        if receptacle is not None:
            metadata = {**dict(metadata), "receptacle": receptacle}
        if target_pos is not None:
            metadata = {
                **dict(metadata),
                "target_pos": tuple(float(item) for item in target_pos),
            }
        stage = OperateStage(
            operation=operation,
            target=target,
            metadata=dict(metadata),
            name=_optional_string(raw.pop("name", None)),
        )
    else:
        raise ValueError(
            f"stage {index} in {source} has unsupported kind {kind!r}"
        )
    if raw:
        raise ValueError(
            f"stage {index} in {source} has unknown fields: {sorted(raw)}"
        )
    return stage


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _probe_env(runtime: Any) -> Any:
    env = getattr(runtime, "env", None)
    if env is None:
        raise TypeError("probe requires an EpisodeRuntime exposing env")
    return env


def _probe_runner(runtime: Any) -> Any:
    runner = getattr(getattr(runtime, "initialized", None), "runner", None)
    if runner is None:
        runner = getattr(runtime, "runner", None)
    if runner is None:
        raise TypeError("probe requires an expert EpisodeRuntime with a StageRunner")
    return runner


def _probe_navigation(runtime: Any) -> Any:
    runner = _probe_runner(runtime)
    return getattr(runner, "navigation", getattr(runner, "navigation_controller", None))


def _probe_manipulation(runtime: Any) -> Any:
    runner = _probe_runner(runtime)
    return getattr(runner, "manipulation", getattr(runner, "manipulation_controller", None))


def station_catalog(runtime: Any) -> list[str]:
    nav = _probe_navigation(runtime)
    sampler = getattr(nav, "_sampler", getattr(nav, "station_sampler", None))
    if sampler is None:
        return []
    stations = getattr(sampler, "stations", None)
    if isinstance(stations, Mapping):
        return sorted(str(name) for name in stations)
    payload = getattr(sampler, "payload", None)
    if isinstance(payload, Mapping):
        values = payload.get("stations") or {}
        if isinstance(values, Mapping):
            return sorted(str(name) for name in values)
    return []


def resolve_station_pose(
    runtime: Any,
    *,
    station_id: str,
    station_file: str | Path | None = None,
    station_xy_yaw: Sequence[float] | None = None,
    sample_index: int | None = None,
    purpose: str | None = None,
) -> ProbeStationPose:
    """Resolve an inline, external-file, or task-owned station pose."""

    name = str(station_id).strip()
    if not name:
        raise ValueError("station_id must be non-empty")
    if station_xy_yaw is not None:
        values = np.asarray(station_xy_yaw, dtype=float).reshape(-1)
        if values.size != 3 or not np.all(np.isfinite(values)):
            raise ValueError("station_xy_yaw must contain finite x, y, yaw")
        sample = {
            "station_id": name,
            "sample_index": 0,
            "base_xy": values[:2].tolist(),
            "base_yaw": float(values[2]),
            "record_source": "inline_cli",
        }
        return ProbeStationPose(
            station_id=name,
            xy=(float(values[0]), float(values[1])),
            yaw=float(values[2]),
            source="inline_cli",
            sample=sample,
        )
    if station_file is not None:
        return _station_pose_from_file(
            station_file,
            station_id=name,
            sample_index=sample_index,
        )

    nav = _probe_navigation(runtime)
    sample_pose = getattr(nav, "sample_station_pose", None)
    if not callable(sample_pose):
        raise TypeError("task navigation controller cannot expose named station poses")
    pose = sample_pose(name, purpose=purpose)
    yaw = (
        0.0
        if pose.quaternion is None
        else quat_xyzw_to_yaw(list(pose.quaternion))
    )
    sample = {
        "station_id": name,
        "base_xy": list(pose.position[:2]),
        "base_yaw": float(yaw),
        "record_source": "task_station_sampler",
    }
    if pose.torso_height is not None:
        sample["torso_height"] = float(pose.torso_height)
    return ProbeStationPose(
        station_id=name,
        xy=(float(pose.position[0]), float(pose.position[1])),
        yaw=float(yaw),
        source="task_station_sampler",
        sample=sample,
        torso_height=pose.torso_height,
    )


def _station_pose_from_file(
    path: str | Path,
    *,
    station_id: str,
    sample_index: int | None,
) -> ProbeStationPose:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    stations = payload.get("stations") if isinstance(payload, Mapping) else None
    if not isinstance(stations, Mapping) or station_id not in stations:
        raise KeyError(
            f"station {station_id!r} not found in {source}; "
            f"available={sorted(stations or {})}"
        )
    station = stations[station_id]
    if not isinstance(station, Mapping):
        raise ValueError(f"station {station_id!r} in {source} must be an object")
    samples = station.get("samples")
    if not isinstance(samples, list):
        samples = [station]
    valid: list[tuple[dict[str, Any], np.ndarray, float]] = []
    for list_index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            continue
        sample = dict(raw_sample)
        parsed = station_sample_pose(sample)
        if parsed is None:
            continue
        xy, yaw = parsed
        sample.setdefault("sample_index", list_index)
        valid.append((sample, xy, float(yaw)))
    if not valid:
        raise ValueError(f"station {station_id!r} in {source} has no valid samples")
    if sample_index is None:
        selected = valid[0]
    else:
        matches = [
            item
            for item in valid
            if int(item[0].get("sample_index", -1)) == int(sample_index)
        ]
        if not matches:
            raise KeyError(
                f"sample_index {sample_index} is unavailable for {station_id!r}; "
                f"available={[item[0].get('sample_index') for item in valid]}"
            )
        selected = matches[0]
    sample, xy, yaw = selected
    return ProbeStationPose(
        station_id=station_id,
        xy=(float(xy[0]), float(xy[1])),
        yaw=float(yaw),
        source=str(source),
        sample=sample,
        torso_height=station_sample_torso_height(sample),
    )


def inject_station_pose(runtime: Any, station: ProbeStationPose) -> None:
    """Install an external station into the task navigation sampler."""

    nav = _probe_navigation(runtime)
    sampler = getattr(nav, "_sampler", getattr(nav, "station_sampler", None))
    if sampler is None:
        raise TypeError("task navigation controller does not expose its station sampler")
    stations = getattr(sampler, "stations", None)
    if isinstance(stations, dict):
        stations[station.station_id] = StationSpec(
            name=station.station_id,
            base_candidates=[station.as_pose()],
            description=f"Temporary probe station from {station.source}",
            metadata={"probe_station": True, "sample": dict(station.sample)},
        )
    else:
        payload = getattr(sampler, "payload", None)
        if not isinstance(payload, dict):
            raise TypeError(
                f"station sampler {type(sampler).__name__} does not support injection"
            )
        payload.setdefault("stations", {})[station.station_id] = {
            "samples": [dict(station.sample)]
        }
    reset = getattr(sampler, "reset_choices", None)
    if callable(reset):
        reset()


def reset_base_to_station(
    runtime: Any,
    station: ProbeStationPose,
) -> dict[str, Any]:
    """Kinematically place the base, apply torso height, and sync controllers."""

    obs = set_base_pose_kinematic(
        _probe_env(runtime),
        target_xy=np.asarray(station.xy, dtype=float),
        target_yaw=float(station.yaw),
    )
    torso_summary = _move_torso_to_height(
        _probe_env(runtime),
        station.torso_height,
    )
    if bool(torso_summary["applied"]):
        obs = _probe_env(runtime)._get_observations(force_update=True)
    nav = _probe_navigation(runtime)
    driver = getattr(nav, "_driver", getattr(nav, "base_driver", None))
    refresh_driver = getattr(driver, "refresh_obs", None)
    if callable(refresh_driver):
        refresh_driver()
    manip = _probe_manipulation(runtime)
    refresh_manip = getattr(manip, "refresh_obs", None)
    if callable(refresh_manip):
        refresh_manip()
    return {
        "station_id": station.station_id,
        "source": station.source,
        "requested_xy": list(station.xy),
        "requested_yaw": float(station.yaw),
        "requested_torso_height": station.torso_height,
        "actual_base_pos": np.asarray(obs["robot0_base_pos"], dtype=float).tolist(),
        "actual_base_quat_xyzw": np.asarray(
            obs["robot0_base_quat"], dtype=float
        ).tolist(),
        "torso": torso_summary,
    }


def _move_torso_to_height(
    env: Any,
    target_height: float | None,
    *,
    tolerance: float = 0.002,
    max_steps: int = 240,
) -> dict[str, Any]:
    """Apply an optional measured station torso height through normal env steps."""

    robot = env.robots[0]
    composite = robot.composite_controller
    torso = composite.part_controllers.get("torso")
    current = (
        None
        if torso is None
        else float(qpos_for_controller(env, torso).reshape(-1)[0])
    )
    if target_height is None:
        return {
            "applied": False,
            "reason": "station_has_no_torso_height",
            "requested_height": None,
            "initial_height": current,
            "actual_height": current,
            "error": None,
            "steps": 0,
        }
    target = float(target_height)
    if not np.isfinite(target):
        raise ValueError("station torso_height must be finite")
    if torso is None:
        raise RuntimeError(
            f"station requests torso_height={target:.4f}m but robot has no torso controller"
        )
    if tolerance <= 0.0:
        raise ValueError("torso tolerance must be positive")
    if max_steps <= 0:
        raise ValueError("torso max_steps must be positive")

    joint_indexes = np.asarray(torso.joint_index, dtype=int).reshape(-1)
    if joint_indexes.size < 1:
        raise RuntimeError("torso controller does not expose a joint index")
    joint_range = np.asarray(
        env.sim.model.jnt_range[int(joint_indexes[0])],
        dtype=float,
    ).reshape(-1)
    if (
        joint_range.size < 2
        or not np.all(np.isfinite(joint_range[:2]))
        or target < float(joint_range[0]) - 1e-9
        or target > float(joint_range[1]) + 1e-9
    ):
        raise ValueError(
            f"station torso_height={target:.4f}m is outside joint range "
            f"{joint_range[:2].tolist()}"
        )

    right = composite.part_controllers.get("right")
    if right is None:
        raise RuntimeError("robot does not expose the right-arm controller")
    base_start, base_end = composite._action_split_indexes["base"]
    base_action = np.zeros(int(base_end) - int(base_start), dtype=float)
    initial = float(current)
    actual = initial
    executed_steps = 0
    for _ in range(int(max_steps)):
        error = target - actual
        if abs(error) <= float(tolerance):
            break
        torso_action = float(np.clip(error / 0.05, -1.0, 1.0))
        step_base_and_hold_arm(
            env,
            arm_qpos=qpos_for_controller(env, right),
            base_action=base_action,
            base_mode=1.0,
            gripper_action=0.0,
            torso_action=torso_action,
        )
        executed_steps += 1
        actual = float(qpos_for_controller(env, torso).reshape(-1)[0])
    if abs(target - actual) > float(tolerance):
        raise RuntimeError(
            f"torso failed to reach {target:.4f}m within {max_steps} steps; "
            f"actual={actual:.4f}m"
        )

    return {
        "applied": True,
        "reason": "station_torso_height_reached",
        "requested_height": target,
        "initial_height": initial,
        "actual_height": actual,
        "error": actual - target,
        "tolerance": float(tolerance),
        "steps": int(executed_steps),
        "joint_range": joint_range[:2].astype(float).tolist(),
    }


def load_eef_pose_file(
    path: str | Path,
    *,
    env: Any,
    frame_override: str = "auto",
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Load a direct or live-anchor EEF target.

    Supported inputs:

    - ``{"frame": "world|base", "pose": {"p": ..., "q_xyzw": ...}}``
    - a raw ``{"p": ..., "q_xyzw": ...}`` or ``{"matrix": ...}``
    - existing ``config_kind: anchor_pose`` calibration records.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"EEF pose file must be a JSON object: {source}")
    override = str(frame_override).strip().lower()
    if override not in {"auto", "world", "base"}:
        raise ValueError("frame_override must be auto, world, or base")

    if str(payload.get("config_kind", "")).strip() == "anchor_pose":
        matrix, anchor_summary = _resolve_anchor_pose(env, payload, source)
        if override not in {"auto", "world"}:
            raise ValueError("live anchor poses resolve in world frame")
        return pose_from_matrix(matrix), "world", {
            "source": str(source),
            "kind": "anchor_pose",
            **anchor_summary,
        }

    declared_frame = str(payload.get("frame", "")).strip().lower()
    raw_pose: Any = payload.get("pose", payload)
    inferred_frame = ""
    if isinstance(raw_pose, Mapping) and "world" in raw_pose:
        raw_pose = raw_pose["world"]
        inferred_frame = "world"
    elif isinstance(raw_pose, Mapping) and "base" in raw_pose:
        raw_pose = raw_pose["base"]
        inferred_frame = "base"
    if not isinstance(raw_pose, Mapping):
        raise ValueError(f"EEF pose payload in {source} must be an object")
    frame = override if override != "auto" else (declared_frame or inferred_frame)
    if frame not in {"world", "base"}:
        raise ValueError(
            f"EEF pose frame is missing in {source}; set frame or --eef-frame"
        )
    matrix = matrix_from_json_pose(dict(raw_pose))
    _validate_transform(matrix, source)
    return pose_from_matrix(matrix), frame, {
        "source": str(source),
        "kind": "direct_pose",
        "frame": frame,
    }


def _resolve_anchor_pose(
    env: Any,
    payload: Mapping[str, Any],
    source: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    pose_block = payload.get("pose")
    anchor = payload.get("anchor")
    if not isinstance(pose_block, Mapping) or not isinstance(anchor, Mapping):
        raise ValueError(f"anchor pose in {source} needs pose and anchor objects")
    raw_world = pose_block.get("world")
    if not isinstance(raw_world, Mapping):
        raise ValueError(f"anchor pose in {source} needs pose.world")
    matrix = matrix_from_json_pose(dict(raw_world))
    _validate_transform(matrix, source)
    selector = str(anchor.get("selector", "")).strip()
    current = _anchor_position(env, selector)
    offset = pose_block.get("offset_from_anchor_world")
    reference = anchor.get("world")
    if offset is not None:
        resolved_offset = np.asarray(offset, dtype=float).reshape(3)
        matrix[:3, 3] = current + resolved_offset
    elif reference is not None:
        reference_position = np.asarray(reference, dtype=float).reshape(3)
        matrix[:3, 3] += current - reference_position
        resolved_offset = matrix[:3, 3] - current
    else:
        raise ValueError(
            f"anchor pose in {source} needs offset_from_anchor_world or anchor.world"
        )
    return matrix, {
        "selector": selector,
        "current_anchor_world": current.tolist(),
        "offset_from_anchor_world": resolved_offset.tolist(),
        "resolved_pose_world": matrix.tolist(),
    }


def _anchor_position(env: Any, selector: str) -> np.ndarray:
    if selector.startswith("site:"):
        name = selector.removeprefix("site:")
        index = int(env.sim.model.site_name2id(name))
        return np.asarray(env.sim.data.site_xpos[index], dtype=float)
    if selector.startswith("body:"):
        name = selector.removeprefix("body:")
        index = int(env.sim.model.body_name2id(name))
        return np.asarray(env.sim.data.body_xpos[index], dtype=float)
    if selector.startswith("geom:"):
        name = selector.removeprefix("geom:")
        index = int(env.sim.model.geom_name2id(name))
        return np.asarray(env.sim.data.geom_xpos[index], dtype=float)
    raise ValueError(
        f"unsupported anchor selector {selector!r}; expected site:, body:, or geom:"
    )


def _validate_transform(matrix: Any, source: Path) -> None:
    value = np.asarray(matrix, dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"pose in {source} is not a finite 4x4 transform")


def build_probe_task(
    production_task: ExpertPlan,
    stages: Sequence[Stage],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ExpertPlan:
    """Build a diagnostic ExpertPlan without changing task semantics."""

    values = dict(production_task.metadata)
    values.update(dict(metadata or {}))
    values.update(
        {
            "probe": True,
            "diagnostic_probe": True,
        }
    )
    return ExpertPlan(
        task_id=production_task.task_id,
        stages=tuple(stages),
        version=production_task.version,
        metadata=values,
    )


def make_move_eef_stage(
    *,
    target_pose: Mapping[str, Any],
    frame: str,
    name: str = "move_eef_probe",
    gripper_action: float = -1.0,
    holding_object: str | None = None,
    record_trajectory: bool = False,
    settle_steps: int | None = None,
) -> OperateStage:
    metadata: dict[str, Any] = {
        "target_pose": copy.deepcopy(dict(target_pose)),
        "frame": str(frame),
        "planner_stage": str(name),
        "gripper_action": float(gripper_action),
        "record_trajectory": bool(record_trajectory),
    }
    if holding_object is not None:
        metadata["holding_object"] = str(holding_object)
    if settle_steps is not None:
        metadata["settle_steps"] = int(settle_steps)
    return OperateStage(
        operation="move_eef",
        target=str(name),
        metadata=metadata,
        name=str(name),
    )


__all__ = [
    "ProbeStationPose",
    "build_probe_task",
    "inject_station_pose",
    "load_eef_pose_file",
    "load_stage_file",
    "make_move_eef_stage",
    "reset_base_to_station",
    "resolve_registered_task_id",
    "resolve_station_pose",
    "select_workflow_stages",
    "stage_catalog",
    "stage_label",
    "station_catalog",
]
