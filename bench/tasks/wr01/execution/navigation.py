"""WR-01 station mapping and mobile-base assembly."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bench.execution.navigation import (
    BaseDriverAdapter,
    NavigationController,
    RoboCasaNavigationAdapter,
    StationPathPlanner,
    station_specs_from_candidates,
)
from bench.execution.navigation.env_adapter import fixture_xy, object_xy, safe_name
from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.navigation.station_poses import (
    station_sample_pose,
    station_sample_torso_height,
    yaw_to_quat_xyzw,
)
from bench.execution.planning import NavigationConfig
from bench.execution.sampling.rng import SeedStream
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.execution.sampling.circle_sampling import sample_uniform_disk


STATION_ALIASES: dict[Any, str] = {
    "red_can": "item_station",
    "blue_can": "item_station",
    "lower": "lower_storage_station",
    "upper": "upper_storage_station",
    "lower_storage": "lower_storage_station",
    "upper_storage": "upper_storage_station",
    "storage_doors": "upper_door_station",
    "door_station": "upper_door_station",
    "lower_door": "lower_door_station",
    "upper_door": "upper_door_station",
    "memory_handoff_station": "cabinet_front_station",
    "memory_handoff": "cabinet_front_station",
}

CALIBRATED_STATION_IDS = (
    "item_station",
    "lower_storage_station",
    "upper_storage_station",
    "upper_door_station",
    "lower_door_station",
    "cabinet_front_station",
)


@dataclass(frozen=True)
class NavigationAssembly:
    controller: NavigationController
    env_adapter: RoboCasaNavigationAdapter
    station_sampler: Any
    station_sources: dict[str, dict[str, Any]]
    path_planner: StationPathPlanner | None
    base_driver: Any
    config: NavigationConfig


def build_navigation_config(
    settings: dict[str, Any],
    *,
    manual_base_path_file: Path,
) -> NavigationConfig:
    values = dict(settings)
    config = replace(
        NavigationConfig(),
        kinematic_base_navigation=bool(values.pop("kinematic_base_navigation", False)),
        base_action_frame=str(values.pop("base_action_frame", "world")),
        base_translation_action_sign=float(values.pop("base_translation_action_sign", 1.0)),
        base_tolerance=float(values.pop("base_tolerance", 0.04)),
        base_yaw_tolerance=float(values.pop("base_yaw_tolerance", 0.08)),
        base_max_steps=int(values.pop("base_max_steps", 1500)),
        manual_base_path_enabled=bool(
            values.pop("manual_base_path_enabled", True)
        ),
        manual_base_path_file=Path(manual_base_path_file),
        manual_base_path_start_index=int(
            values.pop("manual_base_path_start_index", 0)
        ),
        manual_base_path_xy_tolerance=float(
            values.pop("manual_base_path_xy_tolerance", 0.04)
        ),
        manual_base_path_yaw_tolerance=float(
            values.pop("manual_base_path_yaw_tolerance", 0.08)
        ),
        base_profile_v_max=float(values.pop("base_profile_v_max", 0.30)),
        base_profile_a_max=float(values.pop("base_profile_a_max", 0.45)),
        base_profile_a_dec=float(values.pop("base_profile_a_dec", 0.45)),
        base_profile_v_min=float(values.pop("base_profile_v_min", 0.08)),
        base_profile_w_max=float(values.pop("base_profile_w_max", 0.50)),
        base_profile_alpha_max=float(values.pop("base_profile_alpha_max", 0.80)),
        base_profile_alpha_dec=float(values.pop("base_profile_alpha_dec", 0.80)),
        base_profile_w_min=float(values.pop("base_profile_w_min", 0.08)),
        use_station_astar_paths=bool(values.pop("use_station_astar_paths", False)),
        station_nav_use_docking=bool(values.pop("station_nav_use_docking", False)),
        station_nav_dock_standoff=float(values.pop("station_nav_dock_standoff", 0.0)),
        station_nav_require_footprint=bool(values.pop("station_nav_require_footprint", False)),
    )
    if values:
        raise ValueError("unknown WR-01 navigation settings: " + ", ".join(sorted(values)))
    return config


def station_specs(env: Any) -> dict[str, dict[str, Any]]:
    lower = fixture_xy(getattr(env, "lower_storage_cabinet", None))
    upper = fixture_xy(getattr(env, "upper_storage_cabinet", None))
    return {
        "item_station": {
            "station_id": "item_station",
            "fixture": getattr(env, "counter", None),
            "anchor_xy": object_xy(env, "red_can"),
            "source_fixture": safe_name(getattr(env, "counter", None)),
            "source_object_or_phase": "red_can,blue_can",
        },
        "lower_storage_station": {
            "station_id": "lower_storage_station",
            "fixture": getattr(env, "lower_storage_cabinet", None),
            "anchor_xy": lower,
            "source_fixture": safe_name(getattr(env, "lower_storage_cabinet", None)),
            "source_object_or_phase": "store_lower,retrieve_lower",
        },
        "upper_storage_station": {
            "station_id": "upper_storage_station",
            "fixture": getattr(env, "upper_storage_cabinet", None),
            "anchor_xy": upper,
            "source_fixture": safe_name(getattr(env, "upper_storage_cabinet", None)),
            "source_object_or_phase": "store_upper,retrieve_upper",
        },
        "upper_door_station": {
            "station_id": "upper_door_station",
            "fixture": getattr(env, "upper_storage_cabinet", None),
            "anchor_xy": upper,
            "source_fixture": safe_name(getattr(env, "upper_storage_cabinet", None)),
            "source_object_or_phase": "close_upper,open_upper",
        },
        "lower_door_station": {
            "station_id": "lower_door_station",
            "fixture": getattr(env, "lower_storage_cabinet", None),
            "anchor_xy": lower,
            "source_fixture": safe_name(getattr(env, "lower_storage_cabinet", None)),
            "source_object_or_phase": "close_lower,open_lower",
        },
    }


def build_navigation_assembly(
    env: Any,
    obs: dict[str, Any],
    *,
    config: NavigationConfig,
    seed: int,
    stations_file: str | Path,
    base_driver: Any | None = None,
    station_aliases: Mapping[Any, str] | None = None,
    seed_namespace: str = "wr_01_navigation",
) -> NavigationAssembly:
    adapter = RoboCasaNavigationAdapter(env, obs, station_specs_provider=station_specs)
    driver = base_driver or BaseDriverAdapter(env, obs, nav_config=config)
    pose_provider = driver if hasattr(driver, "current_base_xy") else adapter
    generated_candidates = adapter.station_candidates()
    aliases = STATION_ALIASES if station_aliases is None else dict(station_aliases)
    generated_specs = station_specs_from_candidates(
        generated_candidates,
        aliases={key: value for key, value in aliases.items() if isinstance(key, str)},
    )
    station_path = Path(stations_file)
    calibrated_specs = _calibrated_station_specs(station_path, seed=int(seed))
    generated_specs.update(calibrated_specs)
    sampler: Any = SimpleStationSampler(
        stations=generated_specs,
        seeds=SeedStream(base_seed=int(seed), namespace=str(seed_namespace)),
    )
    planner = (
        StationPathPlanner(obstacles=adapter.build_fixture_obstacles, config=config)
        if not config.manual_base_path_enabled and config.use_station_astar_paths
        else None
    )
    return NavigationAssembly(
        controller=NavigationController(
            station_sampler=sampler,
            base_driver=driver,
            path_planner=planner,
            base_pose_provider=pose_provider,
        ),
        env_adapter=adapter,
        station_sampler=sampler,
        station_sources=_station_source_summary(
            generated_candidates=generated_candidates,
            calibrated_specs=calibrated_specs,
            stations_file=station_path,
            initial_base_xy=np.asarray(obs["robot0_base_pos"], dtype=float)[:2],
        ),
        path_planner=planner,
        base_driver=driver,
        config=config,
    )


def _calibrated_station_specs(path: Path, *, seed: int = 0) -> dict[str, StationSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stations = payload.get("stations") or {}
    resolved: dict[str, StationSpec] = {}
    for station_id in CALIBRATED_STATION_IDS:
        values = dict(stations.get(station_id) or {})
        candidates, samples = _station_candidates(
            values, station_id=station_id, seed=int(seed)
        )
        if not candidates:
            raise ValueError(
                f"invalid WR-01 station {station_id!r} in {path}: "
                "expected base_xy/base_yaw or at least one valid samples[] entry"
            )
        first = samples[0]
        resolved[station_id] = StationSpec(
            name=station_id,
            base_candidates=candidates,
            description="Manually calibrated WR01 base pose",
            metadata={
                "source": str(path),
                "source_kind": "calibrated_override",
                "schema": (
                    "station_samples"
                    if isinstance(values.get("samples"), list)
                    else "single_pose"
                ),
                # Keep the original summary fields stable for recording metadata
                # while also exposing every manually collected sample.
                "base_xy": list(first["base_xy"]),
                "base_yaw": float(first["base_yaw"]),
                "samples": samples,
                "circle": dict(values.get("circle") or {})
                if isinstance(values.get("circle"), Mapping)
                else None,
            },
        )
    return resolved


def _station_candidates(
    values: dict[str, Any],
    *,
    station_id: str = "",
    seed: int = 0,
) -> tuple[list[Pose], list[dict[str, Any]]]:
    circle = values.get("circle")
    if isinstance(circle, Mapping):
        center = np.asarray(circle.get("center_xy", ()), dtype=float).reshape(-1)
        radius = float(circle.get("radius_m", -1.0))
        yaw = float(circle.get("base_yaw", 0.0))
        torso_height = float(circle.get("torso_height", 0.20))
        if center.size < 2 or not np.all(np.isfinite(center[:2])):
            raise ValueError(f"invalid {station_id!r} circle center")
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError(f"invalid {station_id!r} circle radius")
        sampled = sample_uniform_disk(
            center[:2], radius, seed=int(seed),
            salt=0x57524353 + sum(ord(ch) for ch in str(station_id)),
        )
        sample = {
            "sample_index": 0,
            "base_xy": sampled.astype(float).tolist(),
            "base_yaw": yaw,
            "torso_height": torso_height,
            "record_source": str(circle.get("record_source", "circle_region")),
            "circle": {"center_xy": center[:2].astype(float).tolist(), "radius_m": radius},
        }
        pose = Pose(
            position=(float(sampled[0]), float(sampled[1]), 0.0),
            quaternion=tuple(float(v) for v in yaw_to_quat_xyzw(yaw)),
            torso_height=torso_height,
        )
        return [pose], [sample]

    raw_samples = values.get("samples")
    if isinstance(raw_samples, list):
        source_samples = list(raw_samples)
    else:
        source_samples = [
            {
                "base_xy": values.get("base_xy"),
                "base_yaw": values.get("base_yaw"),
                "sample_index": 0,
            }
        ]

    candidates: list[Pose] = []
    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(source_samples):
        if not isinstance(sample, dict):
            continue
        parsed = station_sample_pose(sample)
        if parsed is None:
            continue
        xy, yaw = parsed
        base_pos = np.asarray(sample.get("base_pos") or (), dtype=float).reshape(-1)
        base_z = (
            float(base_pos[2])
            if base_pos.size >= 3 and np.isfinite(base_pos[2])
            else 0.0
        )
        candidates.append(
            Pose(
                position=(float(xy[0]), float(xy[1]), base_z),
                quaternion=tuple(float(v) for v in yaw_to_quat_xyzw(yaw)),
                torso_height=station_sample_torso_height(sample),
            )
        )
        normalized = {
            "sample_index": int(sample.get("sample_index", index)),
            "base_xy": [float(xy[0]), float(xy[1])],
            "base_yaw": float(yaw),
        }
        torso_height = station_sample_torso_height(sample)
        if torso_height is not None:
            normalized["torso_height"] = float(torso_height)
        samples.append(normalized)
    return candidates, samples


def _station_source_summary(
    *,
    generated_candidates: dict[str, list[dict[str, Any]]],
    calibrated_specs: dict[str, StationSpec],
    stations_file: Path,
    initial_base_xy: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Explain which candidate source wins before station sampling."""

    summary: dict[str, dict[str, Any]] = {}
    for station_id in CALIBRATED_STATION_IDS:
        generated = list(generated_candidates.get(station_id, ()))
        selected = calibrated_specs[station_id]
        metadata = dict(selected.metadata)
        base_xy = np.asarray(metadata["base_xy"], dtype=float)
        anchors = [
            candidate.get("anchor_xy")
            for candidate in generated
            if candidate.get("anchor_xy") is not None
        ]
        summary[station_id] = {
            "resolution": "calibrated_override",
            "calibration_file": str(stations_file),
            "calibration_schema": metadata.get("schema"),
            "calibrated_samples": list(metadata.get("samples") or ()),
            "generated_candidate_count": len(generated),
            "selected_candidate_count": len(selected.base_candidates),
            "selected_base_xy": base_xy.astype(float).tolist(),
            "selected_base_yaw": float(metadata["base_yaw"]),
            "initial_base_distance_m": float(
                np.linalg.norm(base_xy - np.asarray(initial_base_xy, dtype=float)[:2])
            ),
            "generated_anchor_xy": None if not anchors else list(anchors[0]),
        }
    return summary


__all__ = [
    "NavigationAssembly",
    "CALIBRATED_STATION_IDS",
    "STATION_ALIASES",
    "build_navigation_assembly",
    "build_navigation_config",
    "station_specs",
]
