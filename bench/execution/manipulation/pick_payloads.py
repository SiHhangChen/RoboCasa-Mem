"""Pick-target payload resolution for manipulation controllers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from bench.execution.planning import (
    ManualGraspConfig,
    PickPlaceConfig,
    make_manual_object_pick_target_poses,
    make_object_pick_target_poses,
)


PickPayloadSelector = Callable[[dict[str, Any]], dict[str, Any] | None]
DryRunStageFn = Callable[..., dict[str, Any]]
DryRunPickSequenceFn = Callable[..., dict[str, Any]]
DryRunPickPlaceFn = Callable[..., dict[str, Any]]


def resolve_pick_payload(
    *,
    env: Any,
    obs: dict[str, Any],
    object_name: str,
    kwargs: dict[str, Any],
    pick_place_config: PickPlaceConfig,
    manual_grasp_config: ManualGraspConfig,
    pick_payload_selector: PickPayloadSelector | None,
    dry_run_stage: DryRunStageFn,
    dry_run_pick_sequence: DryRunPickSequenceFn | None = None,
    dry_run_pick_then_world_place: DryRunPickPlaceFn | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    selector = kwargs.get("pick_payload_selector") or pick_payload_selector
    if selector is not None:
        selected = selector(
            _pick_selector_request(
                env=env,
                obs=obs,
                object_name=object_name,
                kwargs=kwargs,
                pick_place_config=pick_place_config,
                manual_grasp_config=manual_grasp_config,
                dry_run_stage=dry_run_stage,
                dry_run_pick_sequence=dry_run_pick_sequence,
                dry_run_pick_then_world_place=dry_run_pick_then_world_place,
            )
        )
        if selected is not None:
            payload = selected.get("payload")
            selection = selected.get("selection")
            if selection is None:
                selection = {
                    key: value
                    for key, value in selected.items()
                    if key != "payload"
                }
            return payload, selection

    manual_pose_file = (
        kwargs.get("manual_grasp_pose_file")
        or kwargs.get("grasp_pose_file")
        or manual_grasp_config.manual_grasp_pose_file
    )
    if manual_pose_file is not None:
        payload = make_manual_object_pick_target_poses(
            env,
            obs,
            object_name=object_name,
            grasp_pose_file=manual_pose_file,
            pregrasp_distance=manual_pregrasp_distance(
                kwargs,
                manual_grasp_config=manual_grasp_config,
            ),
            lift_height=pick_lift_height(
                kwargs,
                pick_place_config=pick_place_config,
            ),
            object_pos_world=kwargs.get("object_pos_world"),
            object_pose_world=kwargs.get("object_pose_world"),
        )
        return payload, {
            "enabled": False,
            "mode": "single_file",
            "pose_file": str(manual_pose_file),
            "pass": True,
        }

    payload = make_object_pick_target_poses(
        env,
        obs,
        object_name=object_name,
        orientation_mode=kwargs.get(
            "orientation_mode",
            pick_place_config.orientation_mode,
        ),
        pregrasp_height=float(
            kwargs.get("pregrasp_height", pick_place_config.pregrasp_height)
        ),
        grasp_clearance=float(
            kwargs.get("grasp_clearance", pick_place_config.grasp_clearance)
        ),
        lift_height=pick_lift_height(
            kwargs,
            pick_place_config=pick_place_config,
        ),
        grasp_yaw_offset_deg=float(
            kwargs.get("grasp_yaw_offset_deg", pick_place_config.grasp_yaw_offset_deg)
        ),
        object_pos_world=kwargs.get("object_pos_world"),
        grasp_offset_x=float(kwargs.get("grasp_offset_x", 0.0)),
        grasp_offset_y=float(kwargs.get("grasp_offset_y", 0.0)),
        grasp_offset_z=float(kwargs.get("grasp_offset_z", 0.0)),
        grasp_finger_offset_x=float(kwargs.get("grasp_finger_offset_x", 0.0)),
        grasp_finger_offset_y=float(kwargs.get("grasp_finger_offset_y", 0.0)),
    )
    return payload, None


def pick_lift_height(
    kwargs: dict[str, Any],
    *,
    pick_place_config: PickPlaceConfig,
) -> float:
    if "lift_height" in kwargs:
        return float(kwargs["lift_height"])
    is_mug = _bool_kw(kwargs, ("is_mug_object", "use_mug_lift_height"), False)
    return pick_place_config.lift_height_for_object(is_mug=is_mug)


def _bool_kw(
    kwargs: dict[str, Any],
    names: tuple[str, ...],
    default: bool,
) -> bool:
    for name in names:
        if name not in kwargs:
            continue
        value = kwargs[name]
        if isinstance(value, bool):
            return value
        raw = str(value).strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off"}:
            return False
        return bool(value)
    return bool(default)


def manual_pregrasp_distance(
    kwargs: dict[str, Any],
    *,
    manual_grasp_config: ManualGraspConfig,
) -> float | None:
    value = kwargs.get(
        "manual_grasp_pregrasp_distance",
        kwargs.get("pregrasp_distance", manual_grasp_config.pregrasp_distance),
    )
    return None if value is None else float(value)


def _pick_selector_request(
    *,
    env: Any,
    obs: dict[str, Any],
    object_name: str,
    kwargs: dict[str, Any],
    pick_place_config: PickPlaceConfig,
    manual_grasp_config: ManualGraspConfig,
    dry_run_stage: DryRunStageFn,
    dry_run_pick_sequence: DryRunPickSequenceFn | None,
    dry_run_pick_then_world_place: DryRunPickPlaceFn | None,
) -> dict[str, Any]:
    return {
        "env": env,
        "obs": obs,
        "object_name": object_name,
        "kwargs": dict(kwargs),
        "pick_place_config": pick_place_config,
        "manual_grasp_config": manual_grasp_config,
        "lift_height": pick_lift_height(
            kwargs,
            pick_place_config=pick_place_config,
        ),
        "manual_pregrasp_distance": manual_pregrasp_distance(
            kwargs,
            manual_grasp_config=manual_grasp_config,
        ),
        "dry_run_stage": dry_run_stage,
        "dry_run_pick_sequence": dry_run_pick_sequence,
        "dry_run_pick_then_world_place": dry_run_pick_then_world_place,
    }


__all__ = [
    "PickPayloadSelector",
    "resolve_pick_payload",
]
