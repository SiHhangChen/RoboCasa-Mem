"""Thin WR-01 adapters between declarative stages and generic controllers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from bench.tasks.wr03.execution.geometry import storage_target
from bench.tasks.wr03.execution.door import move_slide_door
from bench.tasks.wr03.execution.retrieval import retrieve_item_from_cabinet
from bench.tasks.wr03.execution.storage import (
    adopt_picked_item,
    place_item_with_calibration,
)


def build_stage_handlers(
    controller: Any,
    *,
    resources: Mapping[str, str | Path] | None = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Install task-specific checks and calibrated physical action adapters."""

    paths = {key: Path(value) for key, value in dict(resources or {}).items()}

    def place_storage(target: str, **kwargs: Any) -> dict[str, Any]:
        level = str(kwargs.pop("level"))
        if not controller.runtime_config.execute:
            return _dry("place_in_storage", target, level=level)
        return place_item_with_calibration(
            controller,
            str(target),
            level=level,
            preinsert_file=_required_path(paths, "lower_preinsert_pose"),
            release_file=_required_path(paths, "lower_release_pose"),
            insertion_plan_attempts=int(kwargs.pop("insertion_plan_attempts", 3)),
            release_settle_steps=int(kwargs.pop("release_settle_steps", 15)),
            held_gripper_action=float(kwargs.pop("held_gripper_action", 0.0)),
        )

    def pick_colored_item(target: str, **kwargs: Any) -> dict[str, Any]:
        if not controller.runtime_config.execute:
            return _dry("pick_colored_item", str(target))
        mode = str(kwargs.pop("mode", "manual_side"))
        disable_collisions = bool(
            kwargs.pop("disable_collisions_during_transport", True)
        )
        if mode == "cabinet_front_corridor":
            return retrieve_item_from_cabinet(
                controller,
                str(target),
                lower_prepose_file=_required_path(
                    paths, "lower_retrieval_prepose"
                ),
                front_clearance_distance=float(
                    kwargs.pop("front_clearance_distance", 0.10)
                ),
                front_pregrasp_distance=float(
                    kwargs.pop("front_pregrasp_distance", 0.10)
                ),
                front_clearance_z_offset=float(
                    kwargs.pop("front_clearance_z_offset", 0.10)
                ),
                grasp_z_offset=float(kwargs.pop("grasp_z_offset", 0.010)),
                grasp_lateral_offset=float(
                    kwargs.pop("grasp_lateral_offset", -0.0067)
                ),
                grasp_inward_offset=float(
                    kwargs.pop("grasp_inward_offset", 0.0042)
                ),
                close_gripper_steps=int(kwargs.pop("close_gripper_steps", 6)),
                post_close_hold_steps=int(kwargs.pop("post_close_hold_steps", 25)),
                held_gripper_action=float(kwargs.pop("held_gripper_action", 0.0)),
                disable_collisions_during_transport=disable_collisions,
            )
        if mode != "manual_side":
            raise ValueError(f"unsupported WR-01 item pick mode: {mode!r}")
        kwargs["manual_grasp_pose_file"] = _required_path(paths, "item_grasp_pose")
        result = pick_colored_item_for_task(controller, str(target), **kwargs)
        adoption = (
            adopt_picked_item(
                controller,
                str(target),
                disable_collisions=disable_collisions,
            )
            if bool(result.get("pass", False))
            else {"pass": False, "reason": "pick_failed"}
        )
        return {
            **result,
            "transport_adoption": adoption,
            "pass": bool(result.get("pass", False) and adoption.get("pass", False)),
            "reason": (
                result.get("reason")
                if not result.get("pass", False)
                else adoption.get("reason")
            ),
        }

    def verify_storage_handler(target: str, **kwargs: Any) -> dict[str, Any]:
        return verify_storage(controller, target, level=str(kwargs.pop("level")), **kwargs)

    def close_door(target: str, **kwargs: Any) -> dict[str, Any]:
        level = str(target)
        result = _physical_door(
            controller,
            paths=paths,
            level=level,
            opening=False,
            record_retrieval=False,
            kwargs=kwargs,
        )
        if not controller.runtime_config.execute:
            return result

        env = _refresh(controller)
        physically_closed = bool(env.is_door_closed(level))
        close_recorded = bool(
            result.get("pass", False)
            and physically_closed
            and env.record_storage_door_closed(level)
        )
        checks = dict(result.get("checks") or {})
        checks.update(
            {
                "door_physically_closed": physically_closed,
                "post_storage_close_recorded": close_recorded,
            }
        )
        passed = bool(result.get("pass", False) and all(checks.values()))
        result.update(
            {
                "checks": checks,
                "pass": passed,
                "reason": None if passed else "wr03_storage_door_close_failed",
            }
        )
        return result

    def open_storage_door(target: str, **kwargs: Any) -> dict[str, Any]:
        return _physical_door(
            controller,
            paths=paths,
            level=str(target),
            opening=True,
            record_retrieval=False,
            kwargs=kwargs,
        )

    def open_door(target: str, **kwargs: Any) -> dict[str, Any]:
        return _physical_door(
            controller,
            paths=paths,
            level=str(target),
            opening=True,
            record_retrieval=True,
            kwargs=kwargs,
        )

    def hide(_: str, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return verify_and_latch_storage_hidden(controller)

    def verify_retrieval_door_handler(target: str, **kwargs: Any) -> dict[str, Any]:
        return verify_retrieval_door(controller, str(target), **kwargs)

    return {
        "pick_colored_item": pick_colored_item,
        "place_in_storage": place_storage,
        "verify_storage": verify_storage_handler,
        "close_storage_door": close_door,
        "open_storage_door": open_storage_door,
        "open_retrieval_door": open_door,
        "latch_storage_hidden": hide,
        "verify_retrieval_door": verify_retrieval_door_handler,
    }


def pick_colored_item_for_task(
    controller: Any,
    item_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Pick an item physically, while keeping assembly dry-runs planner-free.

    ``execute=False`` is used to validate task assembly and record the fixed
    workflow; it intentionally does not claim a simulated grasp.  The real
    execution path is still the shared physical pick controller.
    """

    if not controller.runtime_config.execute:
        return _dry("pick_colored_item", item_id)
    result = controller.pick(item_id, **kwargs)
    return {
        **result,
        "primitive": "pick_colored_item",
        "object_name": item_id,
    }


def place_in_storage(controller: Any, item_id: str, *, level: str, **kwargs: Any) -> dict[str, Any]:
    """Place a held item at its physical cabinet-floor target without semantic fallback."""

    if not controller.runtime_config.execute:
        return _dry("place_in_storage", item_id, level=level)
    target = storage_target(controller, level)
    result = controller.place_at_world_pose(
        item_id,
        target_name=target["target_name"],
        target_pos_world=target["target_pos_world"],
        **kwargs,
    )
    return {
        "primitive": "place_in_storage",
        "object_name": item_id,
        "level": level,
        "target_info": target["target_info"],
        "place_result": result,
        "pass": bool(result.get("pass", False)),
        "reason": result.get("reason"),
    }


def verify_storage(controller: Any, item_id: str, *, level: str, **kwargs: Any) -> dict[str, Any]:
    """Commit a storage relation only after its physical containment is observed."""

    del kwargs
    if not controller.runtime_config.execute:
        return _dry("verify_storage", item_id, level=level)
    env = _refresh(controller)
    expected = env.level_by_item.get(item_id)
    checks = {
        "known_expected_level": expected == level,
        "item_inside_level": env.is_item_in_level(item_id, level),
        "door_open_for_storage": env.is_door_open(level),
    }
    committed = False
    if all(checks.values()):
        committed = bool(env.record_storage(item_id, level))
    checks["storage_event_recorded"] = committed
    return {
        "primitive": "verify_storage",
        "object_name": item_id,
        "level": level,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": None if all(checks.values()) else "invalid_physical_storage",
    }


def verify_and_latch_storage_hidden(controller: Any) -> dict[str, Any]:
    if not controller.runtime_config.execute:
        return _dry("latch_storage_hidden")
    env = _refresh(controller)
    stored = all(
        env.is_item_in_level(item, env.level_by_item[item]) for item in env.level_by_item
    )
    closed = env.are_cabinets_closed()
    latched = env.latch_storage_hidden()
    checks = {
        "both_items_physically_stored": stored,
        "both_cabinets_closed": closed,
        "storage_hidden_latched": latched,
    }
    return {
        "primitive": "latch_storage_hidden",
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": None if all(checks.values()) else "storage_not_hidden",
    }


def verify_retrieval_door(controller: Any, level: str, **kwargs: Any) -> dict[str, Any]:
    del kwargs
    if not controller.runtime_config.execute:
        return _dry("verify_retrieval_door", level)
    env = _refresh(controller)
    checks = {
        "door_open": env.is_door_open(level),
        "target_level": level == env.target_level,
        "no_wrong_level_opened": not env._semantic_state.wrong_retrieval_level_opened,
    }
    return {
        "primitive": "verify_retrieval_door",
        "level": level,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "reason": None if all(checks.values()) else "wrong_or_unopened_retrieval_door",
    }


def _refresh(controller: Any) -> Any:
    controller.refresh_obs()
    controller.env.update_state()
    return controller.env


def _dry(primitive: str, object_name: str | None = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "primitive": primitive,
        "dry_run": True,
        "checks": {"deferred_to_execute": True},
        "pass": True,
        **extra,
    }
    if object_name is not None:
        result["object_name"] = object_name
    return result


def _physical_door(
    controller: Any,
    *,
    paths: Mapping[str, Path],
    level: str,
    opening: bool,
    record_retrieval: bool,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if not controller.runtime_config.execute:
        return _dry("open_storage_door" if opening else "close_storage_door", level)
    values = dict(kwargs)
    values.pop("open_fraction", None)
    values.pop("control_mode", None)
    pose_key = "lower_door_grasp_pose" if level == "lower" else "upper_door_grasp_pose"
    result = move_slide_door(
        controller,
        level=level,
        pose_file=_required_path(paths, pose_key),
        opening=opening,
        pregrasp_distance=float(values.pop("pregrasp_distance", 0.14)),
        retreat_distance=float(values.pop("retreat_distance", 0.10)),
        open_distances=tuple(float(v) for v in values.pop("open_distances", (0.10, 0.22, 0.36))),
        close_deltas=tuple(float(v) for v in values.pop("close_deltas", (0.12, 0.27, 0.36))),
        gripper_open_steps=int(values.pop("gripper_open_steps", 25)),
        gripper_close_steps=int(values.pop("gripper_close_steps", 60)),
        gripper_release_steps=int(values.pop("gripper_release_steps", 45)),
        motion_settle_steps=max(0, int(values.pop("motion_settle_steps", 2))),
        final_motion_settle_steps=max(
            0, int(values.pop("final_motion_settle_steps", 5))
        ),
        retreat_settle_steps=max(0, int(values.pop("retreat_settle_steps", 5))),
        opened_threshold=float(values.pop("opened_threshold", 0.70)),
        closed_threshold=float(values.pop("closed_threshold", 0.08)),
    )
    if values:
        result["unused_metadata"] = values
    if opening and record_retrieval and bool(result.get("pass", False)):
        result["recorded_retrieval_decision"] = bool(
            controller.env.record_retrieval_door_open(level)
        )
        result["pass"] = bool(
            result["pass"] and result["recorded_retrieval_decision"]
        )
    return result


def _required_path(paths: Mapping[str, Path], name: str) -> Path:
    path = paths.get(name)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"missing WR-01 execution resource {name!r}: {path}")
    return path


__all__ = [
    "build_stage_handlers",
    "pick_colored_item_for_task",
    "place_in_storage",
    "verify_and_latch_storage_hidden",
    "verify_retrieval_door",
    "verify_storage",
]
