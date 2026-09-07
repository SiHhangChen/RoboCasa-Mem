"""WR-08 physical operation handlers."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.manipulation.object_interactions import get_object_hold_status
from bench.execution.planning import object_reference_pos


def build_operation_handlers(
    controller: Any, *, dry_run_only: bool = True
) -> dict[str, Any]:
    def latch(item: str, **kwargs: Any) -> dict[str, Any]:
        source_color = kwargs.pop("source_color", None)
        passed = bool(controller.env.latch_source_color_memory(str(item)))
        if source_color is not None:
            passed = passed and controller.env.source_color_by_item.get(
                str(item)
            ) == str(source_color)
        return {
            "primitive": "latch_source_color_memory",
            "pass": passed,
            "checks": {"source_color_memory_latched": passed},
            "item": str(item),
            "source_color": controller.env.source_color_by_item.get(str(item)),
            "reason": None if passed else "source_color_memory_not_ready",
        }

    def place(target: str, **kwargs: Any) -> dict[str, Any]:
        env = controller.env
        resolver = str(kwargs.pop("target_resolver", ""))
        slot = str(kwargs.pop("target_slot", ""))
        if resolver == "target_plate_pose":
            target_pos = env.target_plate_pose(slot, str(target))
        elif resolver == "source_plate_pose":
            target_pos = env.source_plate_pose(slot)
        else:
            target_pos = kwargs.pop("target_pos_world", None)
        if target_pos is None:
            return {"primitive": "place_at_accessible", "object_name": str(target), "pass": False, "reason": "target_pos_required"}
        target_pos = np.asarray(target_pos, dtype=float).reshape(-1)[:3]
        object_name = str(target)
        execute = bool(controller.runtime_config.execute)
        target_name = str(kwargs.pop("target_name", f"{object_name}_{slot}"))
        collision_options = kwargs.pop("collision_options", None)
        placement_reference = kwargs.pop(
            "placement_reference",
            kwargs.pop("place_reference", None),
        )
        placement_object_upright = bool(
            kwargs.pop("placement_object_upright", False)
        )
        placement_approach_angles = kwargs.pop(
            "placement_approach_angles_deg",
            kwargs.pop("place_approach_angles_deg", None),
        )
        placement_yaw_offsets = kwargs.pop("placement_yaw_offsets_deg", None)
        if placement_yaw_offsets is None and "placement_yaw_offset_deg" not in kwargs:
            yaw = float(env.target_item_yaw_by_item.get(object_name, 0.0))
            kwargs["placement_yaw_offset_deg"] = float(np.degrees(yaw))

        expected_color = kwargs.pop("source_color", None)
        if resolver == "target_plate_pose" and expected_color is not None:
            if env.target_plate_color_by_slot.get(slot) != str(expected_color):
                return {
                    "primitive": "place_at_accessible",
                    "object_name": object_name,
                    "target_slot": slot,
                    "pass": False,
                    "reason": "target_plate_color_mismatch",
                }

        def pre_release() -> dict[str, Any]:
            controller.refresh_obs()
            hold = get_object_hold_status(env, object_name, max_end_effector_distance=0.15)
            bottom = object_reference_pos(env, object_name, placement_reference="bbox_bottom_center")
            rotation = np.asarray(env.sim.data.body_xmat[env.obj_body_id[object_name]], dtype=float).reshape(3, 3)
            checks = {
                "object_still_held": bool(hold["held"]),
                "object_xy_at_target": bool(np.linalg.norm(bottom[:2] - target_pos[:2]) <= 0.14),
                "object_bottom_near_plate": bool(abs(float(bottom[2] - target_pos[2])) <= 0.08),
                "object_upright": bool(float(rotation[2, 2]) >= 0.90),
            }
            return {"pass": bool(all(checks.values())), "checks": checks, "hold_status": hold, "bottom_world": bottom.tolist()}

        if placement_yaw_offsets is not None:
            if isinstance(placement_yaw_offsets, str):
                raw_offsets = [
                    item.strip()
                    for item in placement_yaw_offsets.replace(";", ",").split(",")
                    if item.strip()
                ]
            elif isinstance(placement_yaw_offsets, (list, tuple, np.ndarray)):
                raw_offsets = list(placement_yaw_offsets)
            else:
                raw_offsets = [placement_yaw_offsets]
            try:
                yaw_offsets = [float(value) for value in raw_offsets]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "WR-08 placement_yaw_offsets_deg must contain numeric values"
                ) from exc
            if not yaw_offsets or not np.all(np.isfinite(yaw_offsets)):
                raise ValueError(
                    "WR-08 placement_yaw_offsets_deg must be a nonempty finite list"
                )
            candidates = [
                {
                    "target_pos_world": target_pos.tolist(),
                    "placement_yaw_offset_deg": float(yaw),
                    "metadata": {
                        "placement_yaw_offset_deg": float(yaw),
                        "fixed_candidate_order": int(index),
                    },
                }
                for index, yaw in enumerate(yaw_offsets)
            ]
            result = controller.place_at_world_pose_candidates(
                object_name,
                candidate_groups=[
                    {
                        "group_name": "wr08_placement_yaw_candidates",
                        "candidates": candidates,
                    }
                ],
                target_name=target_name,
                collision_options=collision_options,
                placement_reference=placement_reference,
                placement_object_upright=placement_object_upright,
                placement_approach_angles_deg=placement_approach_angles,
                pre_release_check=pre_release if execute else None,
                **kwargs,
            )
        else:
            result = controller.place_at_world_pose(
                object_name,
                target_pos_world=target_pos,
                target_name=target_name,
                collision_options=collision_options,
                placement_reference=placement_reference,
                placement_object_upright=placement_object_upright,
                placement_approach_angles_deg=placement_approach_angles,
                pre_release_check=pre_release if execute else None,
                **kwargs,
            )
        result["primitive"] = "place_at_accessible"
        result["dry_run_only"] = bool(dry_run_only)
        result["target_slot"] = slot
        result["target_pos_world"] = target_pos.tolist()
        return result

    def verify(item: str, **kwargs: Any) -> dict[str, Any]:
        slot = str(kwargs.pop("target_slot"))
        source_color = str(kwargs.pop("source_color"))
        env = controller.env
        env.update_state()
        target_color = env.target_plate_color_by_slot[slot]
        physical = bool(env.item_at_target(str(item), slot))
        matches = target_color == source_color
        semantic = bool(
            env._semantic_state.target_color_matches_by_item.get(str(item), False)
        )
        passed = bool(physical and matches and semantic)
        return {
            "primitive": "verify_target_plate",
            "pass": passed,
            "item": str(item),
            "target_slot": slot,
            "source_color": source_color,
            "target_color": target_color,
            "checks": {
                "item_at_target": physical,
                "target_color_matches": matches,
                "semantic_placement": semantic,
            },
            "reason": None if passed else "matching_target_plate_not_verified",
        }

    return {
        "latch_source_color_memory": latch,
        "place_at_accessible": place,
        "verify_target_plate": verify,
    }


__all__ = ["build_operation_handlers"]
