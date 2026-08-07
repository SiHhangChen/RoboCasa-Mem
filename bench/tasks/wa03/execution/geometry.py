"""将 WA-03 领域目标转换为控制器可执行的世界位姿。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from bench.tasks.wa03.domain.geometry import is_xy_inside_region


PLACE_OPTION_KEYS = (
    "place_height",
    "retreat_height",
    "retreat_x",
    "retreat_y",
    "release_offset_x",
    "release_offset_y",
    "post_release_clearance_distance",
    "post_release_clearance_z_offset",
    "placement_reference",
    "placement_yaw_offset_deg",
    "placement_yaw_offsets_deg",
    "placement_approach_angles_deg",
    "placement_entry_standoff",
    "placement_exit_standoff",
    "placement_entry_z_offset",
    "placement_exit_z_offset",
    "allow_place_near_miss",
    "allow_place_qpos_near_miss",
    "place_near_miss_max_position_error",
    "place_near_miss_max_qpos_error",
    "entry_settle_steps",
    "place_settle_steps",
    "clearance_settle_steps",
    "retreat_settle_steps",
    "release_settle_steps",
)


def resolve_dynamic_place_target(
    controller: Any,
    object_name: str,
    *,
    target_kind: str,
    options: Mapping[str, Any],
) -> dict[str, Any] | None:
    """解析 WA-03 正式流程使用的餐桌动态放置目标。"""
    if str(target_kind).strip() != "dining":
        return None
    remaining = dict(options)
    sampled = remaining.pop("dining_sampled_target", None)
    region = remaining.pop("dining_region", None)
    slot = str(remaining.pop("dining_target_slot", ""))
    remaining.pop("dining_region_config", None)
    if not isinstance(sampled, Mapping) or not isinstance(region, Mapping):
        raise ValueError("WA-03 dining placement requires a pre-sampled target and region")
    target_xy = np.asarray(sampled.get("xy", []), dtype=float).reshape(-1)
    if target_xy.size != 2 or not np.all(np.isfinite(target_xy)):
        raise ValueError("WA-03 dining target must contain finite world-frame XY")
    if not is_xy_inside_region(target_xy, region):
        raise ValueError("WA-03 dining target lies outside its assigned region")
    top_world_z = _counter_top_world_z(controller.env.dining_counter)
    target_position = [float(target_xy[0]), float(target_xy[1]), top_world_z]
    target_info = {
        "target_kind": "dining_region",
        "mug_name": object_name,
        "slot_name": slot or str(region.get("slot", "")),
        "target_pos_world": list(target_position),
        "sampled_target": dict(sampled),
        "region": dict(region),
        "target_source": "episode_seeded_region_sample",
    }
    return {
        "target_name": f"{object_name}_dining_spot",
        "target_pos_world": target_position,
        "target_info": target_info,
        "place_kwargs": {
            key: remaining[key] for key in PLACE_OPTION_KEYS if key in remaining
        },
        "collision_options": None,
    }


def build_reachability_target_resolver(
    controller: Any,
) -> Callable[[dict[str, Any]], dict[str, Any] | None]:
    """构建供导航可达性预检查调用的动态目标解析回调。"""

    def resolver(request: dict[str, Any]) -> dict[str, Any] | None:
        """把通用可达性请求转换成 WA-03 世界坐标目标。"""

        options = dict(request.get("kwargs") or {})
        target_kind = str(options.get("target_kind") or "").strip()
        if not target_kind:
            return None
        object_name = str(
            request.get("holding")
            or request.get("object_name")
            or request.get("anchor")
        )
        target = resolve_dynamic_place_target(
            controller,
            object_name,
            target_kind=target_kind,
            options=options,
        )
        if target is None:
            return None
        return {
            "mode": "world",
            "object_name": object_name,
            "target_kind": target_kind,
            "target_name": target["target_name"],
            "target_pos_world": target["target_pos_world"],
            "target_info": target["target_info"],
            "collision_options": target.get("collision_options"),
            **dict(target.get("place_kwargs") or {}),
        }

    return resolver


def _counter_top_world_z(counter: Any) -> float:
    """根据 fixture ext site 或尺寸计算餐桌顶面的世界 Z。"""

    from robocasa.utils import object_utils as OU

    try:
        _, _, _, top_relative = counter.get_ext_sites(relative=True)
        relative_z = float(np.asarray(top_relative, dtype=float)[2])
    except Exception:
        size = np.asarray(getattr(counter, "size", [3.5, 0.65, 0.92]), dtype=float)
        relative_z = float(size[2]) * 0.5
    world = np.asarray(
        OU.get_pos_after_rel_offset(
            counter,
            np.asarray([0.0, 0.0, relative_z], dtype=float),
        ),
        dtype=float,
    )
    return float(world[2])


__all__ = ["build_reachability_target_resolver", "resolve_dynamic_place_target"]
