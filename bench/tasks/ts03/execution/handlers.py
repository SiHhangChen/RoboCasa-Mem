"""Thin TS01 adapters between workflow operations and the controller."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from bench.tasks.ts03.execution.completion_button import (
    PRESS_COMPLETION_BUTTON_OPERATION,
    press_completion_button,
)
from bench.tasks.ts03.execution.geometry import (
    build_saucepan_release_candidate_groups,
    live_saucepan_anchor,
    live_saucepan_pose,
    saucepan_pose_guard,
)


LIVE_SUGAR_PLACE_OPERATION = "place_sugar_in_live_saucepan"


def build_stage_handlers(
    controller: Any,
    *,
    handler_settings: Mapping[str, Any] | None = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    defaults = dict(handler_settings or {})

    def place_live(sugar_name: str, **kwargs: Any) -> dict[str, Any]:
        values = dict(kwargs)
        values.pop("target_pos", None)
        values.pop("target_pos_world", None)
        values.pop("release_offset_x", None)
        values.pop("release_offset_y", None)
        region_order = values.pop("release_region_order")
        release_regions_xy = values.pop("release_regions_xy")
        yaw_offsets = values.pop("placement_yaw_offsets_deg")
        max_radius = float(values.pop("release_target_max_radius_m"))
        translation_guard = float(values.pop("saucepan_translation_guard_m"))
        tilt_guard = float(values.pop("saucepan_tilt_guard_deg"))
        anchor = live_saucepan_anchor(controller.env)
        saucepan_pose_at_start = live_saucepan_pose(controller.env)
        candidate_groups = build_saucepan_release_candidate_groups(
            anchor,
            region_order=region_order,
            release_regions_xy=release_regions_xy,
            yaw_offsets_deg=yaw_offsets,
            max_radius_m=max_radius,
        )

        def check_saucepan_before_release() -> dict[str, Any]:
            return saucepan_pose_guard(
                controller.env,
                saucepan_pose_at_start,
                max_translation_m=translation_guard,
                max_tilt_deg=tilt_guard,
            )

        result = controller.place_at_world_pose_candidates(
            str(sugar_name),
            candidate_groups=candidate_groups,
            pre_release_check=check_saucepan_before_release,
            **values,
        )
        result.update(
            {
                "primitive": LIVE_SUGAR_PLACE_OPERATION,
                "live_receptacle_anchor_world": np.asarray(anchor, dtype=float).tolist(),
                "live_receptacle_anchor_resolved": True,
                "release_region_order": list(region_order),
                "saucepan_pose_at_place_start_world": np.asarray(
                    saucepan_pose_at_start,
                    dtype=float,
                ).tolist(),
            }
        )
        return result

    def press_finish(_target: str, **kwargs: Any) -> dict[str, Any]:
        values = dict(kwargs)
        # Handler settings remain a narrow override surface for calibration;
        # unrelated sugar-delivery flags are never forwarded to the primitive.
        for key in (
            "prepress_z_offset",
            "press_z_offset",
            "hold_steps",
            "gripper_action",
            "near_miss_position_tolerance",
            "minimum_press_displacement_m",
        ):
            if key in defaults and key not in values:
                values[key] = defaults[key]
        return press_completion_button(controller, **values)

    return {
        LIVE_SUGAR_PLACE_OPERATION: place_live,
        PRESS_COMPLETION_BUTTON_OPERATION: press_finish,
    }


__all__ = [
    "LIVE_SUGAR_PLACE_OPERATION",
    "PRESS_COMPLETION_BUTTON_OPERATION",
    "build_stage_handlers",
]
