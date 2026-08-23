"""Physical WA-05 temperature-answer button primitive."""

from __future__ import annotations
from typing import Any
import numpy as np
from bench.tasks.wa05.execution.geometry import button_press_pose_world
from bench.tasks.wa05.execution.water_operations import _run_world_stage

def press_temperature_answer_button(controller: Any, *, temperature: str, prepress_z_offset: float, press_z_offset: float, hold_steps: int, gripper_action: float, near_miss_position_tolerance: float, require_physical_button_motion: bool = True, **_unused: Any) -> dict[str, Any]:
    temp = str(temperature).strip().lower()
    if temp not in {"hot", "cold"}:
        return {"primitive": "press_temperature_answer_button", "pass": False, "reason": "invalid_temperature"}
    if not bool(controller.runtime_config.execute):
        return {"primitive": "press_temperature_answer_button", "temperature": temp, "dry_run": True, "pass": True}
    env = controller.env
    controller.refresh_obs()
    button = np.asarray(env.answer_button_position(temp), dtype=float)
    before = float(env.answer_button_press_depth(temp))
    prepress_pose = button_press_pose_world(controller.obs, button_position_world=button, z_offset=float(prepress_z_offset))
    press_pose = button_press_pose_world(controller.obs, button_position_world=button, z_offset=float(press_z_offset))
    prepress = _run_world_stage(controller, prepress_pose, stage="answer_button_prepress", gripper_action=float(gripper_action))
    press = _run_world_stage(controller, press_pose, stage="answer_button_press", gripper_action=float(gripper_action), record_trajectory=True) if bool(prepress.get("pass", False)) else {"pass": False}
    eef = np.asarray(controller.stage_executor.current_eef_site_world(), dtype=float)
    endpoint_error = float(np.linalg.norm(eef[:3, 3] - press_pose[:3, 3]))
    hold = controller.run_gripper_command(float(gripper_action), int(hold_steps)) if bool(press.get("pass", False)) else {"pass": False}
    after = float(env.answer_button_press_depth(temp))
    env.update_state()
    retreat = _run_world_stage(controller, prepress_pose, stage="answer_button_retreat", gripper_action=float(gripper_action)) if bool(press.get("pass", False)) else {"pass": False}
    # The spring can rebound before the post-hold sample. A semantic answer is
    # still physical evidence because only the qpos edge observer can create it.
    edge_latched = env._semantic_state.pressed_temperature == temp
    physical_motion = max(before, after) >= float(env.button_pressed_threshold_m) or edge_latched
    checks = {"prepress": bool(prepress.get("pass", False)), "press": bool(press.get("pass", False)), "press_endpoint": endpoint_error <= float(near_miss_position_tolerance), "hold": bool(hold.get("pass", True)), "physical_button_motion": physical_motion if require_physical_button_motion else True, "answer_event": edge_latched and not env._semantic_state.absorbing_failure, "retreat": bool(retreat.get("pass", False))}
    return {"primitive": "press_temperature_answer_button", "temperature": temp, "button_position_world": button.tolist(), "press_depth_before_m": before, "press_depth_after_m": after, "endpoint_error_m": endpoint_error, "checks": checks, "pass": all(checks.values()), "reason": None if all(checks.values()) else "answer_button_operation_failed"}

__all__ = ["press_temperature_answer_button"]
