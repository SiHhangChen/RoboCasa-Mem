"""Strict configuration boundary for the TM02 scene and semantic shell."""
from __future__ import annotations
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import yaml
from bench.tasks.tm02.domain.contracts import BUTTON_NAMES, CANONICAL_PROMPT, CANONICAL_PROMPT_SHA256, ENV_NAME, TASK_ID, ActionConfig, ObserverConfig, RunContext, TaskConfig
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_TASK_CONFIG_FILE = PACKAGE_ROOT / 'resources' / 'task.yaml'
SHARED_WA02_RESOURCE_NAMES = ('water_button_positions', 'water_station_poses')
CALIBRATION_RESOURCE_NAMES = ('cabinet_station_poses', 'manual_base_path', 'cabinet_door_pose', 'can_grasp_pose', 'can_retrieval_prepose', 'can_place_pose')

class TM02CalibrationError(RuntimeError):
    """Raised when execution is requested before task-specific calibration."""

def load_task_config(path: str | Path=DEFAULT_TASK_CONFIG_FILE) -> TaskConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, Mapping):
        raise ValueError('TM02 config root must be a mapping')
    allowed_sections = {'schema_version', 'task', 'scene', 'episode', 'initial_state', 'semantics', 'sensors', 'resources', 'navigation', 'manipulation', 'operations', 'recording'}
    _reject_unknown(payload, allowed_sections, 'config sections')
    if int(payload.get('schema_version', 1)) != 1:
        raise ValueError('TM02 only supports schema_version=1')
    sections = {name: dict(_mapping(payload, name)) for name in allowed_sections if name != 'schema_version'}
    task = sections['task']
    if str(task.get('id')) != TASK_ID:
        raise ValueError(f'TM02 config must declare task.id={TASK_ID!r}')
    prompt = str(task.get('canonical_prompt', ''))
    declared_hash = str(task.get('canonical_prompt_sha256', ''))
    actual_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if prompt != CANONICAL_PROMPT:
        raise ValueError('TM02 canonical_prompt differs from the frozen contract')
    if declared_hash != CANONICAL_PROMPT_SHA256 or actual_hash != declared_hash:
        raise ValueError('TM02 canonical_prompt_sha256 mismatch')
    scene = sections['scene']
    if str(scene.get('env_name')) != ENV_NAME:
        raise ValueError(f'TM02 config must declare scene.env_name={ENV_NAME!r}')
    if (int(scene.get('layout_id', -1)), int(scene.get('style_id', -1))) != (15, 34):
        raise ValueError('TM02 must reuse WA02 layout 15 / style 34')
    if str(scene.get('water_control_source')) != 'WA02':
        raise ValueError('TM02 water control must declare the frozen WA02 calibration source')
    if tuple(scene.get('task_added_button_names', ())) != BUTTON_NAMES:
        raise ValueError(f'TM02 task buttons must equal {BUTTON_NAMES}')
    if str(scene.get('water_button_visual_role')) != 'blue':
        raise ValueError('TM02 water button must retain the blue WA02 visual role')
    if str(scene.get('completion_button_visual_role')) != 'red':
        raise ValueError("TM02 completion button must reuse WA02's red visual role")
    if str(scene.get('completion_button_position_source')) != 'water_button_left':
        raise ValueError('TM02 completion button must remain left of the water button')
    offset = tuple((float(value) for value in scene.get('completion_button_offset_from_water_xyz', ())))
    if offset != (0.0, 0.14, 0.0):
        raise ValueError('TM02 completion button offset must be (0, 0.14, 0)')
    water_xyz = tuple((float(value) for value in scene.get('water_button_world_xyz', ())))
    completion_xyz = tuple((float(value) for value in scene.get('completion_button_world_xyz', ())))
    if len(water_xyz) != 3 or len(completion_xyz) != 3 or any((abs(completion_xyz[index] - water_xyz[index] - offset[index]) > 1e-09 for index in range(3))):
        raise ValueError('TM02 completion button position must equal water button + offset')
    if str(scene.get('cabinet_asset_source')) != 'WR01.WR01SlideStorageCabinet':
        raise ValueError('TM02 cabinet source identity is invalid')
    if str(scene.get('can_asset_source')) != 'WR01.RED_CAN_XML':
        raise ValueError('TM02 can source identity is invalid')
    if str(scene.get('can_xml')) != 'objects/objaverse/canned_food/canned_food_15/model.xml':
        raise ValueError("TM02 can_xml must identify WR01's red can")
    config = TaskConfig(source_path=config_path, **sections)
    for name in SHARED_WA02_RESOURCE_NAMES:
        resource = config.path(config.resources.get(name))
        if resource is None or not resource.is_file():
            raise FileNotFoundError(f'missing shared WA02 resource {name!r}: {resource}')
    return config

def resolve_observer_config(config: TaskConfig) -> ObserverConfig:
    values = config.semantics
    result = ObserverConfig(cabinet_open_threshold=float(values['cabinet_open_threshold']), cabinet_closed_threshold=float(values['cabinet_closed_threshold']), button_pressed_threshold_m=float(values['button_pressed_threshold_m']), button_released_threshold_m=float(values['button_released_threshold_m']), button_release_dwell_steps=int(values['button_release_dwell_steps']), can_place_xy_tolerance_m=float(values['can_place_xy_tolerance_m']), pot_water_xy_threshold_m=float(values['pot_water_xy_threshold_m']), pot_translation_guard_m=float(values['pot_translation_guard_m']), pot_tilt_guard_deg=float(values['pot_tilt_guard_deg']), pot_fill_required_steps=int(values['pot_fill_required_steps']), pot_filled_threshold=float(values['pot_filled_threshold']))
    if not 0.0 <= result.cabinet_closed_threshold < result.cabinet_open_threshold <= 1.0:
        raise ValueError('TM02 cabinet thresholds are invalid')
    if not 0.0 < result.button_released_threshold_m < result.button_pressed_threshold_m:
        raise ValueError('TM02 button hysteresis is invalid')
    if result.button_release_dwell_steps <= 0:
        raise ValueError('TM02 button_release_dwell_steps must be positive')
    for name in ('can_place_xy_tolerance_m', 'pot_water_xy_threshold_m', 'pot_translation_guard_m', 'pot_tilt_guard_deg'):
        if float(getattr(result, name)) <= 0.0:
            raise ValueError(f'TM02 {name} must be positive')
    if result.pot_fill_required_steps <= 0:
        raise ValueError('TM02 pot_fill_required_steps must be positive')
    if not 0.0 < result.pot_filled_threshold <= 1.0:
        raise ValueError('TM02 pot_filled_threshold must be within (0, 1]')
    return result

def resolve_action_config(config: TaskConfig) -> ActionConfig:
    operations = config.operations
    can_pick = dict(_mapping(operations, 'can_pick'))
    if 'disable_collisions_during_transport' in can_pick:
        raise ValueError('TM02 is action-only; disable_collisions_during_transport is forbidden')
    water = dict(_mapping(operations, 'water_button'))
    water.update({'temperature': 'cold', 'control_mode': 'buttons', 'water_button_positions_file': str(_resource(config, 'water_button_positions'))})
    return ActionConfig(water_button_metadata=water, completion_button_metadata=dict(_mapping(operations, 'completion_button')), cabinet_open_metadata=dict(_mapping(operations, 'cabinet_open')), can_pick_metadata=can_pick, can_place_metadata=dict(_mapping(operations, 'can_place')), cabinet_close_metadata=dict(_mapping(operations, 'cabinet_close')), return_arm_metadata=dict(_mapping(operations, 'return_arm')))

def resolve_run_context(*, split: str, env_split: str, variant: str, execute: bool, extra_metadata: Mapping[str, Any] | None=None) -> RunContext:
    return RunContext(split=str(split), env_split=str(env_split), variant=str(variant), execute=bool(execute), extra_metadata=dict(extra_metadata or {}))

def resolve_task_options(task_options: Mapping[str, Any] | None) -> dict[str, Any]:
    if task_options:
        raise ValueError('TM02 route and prompt are seed/YAML-owned')
    return {}

def missing_calibration_resources(config: TaskConfig) -> tuple[str, ...]:
    missing: list[str] = []
    for name in CALIBRATION_RESOURCE_NAMES:
        path = config.path(config.resources.get(name))
        if path is None or not path.is_file():
            missing.append(name)
    return tuple(missing)

def require_production_calibration(config: TaskConfig) -> None:
    missing = missing_calibration_resources(config)
    if missing:
        raise TM02CalibrationError('TM02 physical calibration is incomplete: ' + ', '.join(missing))

def _resource(config: TaskConfig, name: str) -> Path:
    path = config.path(config.resources.get(name))
    if path is None or not path.is_file():
        raise FileNotFoundError(f'missing TM02 resource {name!r}: {path}')
    return path

def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f'TM02 {name} must be a mapping')
    return value

def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f'unknown TM02 {label}: ' + ', '.join(unknown))
__all__ = ['CALIBRATION_RESOURCE_NAMES', 'DEFAULT_TASK_CONFIG_FILE', 'SHARED_WA02_RESOURCE_NAMES', 'TM02CalibrationError', 'load_task_config', 'missing_calibration_resources', 'require_production_calibration', 'resolve_action_config', 'resolve_observer_config', 'resolve_run_context', 'resolve_task_options']
