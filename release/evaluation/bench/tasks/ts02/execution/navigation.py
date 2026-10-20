"""TS02 fixture stations and mobile-base controller assembly."""
from __future__ import annotations
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from bench.execution.planning import NavigationConfig

def build_navigation_config(settings: dict[str, Any], *, manual_base_path_file: Path, station_base_pose_file: Path) -> NavigationConfig:
    """Convert YAML settings to the generic navigation dataclass."""
    values = dict(settings)
    allowed = {field.name for field in fields(NavigationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError('unknown TS02 navigation settings: ' + ', '.join(unknown))
    values['station_base_pose_file'] = Path(station_base_pose_file)
    values['manual_base_path_file'] = Path(manual_base_path_file)
    return replace(NavigationConfig(), **values)
