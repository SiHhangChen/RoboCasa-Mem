"""TS01 station mapping and mobile-base assembly."""
from __future__ import annotations
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from bench.execution.planning import NavigationConfig

def build_navigation_config(settings: dict[str, Any], *, manual_base_path_file: Path | None=None, station_base_pose_file: Path | None=None) -> NavigationConfig:
    values = dict(settings)
    allowed = {field.name for field in fields(NavigationConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError('unknown TS03 navigation settings: ' + ', '.join(unknown))
    values['manual_base_path_file'] = None if manual_base_path_file is None else Path(manual_base_path_file)
    values['station_base_pose_file'] = None if station_base_pose_file is None else Path(station_base_pose_file)
    return replace(NavigationConfig(), **values)
