"""TM02 adapter around TM02's task-owned physical water-button primitive."""

from __future__ import annotations

from typing import Any

from bench.tasks.tm02.execution.water_operations import operate_water_button


def operate_tm02_water_button(
    controller: Any,
    *,
    mode: str,
    **settings: Any,
) -> dict[str, Any]:
    """Use TM02's task-owned blue/cold button press trajectory for A1 and A2."""

    values = dict(settings)
    values.pop("temperature", None)
    return operate_water_button(
        controller,
        mode=str(mode),
        temperature="cold",
        **values,
    )


__all__ = ["operate_tm02_water_button"]
