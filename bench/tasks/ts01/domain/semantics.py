"""Authoritative TS01 physical report and final success predicate.

The task asks for an exact number of cubes in the saucepan followed by a
physical completion-button press. Which cube names are preferred is episode
policy metadata, not an additional success condition: the final sugar state is
evaluated from the cubes that are actually inside the receptacle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PhysicalState:
    """Final simulator facts needed by the TS01 predicate.

    ``objects_released`` is the combined release check used by the previous
    task.  The optional component fields make diagnostics more useful and let
    callers provide the two checks separately when available.
    """

    sugars_in_saucepan: Mapping[str, bool] = field(default_factory=dict)
    objects_released: bool | None = None
    sugar_objects_released: bool | None = None
    receptacle_released: bool | None = None
    completion_button_pressed: bool = False

    @property
    def released(self) -> bool:
        """Return the effective all-objects-released result."""

        if self.objects_released is not None:
            return bool(self.objects_released)
        sugar_released = (
            True if self.sugar_objects_released is None else bool(self.sugar_objects_released)
        )
        receptacle_released = (
            True if self.receptacle_released is None else bool(self.receptacle_released)
        )
        return bool(sugar_released and receptacle_released)


@dataclass(frozen=True)
class SuccessReport:
    """Derived TS01 counts and the one authoritative success result."""

    target_sugar_count: int
    num_sugar_added: int
    under_added: bool
    over_added: bool
    next_sugar_step: int | str
    exact_target_sugar_count: bool
    no_extra_sugar_added: bool
    sugars_released: bool
    completion_button_pressed: bool
    sugars_in_saucepan: Mapping[str, bool] = field(default_factory=dict)

    @property
    def sugar_goal_complete(self) -> bool:
        """Return whether the hot-chocolate manipulation itself is complete."""

        return bool(
            self.exact_target_sugar_count
            and self.no_extra_sugar_added
            and self.sugars_released
        )

    @property
    def success(self) -> bool:
        """Require the sugar goal followed by a verified completion press."""

        return bool(
            self.sugar_goal_complete
            and self.completion_button_pressed
        )

    def checks(self) -> dict[str, bool]:
        """Return names matching the task definition's success conditions."""

        return {
            "exact_target_sugar_count": bool(self.exact_target_sugar_count),
            "no_extra_sugar_added": bool(self.no_extra_sugar_added),
            "sugars_released": bool(self.sugars_released),
            "completion_button_pressed": bool(self.completion_button_pressed),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic representation."""

        return {
            "target_sugar_count": int(self.target_sugar_count),
            "num_sugar_added": int(self.num_sugar_added),
            "under_added": bool(self.under_added),
            "over_added": bool(self.over_added),
            "next_sugar_step": self.next_sugar_step,
            "exact_target_sugar_count": bool(self.exact_target_sugar_count),
            "no_extra_sugar_added": bool(self.no_extra_sugar_added),
            "sugars_released": bool(self.sugars_released),
            "completion_button_pressed": bool(self.completion_button_pressed),
            "sugar_goal_complete": bool(self.sugar_goal_complete),
            "success": bool(self.success),
            "sugars_in_saucepan": {
                str(name): bool(value)
                for name, value in self.sugars_in_saucepan.items()
            },
        }


def evaluate_success(
    physical: PhysicalState,
    target_sugar_count: int,
) -> SuccessReport:
    """Compute TS01's sole final-state success report.

    The predicate deliberately does not latch an earlier over-add event. The
    sugar goal still follows the current HS-TQ-S-01 behavior, while final TS01
    success additionally requires the verified completion-button event.
    """

    if not isinstance(physical, PhysicalState):
        raise TypeError("physical must be a PhysicalState")
    target = int(target_sugar_count)
    if target < 1:
        raise ValueError("target_sugar_count must be positive")
    states = {
        str(name): bool(value)
        for name, value in dict(physical.sugars_in_saucepan).items()
    }
    added = int(sum(states.values()))
    under = added < target
    over = added > target
    if over:
        next_step: int | str = "overdone"
    elif added == target:
        next_step = "done"
    else:
        next_step = added + 1
    return SuccessReport(
        target_sugar_count=target,
        num_sugar_added=added,
        under_added=under,
        over_added=over,
        next_sugar_step=next_step,
        exact_target_sugar_count=added == target,
        no_extra_sugar_added=not over,
        sugars_released=physical.released,
        completion_button_pressed=bool(physical.completion_button_pressed),
        sugars_in_saucepan=states,
    )


__all__ = ["PhysicalState", "SuccessReport", "evaluate_success"]
