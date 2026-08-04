"""Target sampler — picks a random target pose for an operation.

For pre-defined targets (button press poses, fixed place locations, etc.).
Dynamic targets (grasp poses based on object position) are still handled
by the ManipulationController internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bench.execution.navigation.contracts import Pose
from bench.execution.sampling.candidate import Candidate
from bench.execution.sampling.selector import select_candidate

if TYPE_CHECKING:
    from bench.execution.sampling.rng import SeedStream


@dataclass
class SimpleTargetSampler:
    """Sample a target pose from pre-defined candidates.

    Targets are organized as ``{operation: {target_name: [Pose, ...]}}``.

    Usage::

        targets = {
            "press": {
                "button_A": [Pose.xyz(0.4, -0.1, 0.8)],
            },
            "place": {
                "dispenser": [Pose.xyz(0.35, -0.05, 0.75), Pose.xyz(0.36, -0.06, 0.75)],
            },
        }
        seeds = SeedStream(base_seed=42)
        sampler = SimpleTargetSampler(targets=targets, seeds=seeds)

        pose = sampler.sample("press", "button_A")
    """

    targets: dict[str, dict[str, list[Pose]]]
    """Pre-defined target poses: ``{operation: {target_name: [poses]}}``."""

    seeds: "SeedStream | None" = None
    """Optional seed stream for reproducible sampling."""

    mode: str = "random_valid"
    """Selection mode passed to ``select_candidate``."""

    fallback_index: int = 0

    _choices: dict[tuple[str, str], Pose] = field(
        default_factory=dict, init=False, repr=False,
    )

    def sample(self, operation: str, target: str) -> Pose:
        """Pick a random pose for *(operation, target)*.

        Raises ``KeyError`` when no pre-defined poses are registered for this
        combination — callers should catch this and fall back to dynamic
        pose generation.
        """
        key = (operation, target)
        if key in self._choices:
            return self._choices[key]

        op_targets = self.targets.get(operation, {})
        poses = op_targets.get(target, [])
        if not poses:
            raise KeyError(
                f"no pre-defined poses for operation={operation!r} target={target!r}"
            )

        candidates = [
            Candidate(
                value=pose,
                id=f"{operation}_{target}_{i}",
                score=0.0,
                metadata={"operation": operation, "target": target, "index": i},
            )
            for i, pose in enumerate(poses)
        ]

        rng = self.seeds.rng("target", operation, target) if self.seeds else None
        selection = select_candidate(candidates, mode=self.mode, rng=rng)

        chosen = (
            selection.selected.value
            if selection.passed and selection.selected is not None
            else candidates[self.fallback_index].value
        )
        self._choices[key] = chosen
        return chosen

    def reset_choices(self) -> None:
        """Clear the choice cache."""
        self._choices.clear()
