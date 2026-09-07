"""Station sampler — picks a random base pose for a named station.

Uses ``select_candidate`` from ``bench.execution.sampling.selector`` for reproducible selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.sampling.candidate import Candidate
from bench.execution.sampling.selector import select_candidate
from bench.execution.sampling.circle_sampling import sample_uniform_disk, sample_uniform_yaw

if TYPE_CHECKING:
    from bench.execution.sampling.rng import SeedStream


def _yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


@dataclass
class SimpleStationSampler:
    """Sample a base pose from a station's pre-defined candidates.

    Usage::

        from bench.execution.sampling.rng import SeedStream

        stations = {
            "station_A": StationSpec(
                name="station_A",
                base_candidates=[Pose.xyz(0.5, -0.2, 0.0), Pose.xyz(0.55, -0.15, 0.0)],
            ),
        }
        seeds = SeedStream(base_seed=42, namespace="episode_0")
        sampler = SimpleStationSampler(stations=stations, seeds=seeds)

        pose = sampler.sample("station_A")  # randomly picks one candidate
    """

    stations: dict[str, StationSpec]
    """Station specs keyed by name."""

    seeds: "SeedStream | None" = None
    """Optional seed stream for reproducible sampling."""

    mode: str = "random_valid"
    """Selection mode passed to ``select_candidate``."""

    fallback_index: int = 0
    """Which candidate to return when selection fails."""

    _choices: dict[str, Pose] = field(default_factory=dict, init=False, repr=False)
    """Cache of choices for fixed-pose stations."""

    _circle_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def sample(self, station_name: str, *, purpose: str | None = None) -> Pose:
        """Pick a base pose, resampling circle stations on every call."""
        del purpose
        spec = self.stations.get(station_name)
        if spec is None:
            raise KeyError(f"station {station_name!r} not found in sampler config")

        circle = spec.metadata.get("circle") if isinstance(spec.metadata, dict) else None
        if isinstance(circle, dict):
            center = np.asarray(circle.get("center_xy", ()), dtype=float).reshape(-1)
            radius = float(circle.get("radius_m", -1.0))
            if center.size < 2 or not np.all(np.isfinite(center[:2])):
                raise ValueError(f"station {station_name!r} has invalid circle center")
            if not np.isfinite(radius) or radius < 0.0:
                raise ValueError(f"station {station_name!r} has invalid circle radius")
            count = int(self._circle_counts.get(station_name, 0))
            self._circle_counts[station_name] = count + 1
            draw_seed = (
                self.seeds.seed("circle", station_name, count)
                if self.seeds is not None
                else int(np.random.SeedSequence().generate_state(1)[0])
            )
            sampled = sample_uniform_disk(
                center[:2], radius, seed=int(draw_seed), salt=0x53544154
            )
            yaw = sample_uniform_yaw(
                float(circle.get("base_yaw", 0.0)),
                float(circle.get("yaw_half_range_rad", 0.0)),
                seed=int(draw_seed),
                salt=0x59415700,
            )
            torso_height = circle.get("torso_height")
            return Pose(
                position=(float(sampled[0]), float(sampled[1]), 0.0),
                quaternion=_yaw_to_quat_xyzw(yaw),
                torso_height=None if torso_height is None else float(torso_height),
            )

        if station_name in self._choices:
            return self._choices[station_name]

        candidates = [
            Candidate(
                value=pose,
                id=f"{station_name}_{i}",
                score=0.0,
                metadata={"station": station_name, "index": i},
            )
            for i, pose in enumerate(spec.base_candidates)
        ]
        if not candidates:
            raise ValueError(f"station {station_name!r} has no base_candidates")

        rng = self.seeds.rng("station", station_name) if self.seeds else None
        selection = select_candidate(candidates, mode=self.mode, rng=rng)

        if selection.passed and selection.selected is not None:
            chosen = selection.selected.value
        else:
            chosen = candidates[self.fallback_index].value

        self._choices[station_name] = chosen
        return chosen

    def reset_choices(self) -> None:
        """Clear the choice cache (e.g. for a new episode)."""
        self._choices.clear()
        self._circle_counts.clear()
