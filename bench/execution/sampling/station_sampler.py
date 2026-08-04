"""Station sampler — picks a random base pose for a named station.

Uses ``select_candidate`` from ``bench.execution.sampling.selector`` for reproducible selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bench.execution.navigation.contracts import Pose, StationSpec
from bench.execution.sampling.candidate import Candidate
from bench.execution.sampling.selector import select_candidate

if TYPE_CHECKING:
    from bench.execution.sampling.rng import SeedStream


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
    """Cache of choices made this episode (so repeated calls return the same pose)."""

    def sample(self, station_name: str, *, purpose: str | None = None) -> Pose:
        """Pick a random base pose for *station_name*.

        The choice is cached — calling ``sample()`` again with the same
        station name returns the same pose (episode-stable).
        """
        del purpose
        if station_name in self._choices:
            return self._choices[station_name]

        spec = self.stations.get(station_name)
        if spec is None:
            raise KeyError(f"station {station_name!r} not found in sampler config")

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
