"""Sampling layer — station and target pose randomization.

All core algorithms (Candidate, select_candidate, SeedStream) are
self-contained in this package — no dependency on task implementations.
"""

from bench.execution.sampling.candidate import Candidate, CandidateSelection, CandidateValidation
from bench.execution.sampling.rng import SeedStream, stable_seed
from bench.execution.sampling.selector import SelectionMode, select_candidate, valid_candidates
from bench.execution.sampling.station_sampler import SimpleStationSampler
from bench.execution.sampling.target_sampler import SimpleTargetSampler
from bench.execution.sampling.pose_targets import (
    AnchoredWorldTarget,
    anchored_target_from_operation,
    update_operation_from_target,
    vector3,
)

__all__ = [
    "Candidate",
    "CandidateSelection",
    "CandidateValidation",
    "AnchoredWorldTarget",
    "SeedStream",
    "SelectionMode",
    "SimpleStationSampler",
    "SimpleTargetSampler",
    "anchored_target_from_operation",
    "select_candidate",
    "stable_seed",
    "update_operation_from_target",
    "valid_candidates",
    "vector3",
]
