"""Generic candidate selection algorithms."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np

from bench.execution.sampling.candidate import Candidate, CandidateSelection, CandidateValidation


T = TypeVar("T")


class SelectionMode(str, Enum):
    FIRST_VALID = "first_valid"
    RANDOM_VALID = "random_valid"
    BEST_SCORE = "best_score"
    LOWEST_COST = "lowest_cost"


def valid_candidates(validations: Iterable[CandidateValidation[T]]) -> list[Candidate[T]]:
    """Return candidates whose validators passed."""
    return [item.candidate for item in validations if bool(item.valid)]


def _score(candidate: Candidate[T]) -> float | None:
    if candidate.score is not None:
        return float(candidate.score)
    if "score" in candidate.metadata:
        return float(candidate.metadata["score"])
    if "cost" in candidate.metadata:
        return -float(candidate.metadata["cost"])
    return None


def _cost(candidate: Candidate[T]) -> float | None:
    if "cost" in candidate.metadata:
        return float(candidate.metadata["cost"])
    if candidate.score is not None:
        return -float(candidate.score)
    if "score" in candidate.metadata:
        return -float(candidate.metadata["score"])
    return None


def select_candidate(
    candidates: Sequence[Candidate[T]],
    *,
    mode: SelectionMode | str = SelectionMode.FIRST_VALID,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    key: Callable[[Candidate[T]], float] | None = None,
) -> CandidateSelection[T]:
    """Choose one candidate from an already-valid candidate list.

    Validation timing is task-controlled. This function intentionally only
    implements the reusable selection policy.
    """
    mode_value = str(mode.value if isinstance(mode, SelectionMode) else mode)
    items = list(candidates)
    if not items:
        return CandidateSelection(
            selected=None,
            selected_index=None,
            mode=mode_value,
            candidate_count=0,
            valid_count=0,
            seed=seed,
            reason="no_valid_candidates",
            candidates=[],
        )

    if mode_value in {"first", "first_valid"}:
        index = 0
        reason = "first_valid_candidate"
    elif mode_value in {"random", "random_valid"}:
        local_rng = rng if rng is not None else np.random.default_rng(seed)
        index = int(local_rng.integers(0, len(items)))
        reason = "random_valid_candidate"
    elif mode_value in {"best", "best_score"}:
        score_key = key or (lambda candidate: _score(candidate) if _score(candidate) is not None else -np.inf)
        index = max(range(len(items)), key=lambda i: float(score_key(items[i])))
        reason = "highest_score_candidate"
    elif mode_value in {"lowest_cost", "min_cost"}:
        cost_key = key or (lambda candidate: _cost(candidate) if _cost(candidate) is not None else np.inf)
        index = min(range(len(items)), key=lambda i: float(cost_key(items[i])))
        reason = "lowest_cost_candidate"
    else:
        raise ValueError(f"unsupported selection mode: {mode_value!r}")

    return CandidateSelection(
        selected=items[index],
        selected_index=index,
        mode=mode_value,
        candidate_count=len(items),
        valid_count=len(items),
        seed=seed,
        reason=reason,
        candidates=items,
    )
