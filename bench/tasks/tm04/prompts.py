"""Pure TM04 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec

from .domain.contracts import CANONICAL_PROMPT


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    if not str(variant).strip():
        raise ValueError("TM04 variant must be non-empty")
    return PromptSpec(CANONICAL_PROMPT, CANONICAL_PROMPT, str(variant))


__all__ = ["resolve_prompt_spec"]
