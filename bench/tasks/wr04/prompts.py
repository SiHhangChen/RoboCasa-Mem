"""Pure WR-04 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec
from bench.tasks.wr04.domain.episode import public_initial_prompt, public_return_prompt


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(public_initial_prompt(), public_return_prompt(), str(variant))


__all__ = ["resolve_prompt_spec"]
