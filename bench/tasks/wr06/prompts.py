"""Pure WR-06 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec
from bench.tasks.wr06.domain.episode import public_initial_prompt, public_return_prompt


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=public_initial_prompt(),
        policy_prompt=public_return_prompt(),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
