"""Pure WR-02 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec
from bench.tasks.wr02.domain.episode import public_return_prompt


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Observe the open shelf and remember the sponge-to-soap level binding. "
            "The return phase will not reveal their original levels."
        ),
        policy_prompt=public_return_prompt(),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
