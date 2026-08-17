"""Pure WR-05 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec
from bench.tasks.wr05.domain.episode import public_return_prompt


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Observe the open shelf and remember the ketchup-to-mustard level "
            "binding. The return phase will not reveal their original levels."
        ),
        policy_prompt=public_return_prompt(),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
