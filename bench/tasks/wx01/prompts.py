"""Pure WX-01 policy-visible prompt resolution."""

from __future__ import annotations

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    if variant != "memory":
        raise ValueError("WX-01 currently supports variant='memory' only")
    return PromptSpec(
        initial_prompt=(
            "The cup has been moved away from its remembered dining-table spot. "
            "Clean that spot and return the cup to it."
        ),
        policy_prompt=(
            "Start with the sponge already grasped and the cup displaced. "
            "Wipe the remembered cup spot, put down the sponge, then return "
            "the cup to that original spot."
        ),
        variant=variant,
    )


__all__ = ["resolve_prompt_spec"]
