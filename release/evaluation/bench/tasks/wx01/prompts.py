"""Pure WX-01 policy-visible prompt resolution."""

from __future__ import annotations

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    if variant != "memory":
        raise ValueError("WX-01 currently supports variant='memory' only")
    return PromptSpec(
        initial_prompt=(
            "The dining-table spot is dirty. Wipe it clean with the sponge, "
            "then place the cup onto the cleaned spot."
        ),
        policy_prompt=(
            "Start with the sponge already grasped. Wipe the dirty dining-table "
            "spot, then place the cup onto the cleaned spot."
        ),
        variant=variant,
    )


__all__ = ["resolve_prompt_spec"]
