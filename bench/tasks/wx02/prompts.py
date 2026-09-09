"""Pure WX-02 policy-visible prompt resolution."""

from __future__ import annotations

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    if variant != "memory":
        raise ValueError("WX-02 currently supports variant='memory' only")
    return PromptSpec(
        initial_prompt=(
            "Two cups are each covering a stain on the dining table. Fetch the "
            "sponge from the yellow region and stage it near the cups. Move the "
            "first cup, wipe its revealed stain, and return that cup to its exact "
            "starting position. Repeat for the second cup, then return the sponge "
            "to the yellow region."
        ),
        policy_prompt=(
            "Start in front of both cups and remember where each one is. Fetch "
            "the sponge from the yellow region, bring it back, and set it down. "
            "For each cup in order, move it aside, wipe the stain underneath, put "
            "down the sponge, and return that cup to where it started. Finally, "
            "pick up the sponge and return it to the yellow region."
        ),
        variant=variant,
    )


__all__ = ["resolve_prompt_spec"]
