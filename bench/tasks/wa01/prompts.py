"""Pure WA-01 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Two identical coffees are already prepared. Sugar the temporarily "
            "marked cup only, place both cups on the dining counter, then identify it."
        ),
        policy_prompt=(
            "Start at the coffee machine with two identical prepared coffees. Add "
            "one sugar cube to the temporarily marked coffee, carry both coffees "
            "to the dining counter, and identify the sugared coffee."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
