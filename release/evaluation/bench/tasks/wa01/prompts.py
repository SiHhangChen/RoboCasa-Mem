"""Pure WA-01 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Two identical coffees are already prepared. Add one sugar cube to "
            "either coffee, place both cups on the dining counter, then identify "
            "the sugared coffee."
        ),
        policy_prompt=(
            "Start at the coffee machine with two identical prepared coffees. Add "
            "one sugar cube to either coffee, carry both coffees to the dining "
            "counter, and identify the coffee you sugared."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
