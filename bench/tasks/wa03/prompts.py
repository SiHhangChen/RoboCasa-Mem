"""Pure WA-03 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, sugar_target_mode: str = "unmarked", variant: str) -> PromptSpec:
    mode = str(sugar_target_mode).strip().lower()
    if mode == "marked":
        return PromptSpec(
            initial_prompt=(
                "Three identical coffees are already prepared. Sugar the temporarily "
                "marked cup only, place all three cups on the dining counter, then identify it."
            ),
            policy_prompt=(
                "Start at the coffee machine with three identical prepared coffees. Add "
                "one sugar cube to the temporarily marked coffee, carry all three coffees "
                "to the dining counter, and identify the sugared coffee."
            ),
            variant=str(variant),
        )
    if mode != "unmarked":
        raise ValueError(f"unsupported WA-03 sugar_target_mode: {mode!r}")
    return PromptSpec(
        initial_prompt=(
            "Three identical coffees are already prepared. Add one sugar cube to any "
            "one coffee, place all three cups on the dining counter, then identify the "
            "coffee you sugared."
        ),
        policy_prompt=(
            "Start at the coffee machine with three identical prepared coffees. Add "
            "one sugar cube to any one coffee, carry all three coffees to the dining "
            "counter, and identify the coffee you sugared."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
