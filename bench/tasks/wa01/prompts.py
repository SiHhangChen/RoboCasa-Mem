"""Pure WA-01 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, sugar_target_mode: str = "unmarked", variant: str) -> PromptSpec:
    mode = str(sugar_target_mode).strip().lower()
    if mode == "marked":
        return PromptSpec(
            initial_prompt=(
                "Two identical coffees are already prepared. Sugar the temporarily "
                "marked cup only, place both cups on the dining counter, then pick it up."
            ),
            policy_prompt=(
                "Start at the coffee machine with two identical prepared coffees. Add "
                "one sugar cube to the temporarily marked coffee, carry both coffees "
                "to the dining counter, and pick up the coffee cup you sugared."
            ),
            variant=str(variant),
        )
    if mode != "unmarked":
        raise ValueError(f"unsupported WA-01 sugar_target_mode: {mode!r}")
    return PromptSpec(
        initial_prompt=(
            "Two identical coffees are already prepared. Add one sugar cube to "
            "either coffee, place both cups on the dining counter, then pick up "
            "the coffee cup you sugared."
        ),
        policy_prompt=(
            "Start at the coffee machine with two identical prepared coffees. Add "
            "one sugar cube to either coffee, carry both coffees to the dining "
            "counter, and pick up the coffee cup you sugared."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
