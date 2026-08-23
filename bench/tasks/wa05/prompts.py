"""Pure WA-05 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Choose either hot or cold water for the cup. Remember that choice after "
            "placing the cup in the service area."
        ),
        policy_prompt=(
            "Fill the cup with either hot or cold water, place it in the service area, "
            "then press red for hot water or blue for cold water."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
