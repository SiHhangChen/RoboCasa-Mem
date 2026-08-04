"""Pure TS04 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Inspect each of the two drawers exactly once, close it after "
            "inspection, remember which drawer has already been checked, then "
            "press the completion button."
        ),
        policy_prompt=(
            "Please inspect both drawers, close each one, then press the "
            "completion button."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
