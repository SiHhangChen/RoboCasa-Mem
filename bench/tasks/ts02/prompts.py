"""Pure TS02 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Inspect each cabinet exactly once, close it after inspection, and "
            "remember which cabinets have already been checked. After all three "
            "are closed, press the completion button."
        ),
        policy_prompt=(
            "Please inspect the three cabinets, close each drawer, then press "
            "the completion button."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
