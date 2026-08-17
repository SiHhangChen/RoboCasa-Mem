"""Pure TS04 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Exactly one of the two drawers contains a bright red cube. "
            "Search without revisiting drawers and close each inspected drawer. "
            "As soon as the cube is found and its drawer is closed, stop opening "
            "drawers and press the completion button once."
        ),
        policy_prompt=(
            "Search the drawers for the red cube, closing each drawer after "
            "inspection. Stop after finding and closing its drawer, then press "
            "the completion button once."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
