"""Pure TS04 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Exactly one of the two drawers contains a bright red cube. "
            "Search without revisiting drawers and close each inspected drawer. "
            "As soon as the cube is found and its drawer is closed, stop opening "
            "drawers. If finding it required inspecting k drawers, press the "
            "completion button exactly k times, releasing it between successive "
            "presses."
        ),
        policy_prompt=(
            "Search the drawers for the red cube, closing each drawer after "
            "inspection. Stop after finding and closing its drawer, then press "
            "the completion button once for each drawer you inspected."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
