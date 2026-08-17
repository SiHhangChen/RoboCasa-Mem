"""Pure TS02 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    return PromptSpec(
        initial_prompt=(
            "Exactly one of the three drawers contains a bright red cube. Search "
            "without revisiting drawers, and close each inspected drawer. Stop "
            "after finding and closing the target drawer. If finding it required "
            "inspecting k drawers, press the completion button exactly k times, "
            "releasing it between successive presses."
        ),
        policy_prompt=(
            "Search the three drawers for the red cube. Close every drawer after "
            "looking inside. Once you find the cube and close its drawer, press "
            "the completion button once for each drawer you inspected."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
