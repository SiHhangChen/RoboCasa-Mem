"""Pure TS01 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, target_sugar_count: int, variant: str) -> PromptSpec:
    count = int(target_sugar_count)
    if count <= 0:
        raise ValueError("TS01 target_sugar_count must be positive")
    noun = "sugar cube" if count == 1 else "sugar cubes"
    return PromptSpec(
        initial_prompt=(
            f"This hot chocolate needs exactly {count} {noun}. Remember this "
            "count; it will not be repeated."
        ),
        policy_prompt=(
            f"Add exactly {count} {noun} to the hot chocolate, then press the "
            "completion button to finish."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
