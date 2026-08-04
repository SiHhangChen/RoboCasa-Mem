"""Pure WA-02 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, requested_temperature: str, variant: str) -> PromptSpec:
    temperature = str(requested_temperature).strip().lower()
    if temperature not in {"hot", "cold"}:
        raise ValueError("WA-02 requested_temperature must be hot or cold")
    return PromptSpec(
        initial_prompt=(
            "Remember the temperature bound to each identical cup. After staging, "
            f"the final view does not reveal which cup contains {temperature} water."
        ),
        policy_prompt=(
            "Fill two identical cups with hot and cold water, stage both cups, then "
            f"take the {temperature}-water cup."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
