"""Pure WA-04 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, requested_temperature: str, variant: str) -> PromptSpec:
    temperature = str(requested_temperature).strip().lower()
    if temperature not in {"hot", "cold"}:
        raise ValueError("WA-04 requested_temperature must be hot or cold")
    return PromptSpec(
        initial_prompt=(
            "Remember the temperature bound to each of the three identical cups. "
            f"Exactly one cup contains {temperature} water, and the final view does not reveal which one."
        ),
        policy_prompt=(
            f"Fill three identical cups so exactly one contains {temperature} water "
            f"and the other two contain {'cold' if temperature == 'hot' else 'hot'} water, "
            f"stage all three cups, then take the {temperature}-water cup."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
