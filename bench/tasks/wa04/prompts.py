"""Pure WA-04 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec


def resolve_prompt_spec(*, requested_temperature: str, variant: str) -> PromptSpec:
    temperature = str(requested_temperature).strip().lower()
    if temperature not in {"hot", "cold"}:
        raise ValueError("WA-04 requested_temperature must be hot or cold")
    return PromptSpec(
        initial_prompt=(
            "Remember which of the three identical cups is filled with "
            f"{temperature} water. The final view does not reveal which one."
        ),
        policy_prompt=(
            f"We need two cups of {'cold' if temperature == 'hot' else 'hot'} water "
            f"and one cup of {temperature} water. "
            "The left faucet dispenses hot water and the right faucet dispenses cold water. "
            "You may fill hot or cold water in any order. "
            "Finally place all three cups to the left of the sink, "
            f"then pick up the cup filled with {temperature} water."
        ),
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
