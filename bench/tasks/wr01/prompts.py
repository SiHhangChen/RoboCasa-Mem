"""Pure WR-01 policy-visible prompt resolution."""

from __future__ import annotations

from bench.tasks.contracts import PromptSpec
from bench.tasks.wr01.domain.episode import validate_item_id


def resolve_prompt_spec(*, target_item_id: str, variant: str) -> PromptSpec:
    item_id = validate_item_id(target_item_id)
    color = "red" if item_id == "red_can" else "blue"
    if variant not in {"memory", "oracle_reveal"}:
        raise ValueError("WR-01 variant must be 'memory' or 'oracle_reveal'")
    return PromptSpec(
        initial_prompt=(
            "Store both colored cans and close both cabinet doors. Then retrieve "
            f"the {color} can."
        ),
        policy_prompt=(
            "Store the red and blue cans in the two open cabinets. After both doors "
            f"are closed, retrieve the {color} can."
        ),
        variant=variant,
    )


__all__ = ["resolve_prompt_spec"]
