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
            f"Retrieve the {color} can. Observe both open cabinets, close both doors "
            "in any order, return to the neutral position, then open the cabinet "
            f"containing the {color} can and bring it back to the table."
        ),
        policy_prompt=(
            f"Retrieve the {color} can. Close both open cabinet doors in any order, "
            "return to the neutral position, then open the cabinet containing the "
            f"{color} can and bring it back to the table."
        ),
        variant=variant,
    )


__all__ = ["resolve_prompt_spec"]
