"""Pure WR-05 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec
from bench.tasks.wr05.domain.episode import public_task_prompt


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    prompt = public_task_prompt()
    return PromptSpec(
        initial_prompt=prompt,
        policy_prompt=prompt,
        variant=str(variant),
    )


__all__ = ["resolve_prompt_spec"]
