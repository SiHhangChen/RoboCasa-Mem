"""Pure WR-03 policy-visible prompt resolution."""

from bench.tasks.contracts import PromptSpec
from bench.tasks.wr03.domain.episode import public_memory_prompt


def resolve_prompt_spec(*, variant: str) -> PromptSpec:
    prompt = public_memory_prompt()
    return PromptSpec(prompt, prompt, str(variant))


__all__ = ["resolve_prompt_spec"]
