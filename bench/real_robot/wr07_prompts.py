"""Prompt choices used by the two real-robot WR07 Pi0.5 policies."""

FULL_CONFIG = "pi05_real_robot_wr07_v3"
SUBGOAL_CONFIG = "pi05_real_robot_wr07_v3_subgoal"

SUBGOAL_PROMPTS = {
    color: (
        f"Move the movable item from the {color} source container "
        f"to the matching {color} container on the opposite side."
    )
    for color in ("brown", "white", "pink")
}


def prompt_policy(config_name: str, *, default_prompt: str | None = None) -> dict[str, object]:
    if config_name == FULL_CONFIG:
        if not default_prompt:
            raise ValueError(f"{FULL_CONFIG} requires a training default prompt")
        return {"prompt_mode": "fixed", "default_prompt": default_prompt}
    if config_name == SUBGOAL_CONFIG:
        return {"prompt_mode": "episode_choice", "prompt_choices": dict(SUBGOAL_PROMPTS)}
    return {}
