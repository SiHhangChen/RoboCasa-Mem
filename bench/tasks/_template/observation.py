"""Template task-state observation boundary."""

def observe_task_state(runtime):
    return dict(getattr(runtime, "observation", {}) or {})
