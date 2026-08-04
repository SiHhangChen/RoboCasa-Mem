"""Template task semantics boundary."""

def evaluate_task_state(runtime, state=None):
    del runtime, state
    return {"success": False, "reason": "template_task"}
