"""Template ExpertPlan factory."""

from bench.tasks.contracts import ExpertPlanInputs

def build_expert_plan(inputs: ExpertPlanInputs):
    del inputs
    raise NotImplementedError("template task has no expert plan")
