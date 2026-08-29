# Stage Evaluation

`bench.evaluation.stage_rollout.evaluate_stage` implements the stage-conditioned
protocol used for memory-demand evaluation. A task exposes:

- `instruction` and ordered `stage_specs`;
- `reset(seed)`, `observe()` (or `get_observation()`), and `step(action)`;
- `get_privileged_state()` for evaluator-only predicates;
- an optional `close()`.

Each `StageSpec` contains a completion predicate, an optional failure and entry
predicate, a model-control budget, and an optional consecutive-success window.
The model must expose only `reset_episode(instruction)` and
`predict(observation)`. The expert must expose `predict(privileged_state)`.

For a target stage after the first one, the evaluator runs the expert prefix
while still invoking the model on each policy observation. Proposed model
actions are discarded and expert actions are executed. Once all prior stage
predicates and the target entry predicate hold, the next observation is the
handoff observation; a newly predicted model action is the first executed model
action. The model is reset once per episode and is never reset at handoff.

Results use `SUCCESS`, `FAILURE`, and `INVALID_PREFIX`. Invalid prefixes are
kept in the attempted count but excluded from StageSR's denominator. Use
`aggregate_stage_results`, `stage_success_rate`, `prefix_validity`, and
`macro_stage_sr` for report generation.

Call `evaluate_stage` with a fresh task/environment for every target stage. In
particular, a C-stage run must replay the expert A and B prefix from a new
episode rather than continuing a model-controlled B run.

The `stages` module re-exports the same API for integrations that prefer
`bench.evaluation.stages`.
