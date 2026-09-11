# WA-01

This package directly owns WA-01's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Package layout version: `membench.task_package/v1`.

## Evaluation boundary

- `environment.py` detects physical facts and shared task events. It does not
  decide whether an evaluation stage is reportable.
- `evaluation.py` owns the ordered stage contract, historical qualification,
  final success, failure codes, and task diagnostics.
- `runtime.py` creates one evaluator per episode and routes the production
  success and diagnostics callbacks through it. Expert execution is not part
  of this path.

The reporting funnel is:

`sugar -> one_ready -> ready -> selection -> success`

`selection` records a valid final re-grasp regardless of which cup was chosen.
`success` is derived from that selection result and is qualified only when the
selected cup is the one that actually received the sugar; it does not require a
second action after selection. The public reporting
funnel exposes the same five events as
`sugar_added -> one_cup_ready -> two_cups_ready -> select -> success`.

Each stage exposes both `observed` and `qualified`. `observed` only means that
the underlying detector saw the condition. A stage becomes `qualified` only
after its prerequisite was already qualified, so stage aggregation must read
`diagnostics.stages.<stage>.qualified`.
