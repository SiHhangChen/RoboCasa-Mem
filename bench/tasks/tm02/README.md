# TM02

This package directly owns TM02's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

## Evaluation: report-v2

The five cumulative milestones are `faucet_on`, `pot_filled_and_can_retrieved`,
`one_thread_completed`, `both_threads_completed`, and `success`.

`one_thread_completed` means A2 (water off) **or** B2 (cabinet closed) was legally
committed. The first completed thread passes S3 and the second passes S4, in either
order. Both remain passed once complete. Event timestamps use `min(A2, B2)` for S3
and `max(A2, B2)` for S4, not the time a report happens to read the state.
`diagnostics.thread_completion` includes a count and independent `completed` /
`completed_step` entries for A2 and B2.

This is a reporting change, not a task-rule change: B1 still combines filling and
can retrieval; all physical checks, legal routes, expert actions, final success,
and stopping behavior are unchanged. Explicit `TM02/report-v1` reports remain
readable, but v1/v2 report statistics must not be mixed.

## Stage-conditioned evaluation

The default integration and GR00T launchers use the five-stage v2 suite. The policy
takes over after B1 for S3, after either A2 or B2 for S4, and after both for success.
A v1 prefix without the new S3 boundary cannot be used for the v2 S4 handoff.
Explicitly selected four-stage v1 suites retain their old integration and budgets.
The B1 diagnostic integration is unchanged.
Default full-rollout and stage-run ids also carry the new version, so new results
do not resume or overwrite an old four-stage run. Explicit run-id overrides are
still supported.

The v2 configs use `outputs/stage_prefixes/tm02_gr00t_r40_v2`, separate from v1.
To derive them from existing successful v1 artifacts:

```bash
.venv-robocasa/bin/python scripts/eval/upgrade_tm02_stage_prefixes.py --source-root outputs/stage_prefixes/tm02_gr00t_r40_v1 --stage-suite configs/eval/stages/tm02_gr00t_r40_formal_v2.yaml
```

Migration validates complete semantic event evidence and matching old action
boundaries, preserves every action and its hash, and refuses existing destinations.
It records the derivation in provenance and does not claim to have physically
replayed the trajectory. If evidence is missing, rebuild/replay the prefix instead
of guessing the missing boundary. The normal expert-prefix builder can also capture
fresh v2 artifacts without changing the expert's action plan.

Formal budgets are twice each interval's P95, using the existing prefix-statistics
tool: `220, 3144, 1644, 1594, 868` steps for S1 through S5. These are separate
stage-conditioned trials, not a larger full-rollout horizon. To reproduce the
budget calculation in a separate output file:

```bash
.venv-robocasa/bin/python scripts/eval/freeze_stage_suite.py --stage-suite configs/eval/stages/tm02_gr00t_r40_pilot_v2.yaml --output /tmp/tm02_formal_v2.yaml --stage-suite-version tm02_gr00t_stage_r40_formal_v2
```

Package layout version: `membench.task_package/v1`.
