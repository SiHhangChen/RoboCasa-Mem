# TM01

This package directly owns TM01's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

## Evaluation: report-v2

The public funnel contains five cumulative milestones:

1. `burner_on`: the physical observer committed A1.
2. `one_drawer_inspected`: one complete open / inspection-dwell / close cycle (B1).
3. `one_thread_completed`: A2 (burner off) **or** B2 (second drawer cycle) completed.
4. `both_threads_completed`: both A2 and B2 completed.
5. `success`: a valid completion-button press and the full final-state verdict.

A1 commits when the native stove-knob joint reaches `on_commit_min_rad` (0.50
rad); it does not require gripper contact, motion evidence, or dwell. A2 still
requires its original contact, motion, and dwell evidence to turn the burner off.

A2 and B2 may occur in either order. S3 remains passed after both complete.
S3 uses the earlier semantic event step; S4 uses the later one, even if reporting
first reads the history after both events. `diagnostics.thread_completion` exposes
the count and each event's `completed` / `completed_step` independently. A missing
event timestamp remains unknown in these diagnostics rather than being invented.

The A1 physical predicate is angle-only; the other predicates, legal routes,
expert plan, failure rules, and stop behavior are unchanged. Completed progress
is historical; final success can still regress if execution is deliberately
continued. Normal evaluation stops on success.
The JSON schema remains `membench.report_stages/v1`, while the task profile is
`TM01/report-v2`. Explicit `TM01/report-v1` reports remain readable, but their
four-stage statistics cannot be aggregated with v2 reports.

Package layout version: `membench.task_package/v1`.
