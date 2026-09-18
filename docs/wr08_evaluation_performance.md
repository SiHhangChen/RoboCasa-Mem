# WR08 evaluation performance

## Scope and invariants

The policy evaluator and `WR-08/report-v3` are unchanged. Evidence is still
observed every environment/control step. Pickup requires the existing closed
gripper/contact/proximity/lift checks; placement requires release and support.
Target identification uses actual unique tray contact, not XY position.
Consecutive-step dwell, wrong-target termination, and current matching support
for both cartons at final success remain unchanged. Expert recording continues
to use its existing physical observer and geometry checks.

## Implementation

- `environment.evaluation_snapshot()` caches one policy snapshot per physical
  state. Reward, runtime diagnostics, and rollout success checks reuse it.
- Cache keys include environment timestep, simulation time, and qpos bytes.
  Normal steps, resets, episode configuration, evaluation resets, and the
  environment's object pose setters explicitly invalidate the cache.
- External model edits, mocap edits, or derived-state/contact updates that do
  not change qpos/time must call `env.invalidate_policy_evidence()` before
  querying evidence. Use `model_changed=True` if geom name/ID mappings changed.
  Invalidation refreshes evidence; it does not reset the evaluator's history.
- Snapshot objects and nested diagnostic data are shared read-only values.
- `ContactIndex` scans `contact[:ncon]` once, builds symmetric ID adjacency,
  and reuses it for gripper, plate, and counter queries. Name-to-ID mappings
  persist until model reset. Collision flags are read afresh per snapshot;
  they are not assumed immutable.
- `get_object_hold_status(contact_checker=...)` reuses one contact query and
  one pair of joint readings at both thresholds. Its default behavior for
  other tasks is unchanged; signed threshold checks are preserved.
- Shared runtime diagnostics no longer evaluate the success fallback when
  diagnostics already contain a success field.

Stage-dependent evidence pruning and lazy diagnostic construction are not
enabled in this first batch. They are unnecessary to obtain the initial gain
and would require additional dependency/schema checks.

## Measurement and equivalence tool

Use saved checkpoint actions without starting a policy server:

```bash
source scripts/tools/activate_sim_env.sh
"$ROBOCASA_PYTHON" scripts/eval/profile_wr08_evaluation.py \
  outputs/eval/wr08-pi05-9999-r40-stability-a \
  --seeds 1053,1075 --mode verify \
  --output outputs/eval-profiles/wr08-verify-example.json
```

`verify` compares uncached legacy evidence and indexed cached evidence at
identical states, then feeds independent evaluators and compares the runtime
evaluator too. Comparison includes diagnostics, stages, counters, and terminal
results. It checks equivalence on the replayed trajectory, not reproduction of
the original independent simulation. Failures stop the tool with a nonzero exit.

For timing, run `--mode legacy` and `--mode optimized` separately, with the same
seeds and `--max-steps`. `legacy` reconstructs the old uncached/contact-scanning
path and the eager success fallback. Keep load and camera settings identical.
Add `--record-video` to encode temporary videos using the existing artifact
writer; these videos are discarded afterward. `--no-cameras` is available for
isolating evaluation/physics and cannot be combined with video recording.

The output path must be new. Source runs are read-only, and no benchmark scores
are published. Initialization and inference are excluded. Function timings are
inclusive and overlap; do not sum them. Verify mode includes legacy comparison
overhead and must not be used as an optimized performance measurement.

## Initial measurement

Seed 1050, first 300 saved actions, three cameras, video encoding, sequential
runs on this machine:

| Metric | Legacy path | Optimized path |
| --- | ---: | ---: |
| Replay duration | 77.50 s | 20.00 s |
| Wall time per step | 258.33 ms | 66.65 ms |
| Snapshot construction per step | about 4 | 1 |
| Snapshot construction time per step | 194.87 ms | 1.82 ms |
| Physics calls time per step | 42.99 ms | 43.29 ms |
| Render time per step | 4.33 ms | 5.07 ms |

Measured wall-time reduction is approximately 74.2% (3.88x throughput).
This is one short trajectory, not a promise for every seed or a full
checkpoint evaluation. Raw results are in
`outputs/eval-profiles/wr08-{legacy,optimized}-1050-300-video.json`.

## Tests

Run with the repository-local simulation environment:

```bash
"$ROBOCASA_PYTHON" -m pytest --import-mode=importlib -q \
  tests/unit/test_wr08_contact_evidence.py \
  tests/test_wr08_physical_evaluation.py \
  tests/test_wr08_policy_evaluation.py \
  tests/test_wr08_evaluation.py \
  tests/unit/test_runtime_stop_execution.py \
  tests/simulator/test_wr08_policy_evaluation.py \
  tests/simulator/test_runtime_diagnostic_success.py
```

Real simulator tests require access to the EGL device. Importlib mode avoids
the existing duplicate test-module basenames under `tests/` and
`tests/simulator/`.

Validation completed for this change:

- 178 related unit/physical-event tests passed (WR08, WA01/WA05 events,
  WA05 evaluator, and shared runtime).
- Five real simulator tests passed: WR08 contacts and success/wrong-target
  termination, plus explicit-success runtime checks for WA01, WA05, and WR08.
- Same-state evidence and independent-evaluator comparisons passed for all
  2000 steps of seed 1053 and all 607 steps of seed 1075, including two-carton
  pickup history, correct placement, and wrong-target termination. An earlier
  100-step comparison of seed 1050 also passed. Full comparison output is
  `outputs/eval-profiles/wr08-verify-1053-1075-full.json`.
- New standalone files pass Ruff; `git diff --check` passes.

Broader test runs expose pre-existing stage-schema mismatches: three WA01
tests expect report-v2 / a `select` stage while the current profile is v1;
48 WR07 terminal-test parametrizations fail on `memory_color_latched`.
All 51 failures were reproduced in a separate process after loading the
unmodified HEAD versions of the shared runtime and hold helper into memory.
The worktree was not reverted, and these unrelated schemas were not changed.

This validation replays saved checkpoint actions. A new live policy-server
evaluation has not been launched. Use a new run directory for subsequent
evaluations rather than mixing old and optimized execution in one report.
