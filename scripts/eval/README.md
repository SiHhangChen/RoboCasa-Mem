# Evaluation Scripts

Policy-specific evaluation scripts are grouped by policy while remaining under
`scripts/eval/`:

- `pi05/`: OpenPI pi0.5 launchers, offline checkpoint evaluation, and pi0.5
  diagnostics.
- `gr00t/`: GR00T launchers and GR00T-specific reporting.
- `memer/`: MemER high-level and composed-policy evaluation.
- `memvla/`: MemoryVLA evaluation.
- `act/`: ACT offline evaluation.

Policy-independent tools stay at this directory level:

- `run_remote.py`: canonical remote closed-loop evaluation entry point.
- `summarize_run.py`: optional recovery/audit command that rebuilds the
  canonical `summary.json` from committed episode artifacts.
- `replay_run_actions.py`: replay `actions.npz` from an evaluation run.
- `replay_recorded_actions.py`: replay `trajectory.h5` from a native dataset.
- `eval_remote_dataset.py`: policy-protocol dataset diagnostics.

Every rollout and replay writes `summary.json` automatically. Report-stage
verdicts are produced by the task evaluator, persisted in each
`episodes/seed_*/result.json`, and aggregated by `bench.evaluation.summary`.
There are no task- or policy-specific benchmark summary scripts; rebuilding a
run must not reinterpret evaluator diagnostics.

The `summary` subcommand in policy launchers is therefore not required after a
normal run. It is useful for validating committed artifacts before publishing,
or for recovering a missing/stale summary:

```bash
python scripts/eval/summarize_run.py outputs/eval/<run-id>
```

Policy-specific offline analyses may still have their own aggregation tools
because they consume different artifacts and are not benchmark run summaries.

Run commands from the repository root. For example:

```bash
./scripts/eval/pi05/eval_pi05_wa01.sh eval
./scripts/eval/gr00t/eval_gr00t_wa01.sh eval
./scripts/eval/memer/eval_memer_wa01.sh eval
```

Both replay commands use the same `RolloutRunner`, task adapter, report-stage
contract, and artifact writer as remote evaluation:

```bash
python scripts/eval/replay_run_actions.py SOURCE_RUN OUTPUT_RUN
python scripts/eval/replay_recorded_actions.py \
  --task-id WR-06 \
  --dataset-root outputs/datasets/WR06 \
  --output-run outputs/eval/wr06-recorded-replay
```

Each launcher uses the canonical task suite in `configs/eval/suites/`. Change
`rollout.replan_steps` in that YAML before a run when the policy needs a
different replanning interval.
