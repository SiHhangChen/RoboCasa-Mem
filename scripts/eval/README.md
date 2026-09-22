# Evaluation Scripts

Policy-specific evaluation scripts are grouped by policy while remaining under
`scripts/eval/`:

- `pi05/`: OpenPI pi0.5 launchers, offline checkpoint evaluation, and pi0.5
  diagnostics.
- `gr00t/`: GR00T launchers and GR00T-specific reporting.
- `memer/`: MemER high-level and composed-policy evaluation.
- `memvla/`: MemoryVLA evaluation.
- `act/`: ACT offline evaluation.
- `diffusion/`: LeRobot DiffusionPolicy closed-loop launchers.

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

## Real-robot WR07 ACT diagnostics

The WR07 real-robot ACT checkpoint is served through the same WebSocket
protocol, but its contract is a closed `state17/action17`, `front/left/right`,
30 Hz profile. Attach or refresh the sidecar after validating the dataset
metadata:

```bash
.venv-gr00t/bin/python scripts/eval/real/prepare_act_checkpoint.py \
  --checkpoint outputs/train/act/wr07_real_robot_v3_bs48_224_stride10_chunk100/checkpoints/040000 \
  --dataset-root data/real_robot/wr07_lerobot_v3
```

Start the server with the existing launcher:

```bash
.venv-gr00t/bin/python scripts/serve/serve_lerobot.py \
  --checkpoint outputs/train/act/wr07_real_robot_v3_bs48_224_stride10_chunk100/checkpoints/040000 \
  --device cuda --host 127.0.0.1 --port 8000
```

Replay recorded WR07 observations through the server with hardware execution
disabled and save prediction/action MAE diagnostics:

```bash
.venv-gr00t/bin/python scripts/eval/real/replay_remote_dataset.py \
  --endpoint ws://127.0.0.1:8000 \
  --output outputs/eval/real/wr07_act_040000_replay.json
```

This is diagnostic-only. No script in this profile sends action17 to a robot;
the sidecar marks hardware execution semantics as `unverified` until the Cobot
Magic SDK command units, limits, hold, and emergency-stop behavior are audited.

## DiffusionPolicy checkpoints

The checkpoints under `outputs/train/diffusion_final_aligned_v1/` use two
consecutive observations and a native 47-action cache. Evaluation therefore
uses the dedicated suites in `configs/eval/diffusion/` with
`replan_steps: 1`: the server receives every observation and LeRobot's
`select_action()` owns the native action queue.

Start the task-specific policy server in one terminal, then preflight and run
the single-seed smoke in another:

```bash
bash scripts/eval/diffusion/eval_diffusion_final.sh serve wa05 40000 0
bash scripts/eval/diffusion/eval_diffusion_final.sh preflight wa05 40000
bash scripts/eval/diffusion/eval_diffusion_final.sh smoke wa05 40000
bash scripts/eval/diffusion/eval_diffusion_final.sh smoke-summary wa05 40000
```

After inspecting the smoke artifacts, run the canonical 50-seed suite with:

```bash
bash scripts/eval/diffusion/eval_diffusion_final.sh eval wa05 40000
```

The supported task names are `tm01`, `tm02`, `ts02`, `ts03`, `wa01`, `wa05`,
`wr01`, and `wr08`. Set `PORT` or `RENDER_GPU` when the defaults conflict with
another local service.
