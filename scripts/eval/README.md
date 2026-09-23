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

## Real-robot WR07 OpenPI replay

The two OpenPI WR07 configs use the same `state17/action17`, `front/left/right`,
30 Hz protocol. The server maps the wire cameras to the trained OpenPI aliases:
`front -> agentview_left`, `left -> agentview_right`, and
`right -> eye_in_hand`. It loads normalization statistics from the selected
checkpoint. Use a checkpoint trained with the matching config.

Run the server in one terminal (requires an OpenPI environment with the model
dependencies) and the recorded-data client in another:

```bash
MODEL=full CKPT_DIR=/path/to/full/checkpoint \
  bash scripts/eval/real/eval_pi05_wr07.sh serve
MODEL=full bash scripts/eval/real/eval_pi05_wr07.sh replay
```

The full model uses the complete task instruction from its training config.
For the subgoal model, `replay` asks once per episode for the source-container
color: brown, white, or pink. The chosen training prompt is sent on every
inference request in that episode.

```bash
MODEL=subgoal CKPT_DIR=/path/to/subgoal/checkpoint \
  bash scripts/eval/real/eval_pi05_wr07.sh serve
MODEL=subgoal bash scripts/eval/real/eval_pi05_wr07.sh replay
```

For reproducible, noninteractive replay of labeled episodes, use
`PROMPT_SOURCE=annotation` with the local WR07 annotations. `MAX_FRAMES`,
`STRIDE`, `PORT`, `OUTPUT`, `OPENPI_PYTHON`, and `REPLAY_PYTHON` can be set as
environment variables. Output includes the selected prompt for each episode,
predicted action chunks' first action, recorded first action, and action MAE.
This replay is diagnostic and does not command hardware.

## Real-robot WR07 baseline Omega inference

The `policy/baseline` branch `feature/pi05-multitask-omega-training` defines a
different `pi05_real_robot_wr07_v3` model: it consumes the active color subtask
and VGGT-Omega tokens from a six-step temporal window. Its server must run in
an environment with that branch's OpenPI dependencies and VGGT-Omega installed.
Set `BASELINE_PYTHON` to that environment's Python executable. The model
checkpoint and Omega checkpoint must be the weights used by this training run.
The server reads the same `meta/stats.json` used during training from
`DATASET_ROOT`.

```bash
BASELINE_PYTHON=/path/to/baseline-env/bin/python \
CKPT_DIR=/path/to/pi05_real_robot_wr07_v3/checkpoint \
OMEGA_REPO=/path/to/vggt-omega \
OMEGA_CHECKPOINT=/path/to/vggt_omega_1b_256_text.pt \
bash scripts/eval/real/eval_baseline_wr07.sh serve
```

Replay recorded observations from another terminal. The client sends native
`480x640` RGB frames alongside protocol-sized images. The server feeds the
native frames to Omega and Pi0.5; OpenPI applies the same `224x224` PIL resize
used during training. Replay selects frames at episode-local multiples of 10,
uses the recorded color prompt, resets at each episode, and records action
error and timing. The server feeds Omega the current frame followed by five
earlier decision frames, repeating the episode's first frame during warm-up,
as in the training cache.

```bash
bash scripts/eval/real/eval_baseline_wr07.sh replay
```

Before the model weights and Omega installation are available, this command
checks one real recorded frame through LeRobot decoding, WebSocket transport,
the baseline contract, and a fake 50x17 policy response:

```bash
.venv-gr00t/bin/python scripts/eval/real/smoke_baseline_wr07_replay.py
```

For a robot-side client, implement `create_robot_io()` in an importable module
according to `bench/real_robot/io.py`. The driver returns synchronized
`state17` and raw RGB `front/left/right` observations. The client defaults to
shadow mode, which records actions without sending them to the robot:

```bash
REPLAY_PYTHON=.venv-gr00t/bin/python
PYTHONPATH=/path/to/robot/driver:${PYTHONPATH:-} \
  "$REPLAY_PYTHON" scripts/eval/real/run_baseline_wr07_client.py \
  --endpoint ws://127.0.0.1:59219 --driver my_wr07_driver \
  --color brown --episode-id wr07-live-001 \
  --output outputs/eval/real/wr07_baseline_shadow_001.json
```

`--execute` sends only the first 10 actions from each 50-action prediction;
the client requires a 10-frame replan interval to match the Omega history.
It requires `verify_execution_contract()` in the robot driver to return
`True` after validating command units, arm/base mapping, limits, hold and
emergency-stop behavior. Each completed decision is also flushed to a `.jsonl`
journal beside the final JSON result. The repository does not provide a Cobot
Magic driver.

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
