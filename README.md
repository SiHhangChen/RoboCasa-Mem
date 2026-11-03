# RoboCasa-Mem

[English](README.md) | [简体中文](README.zh-CN.md)

RoboCasa-Mem provides nine memory-dependent mobile manipulation tasks in
RoboCasa, with evaluation interfaces for Pi0.5, ACT, Diffusion Policy, and
compatible custom policy servers.

Each recorded evaluation suite covers 50 seeds (1050–1099). Evaluation runs
save episode results, stage reports, and a summary for each task and model.
This repository contains the simulation evaluation code and configurations;
simulator sources, assets, model implementations, and checkpoints are installed
separately.

Downloads on ModelScope:

- [Simulator assets](https://modelscope.cn/datasets/ouhebai/RoboCasa-Mem/tree/master/sim-asset)
- [Dataset](https://modelscope.cn/datasets/ouhebai/RoboCasa-Mem/tree/master/data)

**Release status:** this is version `0.1.0rc1`. The current package needs installed-wheel
acceptance after the removal of real-robot integration; see
[Validation and release status](#validation-and-release-status).

## Tasks

| Task name | Task ID | Description | Suite suffix |
| --- | --- | --- | --- |
| PutBackCup | WX-01 | Return an object to its original place | `wx01` |
| SweetCup | WA-01 | Sort sugared and unsugared coffee | `wa01` |
| TempReport | WA-05 | Fill one cup and match its temperature button | `wa05` |
| CanRecall | WR-01 | Retrieve a remembered can | `wr01` |
| MilkMatch | WR-08 | Match milk cartons to colored sugar plates | `wr08` |
| DrawerSearch | TS-02 | Inspect three cabinets | `ts02` |
| SugarCount | TS-03 | Add the specified amount of sugar to hot chocolate | `ts03` |
| HeatAndInspect | TM-01 | Boil water and inspect two drawers | `tm01` |
| FillAndRetrieve | TM-02 | Fill a pot and retrieve a can | `tm02` |

Task IDs consistently use a hyphen, such as `WX-01` and `TS-02`. Legacy spellings
such as `TS02` remain accepted on input and when reading historical results.
Suite suffixes, filenames, and Python package names use compact lowercase forms
such as `ts02`.

Select a suite using `<family>:<suffix>`, for example `pi05:wx01`, `act:wa01`,
or `diffusion:ts03`. All three families have configurations for all nine tasks.
The release contains only these 27 evaluation suites, organized under
`configs/eval/act/`, `configs/eval/diffusion/`, and `configs/eval/pi05/`,
with task names such as `wa01.yaml`.
The full mapping is in
[`evaluation_suite_sources.json`](configs/benchmark/evaluation_suite_sources.json).

## Environments and terminals

**`.venv` is the main Python environment required to run RoboCasa-Mem.** Install
the project, run task simulation and model evaluation, and summarize results in
this environment. Use two terminals with separate project and policy
environments:

- **Terminal A, project runtime environment (`.venv`, Python 3.11):** install RoboCasa,
  robosuite, simulator assets, and this repository. Run `membench-eval` and
  `membench-summarize` here. This process creates the task, renders observations,
  executes actions, and saves results.
- **Terminal B, policy environment:** install the chosen model implementation
  and this repository. Load the checkpoint and run inference here, using
  `membench-serve-lerobot` for ACT or DP, or `membench-serve-openpi` for Pi0.5.
  Choose Python and dependency versions for that model implementation.

Terminal A sends observations to Terminal B over WebSocket and receives model
actions. Keep the model server running while evaluating. The environments can
share a GPU if memory permits; separate environments do not require two GPUs.
The simulator requires Linux and an NVIDIA GPU with EGL rendering.

## Install the project runtime environment (Terminal A)

### 1. Create the project runtime environment

```sh
git clone https://github.com/SiHhangChen/RoboCasa-Mem.git
cd RoboCasa-Mem
export ROBOCASA_MEM_ROOT="$PWD"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install the pinned simulator sources

Replace `/path/to/simulator-sources` with a directory outside this checkout.
Keep `robosuite/` and `robocasa/` under that same directory so the asset archive
can be restored into both source trees.

```sh
mkdir -p /path/to/simulator-sources
git clone https://github.com/ARISE-Initiative/robosuite.git /path/to/simulator-sources/robosuite
git -C /path/to/simulator-sources/robosuite checkout 232ce7d4a6ed89c949a9aba024a05c8c32fdd08b
git clone https://github.com/robocasa/robocasa.git /path/to/simulator-sources/robocasa
git -C /path/to/simulator-sources/robocasa checkout 8f124f0e263de0c583ba5d2acc87137bfd905986
python -m pip install -e /path/to/simulator-sources/robosuite -e /path/to/simulator-sources/robocasa
python -m pip install 'mujoco==3.3.1' 'numpy==2.2.5' 'numba==0.61.2' 'scipy==1.15.3' 'Pillow==12.3.0' 'PyOpenGL==3.1.10'
```

From the RoboCasa-Mem repository root, install the evaluation package:

```sh
python -m pip install -e '.[video]'
```

The `video` extra supplies OpenCV for video output. The pinned simulator
revisions and tested dependency versions are recorded in
[`simulator_environment.json`](configs/benchmark/simulator_environment.json).
MuJoCo and NumPy versions matter for initialization replay equivalence.

### 3. Restore simulator assets

Download `membench-sim-assets.tar.gz` from the
[ModelScope simulator assets directory](https://modelscope.cn/datasets/ouhebai/RoboCasa-Mem/tree/master/sim-asset)
before running simulation. The pinned archive contains
3035 files and is 270484609 bytes (about 258 MiB). Its SHA-256 is:

```text
140176b06cf220f6d11440fe3cede54cca7577edf13473d62fb210bd03ca68e3
```

From the repository root, verify the archive, extract it into the parent of
the simulator checkouts, and verify the restored files:

```sh
python scripts/tools/verify_assets.py --archive /path/to/membench-sim-assets.tar.gz
tar -xzf /path/to/membench-sim-assets.tar.gz -C /path/to/simulator-sources
python scripts/tools/verify_assets.py --root /path/to/simulator-sources
```

Both checks must report `passed: true`; the restored-file check must report
`checked_files: 3035`. The inventory is
[`simulator_assets.json`](configs/benchmark/simulator_assets.json).
Keep the asset directories writable: RoboCasa creates temporary object XML
files alongside its assets.

### 4. Configure headless rendering

Set these variables in the simulation terminal. Choose an EGL device index
available on your machine:

```sh
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

## Start a policy server (Terminal B)

### Prepare the policy environment

The policy environment supplies the implementation and dependencies for the
chosen model. Keep policy environments and model sources under this checkout's
`policy/` directory. ACT and DP share `policy/.venv` and `policy/lerobot`;
Pi0.5 uses its OpenPI environment under `policy/openpi/`. Install this repository
in each policy environment to expose its server commands. `policy/` is ignored
by Git and excluded from the release package.

For ACT and DP (Diffusion Policy), the following LeRobot installation is a
reference configuration. It requires Python 3.12. Set `ROBOCASA_MEM_ROOT` to the
same checkout used in Terminal A.
The pinned LeRobot commit is recorded in the historical ACT evaluation manifests;
the `diffusers-dep` extra supplies DP dependencies.

```sh
export ROBOCASA_MEM_ROOT=/absolute/path/to/RoboCasa-Mem
cd "$ROBOCASA_MEM_ROOT"
mkdir -p policy
python3.12 -m venv policy/.venv
source policy/.venv/bin/activate
export TORCH_HOME="$ROBOCASA_MEM_ROOT/policy/.cache/torch"
export HF_HOME="$ROBOCASA_MEM_ROOT/policy/.cache/huggingface"
python -m pip install --upgrade pip
git clone https://github.com/huggingface/lerobot.git policy/lerobot
git -C policy/lerobot checkout 64b23178d5348609c266250d3e1f511eba4c33ff
python -m pip install 'torch==2.7.1' 'torchvision==0.22.1' 'numpy==2.2.5' -e 'policy/lerobot[diffusers-dep]'
python -m pip install -e "$ROBOCASA_MEM_ROOT"
python -m pip check
python -c 'from lerobot.policies.factory import make_pre_post_processors; from lerobot.policies.act.modeling_act import ACTPolicy; from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy; print("ACT and DP imports OK")'
```

If Git cloning times out, replace the clone and checkout commands above with
this download of the same pinned official source, then continue with the pip
installation commands:

```sh
curl -fL --connect-timeout 15 --max-time 90 --retry 1 -o policy/lerobot-64b23178.tar.gz https://codeload.github.com/huggingface/lerobot/tar.gz/64b23178d5348609c266250d3e1f511eba4c33ff
mkdir -p policy/lerobot
tar -xzf policy/lerobot-64b23178.tar.gz -C policy/lerobot --strip-components=1
```

On 2026-10-26, this shared environment passed fresh installation, dependency and
import checks, and CUDA forward passes for both ACT and DP using trained WA-01
checkpoints and synthetic observations. Simulation smoke validation is described
in [Validation and release status](#validation-and-release-status).
The first model load may download ResNet18 backbone weights into `TORCH_HOME`.
The project runtime's LeRobot 0.3.3 dependency lacks `make_pre_post_processors`;
run the LeRobot policy server in the policy environment.

Choose one of the examples below for the model being evaluated. Each example
uses port 8000 and a WA-01 checkpoint. Keep the selected server running while
Terminal A evaluates the matching suite.

### ACT example

In the policy environment, replace the checkpoint path with your ACT model:

```sh
membench-serve-lerobot \
  --checkpoint /path/to/act-wa01/checkpoints/040000/pretrained_model \
  --device cuda \
  --port 8000
```

In Terminal A, use `--suite act:wa01`.

### DP example

In the policy environment with Diffusion dependencies installed, replace the
checkpoint path with your DP model:

```sh
membench-serve-lerobot \
  --checkpoint /path/to/dp-wa01/checkpoints/040000/pretrained_model \
  --device cuda \
  --port 8000
```

In Terminal A, use `--suite diffusion:wa01`. The server selects ACT or DP from
the checkpoint's `config.json`.

Both LeRobot checkpoint directories must include `config.json`, model weights, saved
preprocessor/postprocessor configurations and statistics, and the benchmark
metadata in `membench_policy.json`. That metadata supplies the simulation state
schema and required camera mapping; the server checks the contract on loading.

### Pi0.5 example

In Terminal B, activate a policy environment containing the compatible OpenPI
implementation. Replace `YOUR_OPENPI_CONFIG` with the registered configuration
for your WA-01 checkpoint:

```sh
export ROBOCASA_MEM_ROOT=/absolute/path/to/RoboCasa-Mem
source "$ROBOCASA_MEM_ROOT/policy/openpi/.venv/bin/activate"
python -m pip install -e "$ROBOCASA_MEM_ROOT"
membench-serve-openpi \
  --config YOUR_OPENPI_CONFIG \
  --checkpoint /path/to/pi05-wa01/checkpoint \
  --image-size 224 \
  --seed-by-episode \
  --port 8000
```

In Terminal A, use `--suite pi05:wa01`. Model packages, task-specific OpenPI
configurations, and checkpoints are supplied separately. The LeRobot installation
above does not install OpenPI.

All examples listen on `127.0.0.1`. Use `--host` when running the server on
another machine, and use that machine's address in the evaluation endpoint.
Run the corresponding server command with `--help` for additional options.

Custom servers must implement the benchmark's `membench-policy/v2` protocol
with the simulation runtime profile: 37 state values, 13 action values,
three 224×224 RGB cameras (`agentview_left`, `agentview_right`, `eye_in_hand`),
and 20 Hz control. See the
[protocol definitions](bench/policy/protocol/contracts.py) and
[simulation data contract](bench/runtime/data_contract.py).

## Run an evaluation (Terminal A)

Return to Terminal A and activate the project runtime environment, `.venv`. The following
example connects to the ACT WA-01 server started above. First check the suite
against the running model server:

```sh
cd /absolute/path/to/RoboCasa-Mem
source .venv/bin/activate
membench-eval \
  --endpoint ws://localhost:8000 \
  --suite act:wa01 \
  --preflight-only
```

Preflight validates the task, suite, and remote policy contract without creating
a simulated task instance. It does not verify asset restoration or rendering.

### One round: one task, one model, 50 seeds

One round evaluates the same checkpoint on all 50 environment seeds (1050–1099)
for the selected task. `membench-eval` selects a suite and calls
[scripts/eval/run_remote.py](scripts/eval/run_remote.py). Keep the matching model
server running in Terminal B and run this in Terminal A:

```sh
membench-eval \
  --endpoint ws://localhost:8000 \
  --suite act:wa01 \
  --output-root ./outputs/eval \
  --run-id wa01-act-round1
```

For another model or task, load its matching checkpoint in Terminal B and
choose the corresponding suite in Terminal A, such as `pi05:wx01` or
`diffusion:ts03`. Use a distinct run ID. A YAML file path is also accepted by
`--suite` for custom experiments.

The recorded suites preserve each model family's execution settings:

| Family | Execution settings |
| --- | --- |
| `pi05` | Replan every 40 actions (`r40`) |
| `act` | Use the recorded truncation interval for each task |
| `diffusion` | Request one action at a time (`r1`); retain the server's 47-action queue and update observations on each request |

WA-01 uses an explicit 2000-step budget for ACT, DP, and Pi0.5, with a matching
2000-step task horizon. Historical ACT and DP WA-01 runs used 5000 steps; their
recorded results are not directly comparable to the new budget. The other eight
tasks retain their existing per-family budgets and replanning intervals.

WX-01 begins with a physically held sponge. Included initialization transcripts
cover all 50 seeds and replace reset-time planning. Setup steps precede policy
control and do not consume the evaluation budget. A missing transcript or a
simulator-state mismatch causes an error; MPlib is not needed for evaluation.

### Five rounds: repeat the same 50 seeds

For a formal comparison, evaluate each model/task pair five times, keeping the
checkpoint, suite, and environment seeds fixed. Give each round a different run
ID. This produces 250 episode runs per task with 50 unique environment seeds,
or 2250 episode runs across all nine tasks.

The following command runs five ACT WA-01 rounds sequentially in Terminal A.
Keep the matching checkpoint's server running in Terminal B. Each evaluation
creates fresh task instances and resets policy state at the start of each
episode. The loop stops if a round fails to run or pass its completeness checks:

```sh
(
  set -e
  for eval_round in 1 2 3 4 5; do
    membench-eval \
      --endpoint ws://localhost:8000 \
      --suite act:wa01 \
      --output-root ./outputs/eval \
      --run-id "wa01-act-round${eval_round}"
  done
)
```

Results are saved separately in `outputs/eval/wa01-act-round1/` through
`wa01-act-round5/`. Existing run IDs resume by default; change the run ID prefix
for a fresh five-round experiment. For another model/task, change the checkpoint
in Terminal B, the suite, and the run ID prefix together. For example, DP WA-01
uses `diffusion:wa01` and `wa01-dp-round${eval_round}`.

These rounds measure repeatability on a fixed test set. DP seeds its RNG from
the episode seed by default, and the Pi0.5 example explicitly enables
`--seed-by-episode`. The same seed therefore also repeats the policy sampling
seed across rounds. The five rounds are not five independent policy samples;
scores can repeat and their standard deviation can be zero.

## Results and resuming

Run result commands in Terminal A. The example above writes to
`outputs/eval/wa01-act-round1/`. Its
`run_manifest.json` records the suite, seeds, model contract, and runtime
identity; `summary.json` aggregates committed episode artifacts, including
task success and stage results. Videos are saved when enabled by the suite.

Repeat the same evaluation command with the same run ID, suite, and checkpoint
to resume; resuming is enabled by default. Add `--retry-errors` to retry error
episodes. Use a new run ID for a different configuration or model.

To rebuild and verify a summary from an existing run:

```sh
membench-summarize ./outputs/eval/wa01-act-round1
```

### Single-round SCR and stage results

To compute paper SCR for one task's complete 50-seed run, use the same project
environment. This works with ACT, DP, Pi0.5, or any policy producing the standard
evaluation artifacts; the policy and checkpoint are read from the run manifest:

```sh
python scripts/eval/compute_scr.py ./outputs/eval/wa01-act-round1
```

The input is one run directory containing `run_manifest.json` and all 50
`episodes/seed_*/result.json` records, with their required action and request
metric artifacts. Seeds must exactly match the 50 unique seeds in the manifest.
Incomplete runs, error/cancelled episodes, missing stage reports, and incompatible
stage profiles are rejected with exit code `2`.

The command writes `scr.json`, `stages.csv`, `episode_stages.csv`, and `SCR.md`
under `<run_dir>/scr/`; use `--output-dir PATH` to choose another directory.
It recomputes metrics from episode verdicts without changing the raw artifacts
or `summary.json`. SCR averages each stage's passed/applicable rate. SugarCount
uses each stage's own applicable denominator; MilkMatch uses its six paper
stages, excluding the extra `success` stage. CanRecall's first two stages mean
closing either first door and then having both doors closed, in either order.
A stage with zero applicable episodes makes SCR undefined and is rejected.

### Five-round SCR mean and standard deviation

After all five rounds finish, pass their original run directories in order in
Terminal A:

```sh
python scripts/eval/summarize_rounds.py \
  ./outputs/eval/wa01-act-round1 \
  ./outputs/eval/wa01-act-round2 \
  ./outputs/eval/wa01-act-round3 \
  ./outputs/eval/wa01-act-round4 \
  ./outputs/eval/wa01-act-round5 \
  --output-dir ./outputs/eval/wa01-act-five-rounds
```

This calls the single-run scoring logic on the original episode artifacts; it
does not require precomputed `scr.json` files or trust cached scores. The five
runs must have distinct directories and run IDs, with the same task, policy
metadata, checkpoint, suite, code revision, rollout settings, seeds, and stage
applicability. An incomplete run or incompatible configuration is rejected.

First compute each round's `SCR_r` as the equal mean of its per-stage
passed/applicable rates, then calculate:

```text
mean = (SCR_1 + SCR_2 + SCR_3 + SCR_4 + SCR_5) / 5
std  = sqrt(Σ_r (SCR_r - mean)² / 4)
```

This is the sample standard deviation across five round scores (`ddof=1`).
SR and individual stage rates use the same round-level aggregation. The output
directory receives `summary.json` and `SUMMARY.md`, including the five values,
their mean, and their sample standard deviation. JSON rates use the 0–1 scale;
Markdown displays means as percentages and standard deviations in percentage
points.

Repeat this workflow for each task. Overall nine-task SCR is the equal mean
of the nine task scores within each round, followed by the mean and sample
standard deviation of those five overall values.

The evaluation and summary commands exit with code `0` when results satisfy
the completeness and artifact checks, or `2` when they do not. A valid run may
contain task failures; completeness does not mean every episode succeeded.

## Validation and release status

On 2026-10-25, a fresh Python environment with an independent local release-source
snapshot passed all 450 simulation checks (nine tasks, seeds 1050–1099), including
three hold-action steps, three-camera rendering, stage reports, and artifacts.
This was an editable source installation; it did not test real-model inference
or replace installed-wheel acceptance. On 2026-10-26, the shared policy environment
separately passed dependency and import checks and ACT/DP CUDA inference with
trained WA-01 checkpoints and synthetic observations. An ACT WA-01 smoke run on
seed 1050 then completed 2000 simulation steps and 40 model requests without
runtime errors, with 2000 actions, three 2000-frame videos, and a complete stage
report. It passed the sugar-added stage but did not complete the task within the
budget. This validates one execution path, not full benchmark accuracy; DP model
simulation episodes remain unverified.

The previous wheel passed an installation check in an independent Python
environment with pinned simulator sources and all 3035 restored assets. It
covered all 450 official task/seed inputs (nine tasks × 50 seeds), including
initialization, rendering, and three hold-action steps per case. This used the
same host and GPU driver; model inference and full model rollouts were not rerun.

The current package removes real-robot integration, changes policy runtime
files, reorganizes the evaluation configurations, and sets the WA-01 budget to
2000 steps. An earlier wheel after real-robot removal passed 47 software
regression checks, including a local WebSocket evaluation flow and checks of the
OpenPI simulation interface and all 27 suite selectors. The reorganized release
passed all 27 selectors and resolution of 450 episode input identities; its wheel
and source distribution each contain exactly the 27 official evaluation suites.
The 450-case installed-wheel acceptance must be repeated
for the current package. Final source freezing is also pending.
See [`release_candidate.json`](release_candidate.json).

Historical TS-03 Pi0.5 seed 1056 stored a stage report under an empty JSON field
name. A verified correction restored `report_stages`; all 50 historical stage
reports are complete, and overall success remains 14/50 (28%). The corrected
summary and provenance are in the
[archived repair bundle](https://github.com/SiHhangChen/RoboCasa-Mem/tree/e020312e8146c646c9013d35e9a7fc3523f0cdc1/docs/historical_result_repairs/ts03_pi05_r40).

Maintainer documentation, validation evidence, release-maintenance scripts,
tests, and real-robot code are
excluded from the release tree and distribution packages. Historical evidence
is available in the
[documentation archive](https://github.com/SiHhangChen/RoboCasa-Mem/tree/e020312e8146c646c9013d35e9a7fc3523f0cdc1/docs);
`docs/` paths in release metadata refer to that revision or local maintainer
files. Evidence validation and candidate finalization tools require those
files; normal evaluation does not. Test sources are preserved at commit
`953b8ef`.

## License

Benchmark code is distributed under the [MIT license](LICENSE). RoboCasa and
robosuite are separate dependencies with their own MIT notices and third-party
terms; MuJoCo uses Apache 2.0. Simulator assets and checkpoints have their own
terms. The benchmark license grants no additional rights to those materials;
consult their source notices before redistribution.
