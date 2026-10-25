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

**Release status:** this is version `0.1.0rc1`. The simulator asset archive has
no public download URL yet. The current package also needs installed-wheel
acceptance after the removal of real-robot integration; see
[Validation and release status](#validation-and-release-status).

## Tasks

| Task | Description | Suite suffix |
| --- | --- | --- |
| WX-01 | Return an object to its original place | `wx01` |
| WA-01 | Sort sugared and unsugared coffee | `wa01` |
| WA-05 | Fill one cup and match its temperature button | `wa05` |
| WR-01 | Retrieve a remembered can | `wr01` |
| WR-08 | Match milk cartons to colored sugar plates | `wr08` |
| TS02 | Inspect three cabinets | `ts02` |
| TS03 | Add the specified amount of sugar to hot chocolate | `ts03` |
| TM01 | Boil water and inspect two drawers | `tm01` |
| TM02 | Fill a pot and retrieve a can | `tm02` |

Select a suite using `<family>:<suffix>`, for example `pi05:wx01`, `act:wa01`,
or `diffusion:ts03`. All three families have configurations for all nine tasks.
The full mapping is in
[`evaluation_suite_sources.json`](configs/benchmark/evaluation_suite_sources.json).

## Installation

The tested simulator environment uses Python 3.11, Linux, and an NVIDIA GPU
with EGL rendering. Use a separate Python environment for the simulator and
for each model stack; they communicate over WebSocket.

### 1. Create the simulation environment

```sh
git clone https://github.com/SiHhangChen/RoboCasa-Mem.git
cd RoboCasa-Mem
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

Obtain `membench-sim-assets.tar.gz` separately before running simulation.
**A public download link is not available yet.** The pinned archive contains
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

## Start a model server

Open a separate terminal and activate your model environment. Install the
compatible LeRobot or OpenPI implementation and obtain the checkpoint for the
chosen task. Model packages, task-specific OpenPI configurations, and pretrained
weights are supplied separately; installing this repository alone does not
provide them.

Install this checkout into the model environment to expose the server commands:

```sh
python -m pip install -e /path/to/RoboCasa-Mem
```

For ACT or Diffusion Policy, the server selects the backend from the
checkpoint's `config.json`:

```sh
membench-serve-lerobot \
  --checkpoint /path/to/pretrained_model \
  --device cuda \
  --port 8000
```

Use a compatible LeRobot checkpoint with the benchmark state/action and camera
metadata. `membench_policy.json` supplies the simulation state schema and any
required camera mapping; the server checks the checkpoint contract on loading.

For Pi0.5, replace `YOUR_OPENPI_CONFIG` with the configuration registered in
your OpenPI installation for that checkpoint:

```sh
membench-serve-openpi \
  --config YOUR_OPENPI_CONFIG \
  --checkpoint /path/to/checkpoint \
  --image-size 224 \
  --seed-by-episode \
  --port 8000
```

Both examples listen on `127.0.0.1`. Use `--host` when running the server on
another machine, and use that machine's address in the evaluation endpoint.
Run either command with `--help` for additional options.

Custom servers must implement the benchmark's `membench-policy/v2` protocol
with the simulation runtime profile: 37 state values, 13 action values,
three 224×224 RGB cameras (`agentview_left`, `agentview_right`, `eye_in_hand`),
and 20 Hz control. See the
[protocol definitions](bench/policy/protocol/contracts.py) and
[simulation data contract](bench/runtime/data_contract.py).

## Run an evaluation

Return to the simulation terminal with its environment activated. First check
the selected suite against the running model server:

```sh
membench-eval \
  --endpoint ws://localhost:8000 \
  --suite pi05:wx01 \
  --preflight-only
```

Preflight validates the task, suite, and remote policy contract without creating
a simulator environment. It does not verify asset restoration or rendering.
Then run the full 50-seed suite:

```sh
membench-eval \
  --endpoint ws://localhost:8000 \
  --suite pi05:wx01 \
  --output-root ./outputs/eval \
  --run-id wx01-pi05
```

For another model or task, choose its matching suite, such as `act:wa01` or
`diffusion:ts03`, and use a distinct run ID. A YAML file path is also accepted
by `--suite` for custom experiments.

The recorded suites preserve each model family's execution settings:

| Family | Execution settings |
| --- | --- |
| `pi05` | Replan every 40 actions (`r40`) |
| `act` | Use the recorded truncation interval for each task |
| `diffusion` | Request one action at a time (`r1`); retain the server's 47-action queue and update observations on each request |

Keep the suite settings when comparing against recorded experiments. The 27
suite files preserve their recorded bytes and hashes; for example, ACT WA-01
uses a 5000-step budget.

WX-01 begins with a physically held sponge. Included initialization transcripts
cover all 50 seeds and replace reset-time planning. Setup steps precede policy
control and do not consume the evaluation budget. A missing transcript or a
simulator-state mismatch causes an error; MPlib is not needed for evaluation.

## Results and resuming

The example above writes to `outputs/eval/wx01-pi05/`. Its
`run_manifest.json` records the suite, seeds, model contract, and runtime
identity; `summary.json` aggregates committed episode artifacts, including
task success and stage results. Videos are saved when enabled by the suite.

Repeat the same evaluation command with the same run ID, suite, and checkpoint
to resume; resuming is enabled by default. Add `--retry-errors` to retry error
episodes. Use a new run ID for a different configuration or model.

To rebuild and verify a summary from an existing run:

```sh
membench-summarize ./outputs/eval/wx01-pi05
```

The evaluation and summary commands exit with code `0` when results satisfy
the completeness and artifact checks, or `2` when they do not. A valid run may
contain task failures; completeness does not mean every episode succeeded.

## Validation and release status

The previous wheel passed an installation check in an independent Python
environment with pinned simulator sources and all 3035 restored assets. It
covered all 450 official task/seed inputs (nine tasks × 50 seeds), including
initialization, rendering, and three hold-action steps per case. This used the
same host and GPU driver; model inference and full model rollouts were not rerun.

The current package removes real-robot integration and changes policy runtime
files. Its built wheel passed 47 software regression checks, including a local
WebSocket evaluation flow, and checks of the OpenPI simulation interface and
all 27 suite selectors. The 450-case installed-wheel acceptance must be repeated
for the current package. Final source freezing and public asset delivery are
also pending. See [`release_candidate.json`](release_candidate.json).

Historical TS03 Pi0.5 seed 1056 stored a stage report under an empty JSON field
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
