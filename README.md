# MobileMemBench evaluation release candidate

This candidate contains nine tasks: WX-01, WA-01, WA-05, WR-01, WR-08,
TS02, TS03, TM01, TM02. The repository root contains the evaluation release.
Training, data collection entry points, expert policies, and MPlib are excluded.
Real-robot code is kept locally and excluded from the release; policy servers
support the RoboCasa simulation runtime profile.
Task resources retain their seeded randomization settings and recording metadata.
RoboCasa, robosuite, simulator assets, and model packages are separate dependencies.

Use Python 3.11 for the tested simulator environment. Create and activate a
virtual environment, then clone the simulator sources into a separate directory:

```sh
mkdir -p /path/to/simulator-sources
git clone https://github.com/ARISE-Initiative/robosuite.git /path/to/simulator-sources/robosuite
git -C /path/to/simulator-sources/robosuite checkout 232ce7d4a6ed89c949a9aba024a05c8c32fdd08b
git clone https://github.com/robocasa/robocasa.git /path/to/simulator-sources/robocasa
git -C /path/to/simulator-sources/robocasa checkout 8f124f0e263de0c583ba5d2acc87137bfd905986
pip install -e /path/to/simulator-sources/robosuite -e /path/to/simulator-sources/robocasa
pip install 'mujoco==3.3.1' 'numpy==2.2.5'
```

From this repository root, install the evaluation package:

```sh
pip install -e '.[video]'
```

Obtain `membench-sim-assets.tar.gz` separately; no public download URL has
been assigned yet. The pinned archive contains 3035 files, is 270484609 bytes,
and has SHA-256
`140176b06cf220f6d11440fe3cede54cca7577edf13473d62fb210bd03ca68e3`.
Its file inventory is `configs/benchmark/simulator_assets.json`.
Verify and restore it from this repository root:

```sh
python scripts/release/verify_assets.py --archive /path/to/membench-sim-assets.tar.gz
tar -xzf /path/to/membench-sim-assets.tar.gz -C /path/to/simulator-sources
python scripts/release/verify_assets.py --root /path/to/simulator-sources
```

Both checks must report `passed: true`; the restored-file check must report
`checked_files: 3035`. Keep the simulator asset directories writable because
RoboCasa creates temporary object XML files there. The tested simulator
versions and source revisions are in `configs/benchmark/simulator_environment.json`.

For headless NVIDIA rendering, set `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl`; choose `MUJOCO_EGL_DEVICE_ID` for your machine.
Start the appropriate model server with its checkpoint in a separate policy
environment: LeRobot for ACT/Diffusion, OpenPI for Pi0.5. See
`membench-serve-lerobot --help` and `membench-serve-openpi --help` for server options.
Then run:

```sh
membench-eval --endpoint ws://localhost:8000 --suite pi05:wx01
membench-eval --endpoint ws://localhost:8000 --suite act:wa01
membench-eval --endpoint ws://localhost:8000 --suite diffusion:wa01
membench-summarize /path/to/run
```

Use `configs/benchmark/evaluation_suite_sources.json` to select suites by
model family. The 27 copied YAML files preserve the exact bytes and hashes
of recorded 50-seed experiments (seeds 1050–1099). Pi0.5 uses r40; ACT uses
each recorded suite's truncation interval; Diffusion uses wire r1 with its
server-side 47-action queue. Keep that queue and update observations on every
request. ACT WA-01's recorded budget is 5000 steps.

WX-01 starts with a physically held sponge. Verified initialization action
transcripts replace reset-time planning for all 50 seeds. These setup steps
precede policy control and do not consume the model evaluation budget.
A missing transcript or mismatched simulator state is an error.

TS03 Pi0.5 seed 1056 stored its stage report under an empty JSON field name.
A verified correction restores `report_stages`; all 50 stage reports are
complete, and overall success remains 14/50 (28%). The corrected historical
summary and repair provenance are available in the archived documentation below.

An independent Python installation with pinned simulator sources and restored
assets passed all 450 official task/seed inputs: initialization, rendering,
and three hold-action steps each. This used the same host and GPU driver;
model inference and full model rollouts were not rerun. `release_candidate.json`
records this acceptance; freezing the final source revision and delivering
the asset archive remain release preparation steps.
That 450-case run describes the previous wheel. Removing the real-robot
integration changes the policy runtime files, so installed-wheel acceptance
must be repeated before claiming the same coverage for the current package.

Documentation and validation evidence are kept locally and excluded from Git
tracking and distribution packages. Their last published version is available
in the [documentation archive](https://github.com/SiHhangChen/RoboCasa-Mem/tree/e020312e8146c646c9013d35e9a7fc3523f0cdc1/docs).
All `docs/` paths in `release_candidate.json` refer to that archive or local
maintainer files. Evidence validation and candidate finalization scripts require
those local files; normal evaluation does not. Maintainer tests are also excluded;
their last published source is preserved in Git commit `953b8ef`.

Benchmark code uses the repository's MIT license. RoboCasa and robosuite are
separate dependencies with their own MIT notices and third-party terms;
MuJoCo is licensed under Apache 2.0. The benchmark license grants no additional
rights to third-party assets. Simulator assets and pretrained checkpoints are
not included; consult their source notices before redistribution.
