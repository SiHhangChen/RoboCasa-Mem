# Installed candidate acceptance

The acceptance environment is separate from the existing training/evaluation
environment. The local run is kept under
`outputs/release_acceptance/isolated-20261020-v1/` in the development repository.
The Python environment, pinned simulator source copies, restored assets,
wheel, caches, and working directory are all under that directory.

The simulator packages are editable installs of those independent source
copies because RoboCasa needs writable asset directories. MobileMemBench is
installed from its wheel. Neither simulator points into the original
`membench` directory. A shared base Python interpreter and the host GPU driver
are used; this is isolation on the same host, not a second physical machine.

## Reproduce

Create a directory outside the development checkout and original simulator
checkout, with subdirectories `sources`, `artifacts`, `reports`, `work`, and
`cache`. Use Python 3.11.15 to create `.venv` in that directory.

Put the RoboCasa and robosuite sources at the revisions in
`configs/benchmark/simulator_environment.json` under `sources/robocasa` and
`sources/robosuite`. Use the separately supplied archive and commands in
`docs/ASSETS.md` to restore and verify assets under `sources/`.

The tested dependency versions are recorded in
`docs/installation_evidence/requirements-tested.txt`. They are an inventory
of the working environment, not a proposed minimal dependency set. The
acceptance run installed these packages independently; it did not copy
the old environment's `site-packages`.

From the new acceptance directory:

```sh
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r /path/to/candidate/docs/installation_evidence/requirements-tested.txt
uv pip install --python .venv/bin/python wheel
uv pip install --python .venv/bin/python --no-deps --no-build-isolation \
  -e sources/robocasa -e sources/robosuite \
  artifacts/mobilemembench-0.1.0rc1-py3-none-any.whl
uv pip check --python .venv/bin/python
cp /path/to/candidate/scripts/release/check_installed_candidate.py work/
```

The two indexes above are the official PyPI and PyTorch CPU indexes. Matching
both is needed because the CPU PyTorch build is pinned, while general Python
packages come from PyPI. All packages in the version inventory are pinned.

Run a check using the new interpreter, `-I` (isolated Python), and an unrelated
working directory. Replace paths and the GPU number for your machine:

```sh
/path/to/acceptance/.venv/bin/python -I \
  /path/to/acceptance/work/check_installed_candidate.py \
  --acceptance-root /path/to/acceptance \
  --development-root /path/to/development-checkout \
  --original-root /path/to/original-membench \
  --task WA-01 --seed 1050 --steps 3 --render-device 1
```

The worker checks module origins and the nine-task catalog, blocks Python
file reads from the original/development directories, and blocks expert,
recording, and MPlib imports. A task check initializes the real simulator,
executes three actions holding the current arm position, validates three
224×224 RGB frames and state37/action13, validates the stage report, and
writes ordinary rollout artifacts and PNGs. A task's success/failure verdict
is separate from passing this installation check.

The complete acceptance set is the nine candidate tasks × seeds 1050–1099.
Each case uses its recorded Pi0.5 suite's task configuration, with the
rollout budget shortened to three hold-action steps. Model inference and
full task-success trajectories are not part of this check. Per-case reports
and images remain under the local acceptance directory; the release carries
their aggregate report and installation provenance in
`docs/installation_evidence/`.

The earlier input lock was resolved with `allow_partial_calibration=true`.
Installed evaluation deliberately uses `false`. That explicit policy bit is
part of the configuration hash, so the hashes differ even when file contents
are identical. Acceptance recomputes both hashes with their original policy
bits and compares resource contents against the input lock; it does not
weaken the evaluator's calibration checks.

The original accepted wheel digest is recorded in the evidence. Later changes
to runtime Python files, task resources, or evaluation suites invalidate the
acceptance check in the finalizer. Changes limited to release metadata,
documentation, or acceptance tools do not change the runtime that was tested.

## Recorded result (2026-10-20)

All 450 official task/seed inputs passed. All 3035 restored assets matched
the inventory; 97 software tests, one localhost WebSocket test, and all 27
suite selectors passed. See `docs/installation_evidence/manifest.json` for
artifact identities and `full_input_coverage.json` in that directory for
per-case reports. This acceptance uses the same host and short hold actions;
it does not establish full model task-success rates or a second-host test.
