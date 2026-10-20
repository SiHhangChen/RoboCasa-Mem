# Maintaining the evaluation release

The repository root is the installable nine-task evaluation project. The
former development tree is preserved in Git commit `a10358f`; it is not
required to install or use this checkout. The simulation, task resources,
27 suite files, and 50 WX-01 initialization transcripts were copied unchanged
from the independently accepted candidate. Seeded randomization remains part
of the task definition.

## Check and build

Use Python 3.10+ for the software and the pinned Python 3.11 simulator
environment in `INSTALL.md` for simulation. From the repository root:

```sh
python -m pip install -e '.[video]'
python scripts/release/validate_installation_evidence.py .
python -m pip wheel --no-deps --no-build-isolation . -w dist
```

Maintainer test sources and the pytest development extra are excluded from
this release checkout and its distributions. Their last published source is
preserved in Git commit `953b8ef`; they are not needed to evaluate tasks or
models. Recorded acceptance results remain under `installation_evidence/`
as provenance. Loading real checkpoints requires the corresponding policy
environment.

`validate_installation_evidence.py` checks the recorded 450-case acceptance,
runtime identity, pinned simulator versions, asset evidence, and release
file hashes. `finalize_candidate.py .` refreshes metadata and the release
inventory after checking evidence. Documentation and acceptance-tool changes
can retain existing runtime acceptance. Changes to runtime code, task
resources, or suites require matching new acceptance evidence. Do not edit
historical evidence to make a changed runtime appear accepted.

## Local files

`outputs/`, virtual environments, simulator source copies, checkpoints,
Python caches, `build/`, `dist/`, and `*.egg-info/` are ignored by Git. They
are local working files, not release source. Reviewed installation evidence
under `docs/installation_evidence/`, including its two test logs, is tracked.
The finalizer inventories release directories only and excludes local files.

During the root migration, a local copy of the checkpoint archive was saved
under `outputs/repository_reorganization/a10358f/`. The accepted environment
remains at `outputs/release_acceptance/isolated-20261020-v1/` so its installed
source and interpreter paths remain valid. On the original workstation:

```sh
source outputs/release_acceptance/isolated-20261020-v1/activate.sh
```

These are local convenience paths. A fresh clone is installed using
`INSTALL.md` and `ISOLATED_INSTALL.md`; it does not need this `outputs/` tree.
