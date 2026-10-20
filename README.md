# MobileMemBench evaluation release candidate

This candidate contains nine tasks in this order: WX-01, WA-01, WA-05,
WR-01, WR-08, TS02, TS03, TM01, TM02. Training, recording, expert policies,
and MPlib are excluded. RoboCasa, robosuite, their assets, and policy packages
must be installed separately. An independent installation with pinned sources
and restored assets passed all 450 official task/seed inputs with three short
control steps each. `release_candidate.json` records acceptance and the
remaining release preparation.

Run from this directory, using Python 3.10 or newer:

```sh
pip install -e '.[video]'
membench-eval --endpoint ws://localhost:8000 --suite pi05:wx01
membench-summarize --help
```

Use `configs/benchmark/evaluation_suite_sources.json` to select suites by
model family. The 27 copied YAML files preserve the exact bytes and hashes
of recorded 50-seed experiments. They are not one shared configuration:
pi05 uses r40; ACT uses each recorded suite's truncation interval; Diffusion
uses wire r1 with its server-side 47-action queue. Keep that queue and update
observations on every request. ACT WA-01's recorded budget is 5000 steps.

WX-01 starts with a physically held sponge. Verified initialization action
transcripts replace reset-time planning for seeds 1050–1099. These setup
steps precede policy control and do not consume the model evaluation budget.
A missing transcript or mismatched simulator state is an error. Artifacts
must pass source/replay observation and action equivalence before release.

TS03 pi05 seed 1056 stored its stage report under an empty JSON field name.
A verified correction restores `report_stages`; all 50 stage reports are now
included and overall success remains 14/50 (28%). Use the corrected summary
in `docs/historical_result_repairs/ts03_pi05_r40/summary.json`. That directory
also preserves the original result, original summary, and repair provenance.

See `docs/membench_evaluation_audit.md` for configuration and ACT/DP evidence.
Installed runtime acceptance has passed. Final source version freezing and
asset delivery remain release preparation steps.

See `docs/INSTALL.md` for simulator revisions, installed commands, and headless
rendering. `docs/ASSETS.md` records the existing 3035-file simulator archive,
its pinned inventory, restoration checks, and installed acceptance.
See `docs/ISOLATED_INSTALL.md` for the independent environment and evidence.
Third-party asset terms are described in `docs/THIRD_PARTY.md`.

The repository root contains the evaluation release directly. Task resources
retain their seeded randomization settings and existing recording metadata;
no data collection entry point or expert policy implementation is included.
Maintainer tests are excluded from the release repository and distributions;
their last published source is preserved in Git commit `953b8ef`.
See `docs/DEVELOPMENT.md` for validation, build commands, and release maintenance.
