# Pinned simulator assets

The evaluation candidate reuses the existing `membench-sim-assets.tar.gz`.
The archive is supplied separately from the Python package. Its complete file
inventory is `configs/benchmark/simulator_assets.json`:

| Property | Current archive |
| --- | --- |
| Compressed bytes | 270484609 (about 258 MiB) |
| Regular files | 3035 |
| Uncompressed bytes | 569469884 |
| SHA-256 | `140176b06cf220f6d11440fe3cede54cca7577edf13473d62fb210bd03ca68e3` |

Each inventory entry records a relative path, size, and SHA-256. All 3035
archived files were compared with the installed files in the existing
`membench` environment: no missing or different files were found. Evidence
is recorded in `docs/evaluation_asset_audit.json` and
`docs/asset_evidence/current_bundle_verification.json`.
The archive contains all eight configured WX-01 sponge variants and the
WR-08 `milk_0` and `milk_1` models. This records the existing archive's full
contents; it does not claim to be the minimal dependency set for nine tasks.

## Restore and verify

Obtain the archive separately. The current local source is
`/home/chensihang/membench/membench-sim-assets.tar.gz`; no public download
URL has been assigned. On another machine, transfer that same archive.

First clone and check out the pinned simulator sources as described in
`docs/INSTALL.md`. Keep both checkout directories under one directory,
such as `/path/to/simulator-sources/robocasa/` and
`/path/to/simulator-sources/robosuite/`. The archive augments these checkouts;
base simulator resources also come from the pinned Git revisions.

From the evaluation candidate directory, verify the archive before extracting:

```sh
python scripts/release/verify_assets.py --archive /path/to/membench-sim-assets.tar.gz
tar -xzf /path/to/membench-sim-assets.tar.gz -C /path/to/simulator-sources
python scripts/release/verify_assets.py --root /path/to/simulator-sources
```

Both checks must report `passed: true`; the restored-file check must report
`checked_files: 3035`. A mismatch exits with a nonzero status and lists the
affected paths. The verifier needs only Python 3.10+, not a simulator or GPU.
An explicit `--manifest /path/to/simulator_assets.json` can be provided when
using a different location for the inventory.

RoboCasa's upstream downloader remains another way to obtain files, but its
output must pass the same inventory check to establish file identity.
The benchmark license does not replace third-party asset terms; see
`docs/THIRD_PARTY.md`.

## Historical validation and current acceptance

The earlier environment recovery is documented in
`/home/chensihang/membench/docs/local_test_environment.md`. On 2026-09-12,
that process restored and compared 2835 files from an earlier archive and
passed a WA-01 seed 1050 smoke test: three control steps, three 224×224 RGB
views, state37, action13, and 20 Hz. The original smoke report is preserved
at `docs/asset_evidence/wa01_smoke_20260912.json`.

That historical archive's digest was
`774df8e04ee6863ced749cac32463133952025df41c678503d0dc6201c3dbdf8`.
The 2835-file count and digest describe that earlier validation, not the
current 3035-file archive. The older collector documentation's 2748-file
count also does not describe the current archive.

On 2026-10-20, an independent installation restored and verified all 3035
files from the current archive. The installed wheel passed all 450 official
task/seed inputs (nine tasks × seeds 1050–1099), including initialization,
three RGB views, state37/action13, and three hold-action steps per case.
No development/original-directory reads or expert/MPlib imports were needed.
This closes the current archive coverage and isolated installation checks
for these inputs. The environment uses the same host and GPU driver; model
inference and full task-success trajectories were not rerun. Evidence and
reproduction steps are in `docs/ISOLATED_INSTALL.md` and
`docs/installation_evidence/manifest.json`. The inventory's original scope
flags remain a snapshot of the earlier audit; current acceptance is recorded
in that evidence and `release_candidate.json`.
