# WR-08

WR-08 v3 moves two different milk carton models from the coffee-machine-side
sugar plates to matching colored sugar plates on the opposite counter. Each episode samples
three distinct plate colors from a fixed 20-color pool, randomizes the source
left/center/right arrangement, and applies a reproducible permutation to the
target arrangement. The two cartons occupy distinct source plates and are
transported in a seed-deterministic order.

`boxed_food_a` is bound to the Objaverse `milk_0` model and `boxed_food_b` is
bound to `milk_1`. These are distinct visual and collision assets; their own
bottom offsets are used when computing reset height, and their own geometry
must pass grasp and shelf-clearance checks.
When the taller `milk_0` carton is assigned to `source_right`, its sampled
in-plate offset is deterministically mirrored toward the open side of the tray;
the adjusted offset remains part of the resolved episode facts.

Each transport uses the current WR-07 motion setup: the five calibrated shared
source-counter poses, the target-side 10 cm disk, four object-relative top-down
grasp candidates, four placement yaw candidates, and the asset-native contact
parameters. At reset, each carton's height is computed from its source tray's
top offset and its own bottom offset. Episode configuration validates these
seed-owned assignments without repositioning objects after reset settling.

The task-local pick executor checks the compiled open-cabinet shelf geoms,
including procedural shelves missing from the fixture's cached contact list.
The nominal pick uses a 6 cm pregrasp and 12 cm lift. Each preflight additionally
requires at least 1 cm of arm-to-shelf clearance along the interpolated virtual
path. If all four nominal yaw candidates fail because of collision preflight,
the selector retries the same full pick chains with pregrasp capped at 3 cm and
lift capped at 6 cm. Existing lower heights are never raised; no seed or carton
gets a special-case bypass. The lower candidate still must pass pregrasp, grasp,
lift, runtime collision, and physical grasp/placement checks. Missing geometry or
invalid distance data cannot qualify a path as safe.

The selected pregrasp plan is reused; grasp and lift targets stay anchored in
world coordinates if the mobile base drifts. Multi-stage preflight uses the
preceding virtual arm endpoint and restores the complete integration state,
including solver warm-start fields. Only the virtual arm configuration is
sampled: the live rollout is driven by actions, without carton free-joint writes,
teleporting or follow assistance.
The shared controller, gripper forces, contact materials, four-yaw selection,
and physical acceptance conditions are unchanged.

## Execution and acceptance

The v3 expert plan has 17 execution nodes: one initial source observation, then
eight nodes per carton (source navigation, pick, stow, remember, target
navigation, place, verify, and empty-arm return). These are not 17 physical
actions or public evaluation milestones.
Initial plate assignments are physically checked at reset. Pickup events require
physical grasp evidence as each distinct carton leaves its source region; merely
sliding or falling off the source plate does not latch a color. A held carton passing
over a target plate is not a placement. Both cartons must be released at their
matching targets, with no protocol violation, for the episode to succeed.

The ordered evaluator and action-only replay use the same physical events. The
public profile is `WR-08/report-v2`, in chronological order:

1. `one_item_picked`
2. `one_item_color_latched`
3. `one_item_matched`
4. `two_items_picked`
5. `two_items_color_latched`
6. `two_items_matched`
7. `success`

## Validation

Run the simulator-free contract, negative-case, and reporting tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q \
  tests/test_wr08_task.py tests/test_wr08_evaluation.py tests/test_wr08_manipulation.py \
  tests/evaluation/test_report_stages.py
```

In the RoboCasa environment, record a fresh episode using a new dataset ID:

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python scripts/record_entrypoint.py --task-id WR-08 --seeds 0 \
  --dataset-id wr08_v3_seed0 --output-root outputs/datasets --no-record-depth
```

Then replay its recorded action stream through a fresh production runtime:

```bash
PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python scripts/eval/replay_recorded_actions.py --task-id WR-08 \
  --dataset-root outputs/datasets/wr08_v3_seed0 \
  --output-run outputs/eval/wr08_v3_seed0 --seeds 0 --record-video
```

Inspect both the recording manifest and replay summary: process exit code alone
does not establish task success. Validate seed 0 and its three camera videos
first, expand to seeds 0–9, and then to 50 distinct seeds before promoting a
batch. Keep failed attempts, configuration hashes, and the tested source revision
with the validation report. Recordings made with the old single-asset revision
do not validate this task definition, even when their colors and stage count
match.

`--no-record-depth` controls storage, not necessarily rendering. The current
`rgbd_v3` observation profile still enables depth rendering; these RGB recordings
contain an empty depth group and no saved depth datasets.

For production action replay without a per-step replay video, the audited
`scripts/analysis/replay_wr08_actions_on_demand.py` wrapper renders policy-consumed
observations and the final frame at their original sample times. Full-render
A/B checks for both transport orders matched every physics step, consumed image,
action, diagnostic and public stage. The native recordings still retain all
three 20 Hz RGB videos.

`scripts/analysis/validate_wr08_campaign.py` combines exact seed coverage, native
HDF5/video decoding, full action replay and the seven public stages. When mixing
original and repaired expert batches, supply `--recording-provenance` so original
recordings retain their historical source archives. Do not count diagnostic
prefix probes or failed attempts as accepted recordings.
