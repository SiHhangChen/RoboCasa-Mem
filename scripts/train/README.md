# Training Scripts

MemBench-owned training entry points are grouped by policy:

- `pi05/`: task launchers and the shared OpenPI MemBench wrapper.
- `gr00t/`: GR00T launchers and the LeRobot `accelerate` entry point.
- `act/`: ACT dataset validation and training launcher.
- `memer/`: MemER Qwen/LoRA and pi0.5 launchers, including the GPU queue.

Run commands from the repository root. For example:

```bash
scripts/train/pi05/train_pi05_ts03.sh --resume
bash scripts/train/gr00t/train_gr00t_n1_7_wa01.sh
PYTHONPATH=. python scripts/train/act/train_act_wx01.py --dry-run
scripts/train/memer/train_memer_pi05.sh
```

Training logs belong under `outputs/train/<policy>/`. Launchers that manage
their own logs use this location by default. When redirecting a launcher
manually, follow the same convention:

```bash
scripts/train/pi05/train_pi05_ts03.sh \
  > outputs/train/pi05/ts03_$(date +%Y%m%d_%H%M%S).log 2>&1
```

Third-party training implementations remain in their policy source trees. For
example, `policy/openpi/scripts/train.py` is invoked by the MemBench wrappers
but is not a project launcher and therefore stays with OpenPI.

## Joint GR00T training for the 15-task release

The release dataset list and immutable metadata hashes live in
`configs/train/gr00t/release15_v1.json`. It covers exactly these tasks:

```text
TM01 TM02 TS01 TS02 TS03 TS04 WA01 WA02 WA03 WA05 WR01 WR03 WR07 WR08 WX01
```

LeRobot 0.6.1 does not enable its multi-dataset loader in the training factory,
so create one logical aggregate dataset before training. The preparation tool validates
all 15 sources, applies the approved descriptive metadata normalization for
TS01, TS03, and WX01 in temporary symlink views, and leaves the source datasets
unchanged. Videos are hard-linked on the same filesystem instead of copied. Its
data-file step uses Arrow-native index remapping to preserve the camera-matrix
extension types that LeRobot 0.6.1 otherwise loses through its pandas path.

```bash
conda activate membench

# Fast source/contract validation. No data is written.
python scripts/train/gr00t/prepare_joint_dataset.py check

# Builds data/membench_release15_v1 using about 2.5 GiB of new data blocks.
python scripts/train/gr00t/prepare_joint_dataset.py aggregate

# Full aggregate validation, including one decoded sample per release task.
python scripts/train/gr00t/prepare_joint_dataset.py validate --decode-samples

# Validate model, data, disk, and the eight visible GPUs without starting a run.
bash scripts/train/gr00t/train_gr00t_n1_7_release15.sh --check-only

# Start the 500k-step baseline from the clean N1.7-3B checkpoint.
bash scripts/train/gr00t/train_gr00t_n1_7_release15.sh

# Recover an interrupted run. Keep NUM_GPUS and BATCH_SIZE unchanged for exact sampling order.
bash scripts/train/gr00t/train_gr00t_n1_7_release15.sh --resume 250000
```

The launcher defaults to 8 GPUs, batch size 8 per GPU, LoRA rank 32, learning
rate `2e-4`, and frame-uniform sampling. With a 40-step action horizon, the
aggregate has 5,406,856 valid frame starts. A 500k-step run at global batch 64
is about 5.9 aggregate epochs. A 750k schedule is about 8.9 aggregate epochs,
but it should be configured from the beginning as a separate run rather than
extending a completed 500k cosine schedule.
