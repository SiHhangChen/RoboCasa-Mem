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
