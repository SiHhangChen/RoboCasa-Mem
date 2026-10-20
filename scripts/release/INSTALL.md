# Simulator setup

Use Python 3.11 with the source revisions recorded in
`configs/benchmark/simulator_environment.json`. The tested simulator uses
MuJoCo 3.3.1 and NumPy 2.2.5. Initialize WX-01 using the shipped setup
transcripts; MPlib is not an evaluation dependency.

```sh
git clone https://github.com/ARISE-Initiative/robosuite.git
git -C robosuite checkout 232ce7d4a6ed89c949a9aba024a05c8c32fdd08b
git clone https://github.com/robocasa/robocasa.git
git -C robocasa checkout 8f124f0e263de0c583ba5d2acc87137bfd905986
pip install -e ./robosuite -e ./robocasa
pip install 'mujoco==3.3.1' 'numpy==2.2.5'
pip install 'MobileMemBench[video]'
```

Restore the separately supplied, pinned `membench-sim-assets.tar.gz` into the
parent of the two source checkouts, then verify all 3035 files using
`scripts/release/verify_assets.py`. Archive identity, commands, existing
validation evidence, and installed acceptance are documented in
`docs/ASSETS.md`; the inventory is `configs/benchmark/simulator_assets.json`.
The upstream command `python -m robocasa.scripts.download_kitchen_assets` is
an alternative acquisition method; check its output against the inventory.

For a local candidate checkout, replace the last line with
`pip install -e '.[video]'`. For the built artifact, install its `.whl` file.
Public registry publication is a separate step; the candidate has not been
uploaded to a registry.

Headless NVIDIA rendering uses `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`.
Choose `MUJOCO_EGL_DEVICE_ID` for your machine. RoboCasa writes temporary
object XML files alongside its assets, so the asset directories must be
writable during simulation.

Run from any working directory:

```sh
membench-eval --endpoint ws://localhost:8000 --suite pi05:wx01
membench-eval --endpoint ws://localhost:8000 --suite act:wa01
membench-eval --endpoint ws://localhost:8000 --suite diffusion:wa01
membench-summarize /path/to/run
```

The suite family selector resolves a copied, tested YAML file without editing
it. Use the appropriate model server and checkpoint for the chosen family.
ACT and Diffusion server examples require the separate LeRobot environment;
OpenPI requires its own policy environment. See the configuration audit for
checkpoint and execution semantics.

The current wheel has passed acceptance in an independent Python environment
with pinned simulator sources and all 3035 assets restored from the current
archive. All 450 official task/seed inputs passed initialization, rendering,
and three hold-action steps. This uses the same host and GPU driver and does
not rerun model experiments. See `docs/ISOLATED_INSTALL.md` for the tested
dependency versions, provenance, and reproduction steps.
