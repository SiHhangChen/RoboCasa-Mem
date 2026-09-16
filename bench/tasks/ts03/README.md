# TS03

This package directly owns TS03's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Package layout version: `membench.task_package/v1`.

Sugar cube centers are sampled uniformly inside 1 cm diameter disks around the
five tray-local grasp-safe slot centers, with tray bounds and a 5 cm minimum
center spacing enforced by rejection sampling. Each cube independently samples world
yaw in `[-180, 180)` degrees without changing its reset-time roll/pitch. XY and
yaw use isolated episode seed streams. Settings live in
`resources/task.yaml` under `initial_state.objects`.

Expert station arrivals use three episode-seeded station samplers. Each target
is sampled uniformly inside a 10 cm diameter disk around its calibrated center,
with yaw sampled within +/-5 degrees of the calibrated heading. The saucepan
mesh stays on Saucepan002; only the saucepan, sugar-plate, and sugar-cube
appearance profiles are sampled per episode and applied without changing
collision or physical parameters.
