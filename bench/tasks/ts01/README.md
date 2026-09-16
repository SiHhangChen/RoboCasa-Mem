# TS01

This package directly owns TS01's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Package layout version: `membench.task_package/v1`.

TS01 keeps its seeded count policy: K is at most 3 and K <= N <= 5.
Sugar centers are sampled inside 5 mm radius disks around five inward-shifted
tray-local slots. Minimum center spacing is 5 cm for N <= 3 and 4.5 cm for
N = 4 or 5. Each cube independently samples world yaw in [-180, 180) degrees.

Expert arrivals at the sugar, saucepan, and completion-button stations sample
5 cm radius disks with +/-5 degree yaw variation using the episode seed.
The saucepan stays on Saucepan002 by default; saucepan, tray, and sugar
appearance profiles are sampled without changing physical parameters.

Sugar picks rank reachable grasp directions by trajectory quality and align
the measured finger-pad midpoint after preshaping. Available opening and
corrected grasp/lift paths are checked before descent. TS01 retains its own
evaluator and subtask scheduler.
