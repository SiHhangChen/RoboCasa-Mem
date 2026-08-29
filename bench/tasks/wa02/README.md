# WA-02

This package directly owns WA-02's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Water is dispensed by two task-owned touchless faucets above the sink. The
robot-left faucet is hot (red), the robot-right faucet is cold (blue), and each
stream is active only while a cup is inside that faucet's proximity zone.

The cups have no seed-assigned temperature identity during evaluation. The
temperature of each cup is bound from the faucet the policy actually uses, and
either hot-first or cold-first execution is valid as long as the remaining cup
receives the other temperature. The final requested cup is derived from those
physically observed fill events. Seed-resolved cup, service-slot, and hot-first
assignments exist only to make expert demonstrations deterministic.

Package layout version: `membench.task_package/v1`.
