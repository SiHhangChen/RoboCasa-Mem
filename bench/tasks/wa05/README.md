# WA-05

This package directly owns WA-05's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Water is dispensed by two task-owned touchless faucets above the sink. The
robot-left faucet is hot (red), the robot-right faucet is cold (blue), and each
stream is active only while a cup is inside that faucet's proximity zone.

Each episode contains one cup and one fixed service slot. The policy may choose
either faucet, must stage the filled cup, and then physically press the compact
red or blue answer button matching the water actually dispensed. No faucet or
button is highlighted, and the prompt does not reveal the expert's choice.

Expert recording retains the original placement constraints: the cup center must
be inside the 12 x 12 cm service rectangle and within 12 cm of its target, with
the cup upright, supported by the counter, and released. This also applies when
recording expert prefixes for evaluation-split episodes.

Policy evaluation accepts upright, released placement anywhere on the task's
island counter. It requires counter contact, cup tilt at most 15 degrees, and
the right end effector more than 12 cm from the cup center. Rectangle membership
and distance to the service target do not affect evaluation staging. Canonical
evaluation episodes select this rule via `split="evaluation"`; attaching the
expert runtime restores the stricter recording rule. Diagnostics identify the
active rule in `service.placement_rule`.

Package layout version: `membench.task_package/v1`.
