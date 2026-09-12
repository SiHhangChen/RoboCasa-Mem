# WR-01

This package directly owns WR-01's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Package layout version: `membench.task_package/v1`.

Policy evaluation counts one closed door and then both closed doors, in either
order. The two cans must be verified in the cabinets, and both doors must have
been closed together before retrieval. After that memory barrier, opening the
target cabinet qualifies regardless of which door was opened first or whether
both doors are open. Opening the other cabinet is diagnostic only. Retrieving
the requested can while the other can stays stored and placing the target
anywhere on the task counter are still required for success. Final placement
requires physical counter contact and no right-gripper contact; it does not
require proximity to the sampled table target, an upright pose, or a minimum
gripper retreat distance.

From initialization until terminal success, the non-target can must remain in
its initially assigned cabinet at every evaluated control step. The first
observed departure terminates the episode with `distractor_not_stored`; putting
the can back does not recover the episode. This uses the existing cabinet
containment check, not an exact-position or zero-motion requirement. Only a new
episode reset clears the failure.

The expert's sampled `close_order`, canonical neutral reset, and sampled table
placement point are recording choices, not policy-evaluation requirements.
