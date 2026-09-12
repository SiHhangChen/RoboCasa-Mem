# WR-03

This package directly owns WR-03's canonical episode, RoboCasa scene, semantics, expert plan, runtime composition, and resources.

Package layout version: `membench.task_package/v1`.

Policy evaluation follows WR-01's single-can equivalent (`WR03-v4`):

- `close_first_door`: the can's storage assignment is known and either door is
  currently closed (open fraction <= 0.08).
- `close_second_door`: both doors are currently closed together. The memory
  barrier latches only while the can is still in its recorded cabinet; no
  neutral base/torso pose or close-event dwell is required.
- `open_remembered_cabinet`: after the memory barrier, the target door is open
  (fraction >= 0.70) for three consecutive control observations. Opening the
  empty cabinet first or leaving both doors open is allowed.
- `retrieve_can`: the same can is outside both cabinets for three consecutive
  control observations after opening the remembered cabinet. Grasp/lift
  evidence is diagnostic, not a scoring requirement.
- `place_on_table`: after retrieval, the can contacts the task counter and no
  longer contacts the right gripper. No sampled target point, placement region,
  upright angle, retreat distance, or placement dwell is required. Success
  latches at this point.

Stages qualify in prerequisite order; simultaneous closing of both doors can
qualify both closing stages in one observation. WR-03 has no second can, so
WR-01's non-target-can containment invariant does not apply.

The expert still uses the sampled `close_order`, neutral-reset motions, and
sampled table target for consistent recording. These do not constrain policy
evaluation. Reports use `WR-03/report-v2`: `one_door_closed`, `both_doors_closed`,
`correct_cabinet_opened`, and `success` (which includes retrieval and placement).
Stage probes replay through `close_second_door` before testing opening; dataset
prefixes use `wr03_dataset_id5_v2` and must be regenerated, not reused from v1.
