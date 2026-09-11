# WA-03

WA-03 extends WA-01 to three identical coffee cups. The robot adds sugar to any
one cup, transports all three cups to the dining counter, and must finally select
the cup it sugared. Placement regions may guide expert recording, but are not a
policy-evaluation requirement. The default task exposes no target marker; an explicit
`sugar_target_mode: marked` option remains available for compatibility.

The public evaluation funnel is
`sugar_added -> one_cup_ready -> two_cups_ready -> three_cups_ready -> select -> success`.
`select` records any valid final re-grasp; `success` reuses that selection event
in the same control step and only passes when the selected cup is the one that
actually received sugar.

This package directly owns WA-03's canonical episode, RoboCasa scene, semantics,
expert plan, runtime composition, and resources. It reuses WA-01's calibrated
coffee-machine and dining-table base stations through task-local copied resources.

Package layout version: `membench.task_package/v1`.
