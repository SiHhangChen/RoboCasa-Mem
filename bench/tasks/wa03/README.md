# WA-03

WA-03 extends WA-01 to three identical coffee cups. The robot adds sugar to any
one cup, transports all three cups into parallel dining regions, and must finally
select the cup it sugared. The default task exposes no target marker; an explicit
`sugar_target_mode: marked` option remains available for compatibility.

This package directly owns WA-03's canonical episode, RoboCasa scene, semantics,
expert plan, runtime composition, and resources. It reuses WA-01's calibrated
coffee-machine and dining-table base stations through task-local copied resources.

Package layout version: `membench.task_package/v1`.
