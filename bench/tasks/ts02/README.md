# TS02

TS02 places one high-saturation red cube in one of three closed drawers. The
agent searches without revisiting drawers, closes every inspected drawer, stops
when the cube is found, and then presses the spring-loaded completion button
exactly once per inspected drawer. The target location and expert search order
are deterministically sampled from separate seed streams.

This package directly owns TS02's canonical episode, RoboCasa scene, physical
edge observer, search-and-count semantics, expert plan, runtime composition,
and resources.

Package layout version: `membench.task_package/v1`.
