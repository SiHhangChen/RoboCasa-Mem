# WR-06 task architecture

WR-06 is the single-object simplification of WR-05. One real ketchup bottle
(`aigen_objs/ketchup/ketchup_3`) starts on the lower or upper level of an open
shelf and is staged on the left or right side of a fixed serving plate on the
dining island (both seed-decided, balancing the four WR-05 pairings once the
mustard is removed), and must be returned upright to its remembered shelf
level. The strict physical-event protocol (memory barrier after supported plate
staging, wrong-first-return rejection, terminal verification) follows WR-04.

```text
domain -> environment / physical_events / execution -> expert -> runtime
            ^                                            |
            +---------------- config -------------------+
```

- `domain/episode.py` decides the original level by seed parity and the staged
  plate side by placement-pair parity; neither appears in the public prompt.
- `physical_events.py` is WR-06's dwell-based single-pass state machine.
- `environment.py` owns the fixed plate (`objaverse/plate/plate_14`) at the
  legacy HS-WR-02 world position and the real ketchup bottle.
- `execution/handlers.py` applies the guarded supported-release place
  (descent early stop + pre-release gate) for both plate staging and shelf
  return.  A right-slot staging uses the WR-05-validated
  `ketchup_plate_right_station` base geometry, not a mustard-style station.
- `execution/manipulation.py` installs the WR-05-proven WA-02-style low
  side-grasp selector for the plate re-pick.  Bottle motion is generated solely
  by MuJoCo contacts; no kinematic attachment, release pinning, or post-release
  pose correction is used (mirroring the WR-05 physical-release contract).
- `environment.py` tunes the real bottle's tangential contact friction so the
  recorded grasp-and-carry action stream holds the bottle through contact
  dynamics, and adds geometric fallbacks to the plate/shelf contact checks so
  a released bottle settling on a flat level is not misread as leaving its
  staging/return spot.

Physical validation: full expert rollouts pass for all four
(seed-level, plate-side) combinations (seeds 0-3), including the strict
memory-barrier -> re-pick -> shelf-return -> terminal-verification protocol.

Package layout version: `membench.task_package/v1`.
