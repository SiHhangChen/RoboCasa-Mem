# WR-05 task architecture

WR-05 is the real-condiment-bottle variant of WR-02: a ketchup bottle
(`aigen_objs/ketchup/ketchup_3`) and a mustard bottle
(`lightwheel/mustard/Mustard003`) start on the lower and upper levels of an
open shelf, are staged beside a fixed serving plate on the dining island
(plate-left / plate-right), and must be returned upright to their remembered
shelf levels. The simulator object names remain `ketchup` and `mustard`.

```text
domain -> environment / execution -> expert -> runtime
            ^                                  |
            +----------- config --------------+
```

- `domain/episode.py` owns the four balanced home/use bijections and the joint
  return-XY sampling; the private condiment-to-level binding stays out of the
  policy prompt.
- `environment.py` places the real bottles upright (identity quat) on the open
  shelf and fixes the serving plate (`objaverse/plate/plate_14`) at the legacy
  HS-WR-02 world position.
- `execution/handlers.py` applies the guarded supported-release place (descent
  early-stop + pre-release gate) for both plate staging and shelf return.
  Releases are physical only; no post-release pose correction or attachment
  assist is used.
- `execution/manipulation.py` installs the WA-02-style low side-grasp selector
  for the plate re-pick. Bottle motion is generated solely by MuJoCo contacts.

Package layout version: `membench.task_package/v1`.
