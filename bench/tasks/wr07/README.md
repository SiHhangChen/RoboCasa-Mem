# WR-07

WR-07 is a single-item color-memory transport task. Three sugar plates are
placed on the coffee-machine side and three matching-color sugar plates are
placed on the opposite counter. The three colors are sampled without
replacement from the 20-color pool and both side assignments are deterministic
for the episode seed. The target assignment is a non-identity permutation, so
the matching plate is not inferred from its left/center/right position.

The expert observes the source plates, picks the milk carton from one source
plate, returns the arm to its regular holding pose, latches the source color,
places the carton on the matching target plate, verifies the placement, and
returns the arm to its regular empty pose. Source and target in-plate XY
offsets and yaw are seed-owned and bounded by the configured 4.5 cm radius.

The source pickup uses one shared calibrated ``counter_pick_station`` with five
candidate poses reused by the left, center, and right source zones. The target
side uses one calibrated 10 cm-diameter disk sampled at navigation time.
