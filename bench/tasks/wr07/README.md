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

The task uses the fixed, stable `milk_1` carton geometry. Its appearance is
sampled from four texture-and-finish profiles: kraft-paper matte, light-green
satin, cool-blue glossy, and silver-blue foil. The green packaging region of
the original texture is recolored while its white artwork and text remain
intact. Thus each seed selects one of four clearly visible appearance variants
without changing the carton collision geometry.

The source pickup uses one shared calibrated ``counter_pick_station`` sampled
uniformly from a 10 cm-diameter disk, with a small calibrated yaw variation;
the sampled pose is reused by the observe and pick stages. The target side uses
one calibrated 10 cm-diameter disk sampled at navigation time as well.

The initial base XY is sampled uniformly in area from a 5 cm-radius disk
centered on the environment reset pose. Initial yaw is preserved. This draw
uses an independent seed salt and does not alter source/target station draws.
Initialization metadata records the center, sampled XY, offset and radius.

During held transport the main in-place turn is sampled around the calibrated
signed turn of -178.84 degrees with an independent uniform +/-5 degree offset.
The primitive keeps the clockwise branch, including draws slightly beyond
-180 degrees. The target translation then ends with a small yaw correction
when required.

Policy evaluation is separate from that expert workflow. Its report profile
is `WR-07/report-v4`: `item_picked` → `item_placed` →
`target_plate_matched` → `success`.

- Pickup requires closed-gripper grasp/contact and a lift of at least 2 cm
  from the initialized carton position; proximity or sliding alone is not pickup.
- Putdown requires release from the gripper and contact with a tray or counter
  for two consecutive control steps (0.1 s at 20 Hz), after an observed pickup.
- Matching requires contact with exactly one destination tray, whose color
  matches the source color. Like WA-01's counter-contact check, this uses
  actual simulator contacts, with no tray-center XY radius or carton-bottom
  height threshold. No carton-upright check, expert drop offset, yaw, or arm
  pose is required. Contact is the placement criterion, not full geometric
  containment inside the tray rim.
- A confirmed wrong destination tray earns putdown credit but immediately
  ends evaluation with `wrong_target_placement`; policies cannot correct it.
  Confirmation requires the same wrong destination for two consecutive
  control steps after pickup and release. Source/counter contacts alone are
  not wrong-destination failures.
- A confirmed matching placement immediately ends evaluation successfully.
  Thus the first confirmed placement on any of the three destination trays
  stops policy execution; color matching only decides success versus failure.
  Both outcomes emit `stop_execution`, which overrides `stop_on_success` and
  `ignore_done` in the rollout. The runtime rejects further actions until reset.
  Policy terminal results are frozen; there are no post-success actions or
  later placement checks. Expert recordings do not emit this hard-stop signal.

No expert color-memory latch, source-observation event, or expert trajectory
completion is required for policy success. Expert recordings still explicitly
latch the source color and obey the original semantic protocol, even when
recording an episode from the evaluation split.
