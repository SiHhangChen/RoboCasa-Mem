# TS04

TS04 places one high-saturation 10 cm red cube in one of its two drawers. The
cube location is deterministic per seed. The agent searches without revisiting,
closes every inspected drawer, stops immediately after finding and closing the
target drawer, and then presses the completion button once per inspected drawer.

This package directly owns TS04's canonical episode, RoboCasa scene, semantics,
expert plan, runtime composition, and resources.

Stage-conditioned evaluation uses the same handoff protocol as TS02, with TS04's
own two-drawer evaluator and actual search length (one or two button presses):

| Scored target | Expert-prefix handoff |
| --- | --- |
| `one_drawer_checked` | Initial state; no expert prefix |
| `target_found`, `search_stopped_after_target` | Before checking the target drawer; after the previous drawer's arm return, or the initial state if the target is first |
| `button_pressed`, `correct_press_count` | After closing the target drawer and the expert's arm return, before navigating to the button |

`before_target_drawer` and `target_drawer_returned` are expert-only prefix
boundaries, not scored stages or policy pose requirements. The five public
stages and existing success rules are unchanged: `correct_press_count` is the
final verdict, and `search_stopped_after_target` still qualifies when the target
drawer is closed and can be revoked by later search. No extra stop-observation
window is introduced. Default stage budgets match TS02 and can be overridden by
a stage suite; they are not calibrated TS04 expert-prefix budgets.

Package layout version: `membench.task_package/v1`.
