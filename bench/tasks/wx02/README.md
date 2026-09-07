# WX-02

WX-02 starts at the dining table with two cups, each covering an irregular
visual stain. The robot fetches a sponge from a nearby yellow tool region,
stages it beside the cups, and completes two short memory loops. In each loop it
moves one cup to an independently sampled temporary target, wipes the revealed
stain, releases the sponge, and returns that cup to its episode-specific origin.
It then picks up the sponge, navigates back to the yellow region, and returns it.

The temporary cup targets are sampled in world coordinates independently of the
remembered origins, so they do not reveal where a cup started. Both stains are
non-colliding and fade only from verified contact-and-sweep evidence.

Package layout version: `membench.task_package/v1`.
