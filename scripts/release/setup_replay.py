"""Build-script compatibility imports for the shared setup replay reader."""

from bench.runtime.setup_replay import (
    SCHEMA,
    apply_transcript,
    config_identity,
    observation_digests,
    prepare,
    replay_setup,
)

__all__ = [
    "SCHEMA",
    "apply_transcript",
    "config_identity",
    "observation_digests",
    "prepare",
    "replay_setup",
]
