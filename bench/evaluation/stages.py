"""Public stage-evaluation API.

The implementation lives in :mod:`bench.evaluation.stage_rollout`; this
module keeps the natural plural import path stable for task integrations.
"""

from .stage_rollout import *  # noqa: F401,F403
from .stage_rollout import __all__
