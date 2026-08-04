"""Template reset initialization; concrete tasks may use the shared reset only."""

from bench.runtime.contracts import InitializationReport

def initialize_episode(runtime, episode):
    del runtime, episode
    return InitializationReport.noop(reason="shared_runtime_reset_is_sufficient")
