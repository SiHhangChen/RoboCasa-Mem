"""Policy-neutral client facade."""

from bench.evaluation.client import RemotePolicyClient


class PolicyClient(RemotePolicyClient):
    """Stable policy boundary; transport details remain behind the facade."""


__all__ = ["PolicyClient"]
