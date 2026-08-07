"""Single policy communication and model-integration boundary.

The package initializer deliberately avoids importing the client or server so
protocol-only consumers never pull evaluation or websocket dependencies into
their import graph.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "PolicyClient":
        from .client import PolicyClient

        return PolicyClient
    if name in {"PolicyServer", "WebsocketPolicyServer"}:
        from .server import WebsocketPolicyServer

        return WebsocketPolicyServer
    raise AttributeError(name)


__all__ = ["PolicyClient", "PolicyServer", "WebsocketPolicyServer"]
