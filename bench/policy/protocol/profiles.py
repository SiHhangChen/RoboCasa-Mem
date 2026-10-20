"""Closed registry of policy runtime profiles supported by the wire protocol."""

from __future__ import annotations

from dataclasses import dataclass

from bench.runtime import data_contract as sim_contract

SIM_RUNTIME_PROFILE = "membench.robocasa_pandaomron.v2"


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    state_schema: str
    state_dim: int
    action_schema: str
    action_dim: int
    required_cameras: tuple[str, ...]
    image_size: tuple[int, int]
    control_frequency_hz: float


_PROFILES = {
    SIM_RUNTIME_PROFILE: RuntimeProfile(
        name=SIM_RUNTIME_PROFILE,
        state_schema=sim_contract.STATE_SCHEMA,
        state_dim=sim_contract.STATE_DIM,
        action_schema=sim_contract.ACTION_SCHEMA,
        action_dim=sim_contract.ACTION_DIM,
        required_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
        image_size=(224, 224),
        control_frequency_hz=20.0,
    ),
}


def get_runtime_profile(name: str) -> RuntimeProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unsupported runtime profile {name!r}; supported: {supported}") from exc


def runtime_profile_names() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


__all__ = [
    "SIM_RUNTIME_PROFILE",
    "RuntimeProfile",
    "get_runtime_profile",
    "runtime_profile_names",
]
