"""Regression: activating a proximity sensor faucet must intercept the stream.

Previously only the expert execution path truncated the water stream to the cup
rim via ``set_water_stream_intercept``.  The recorded-action / learned-policy
path that auto-ramps fill progress through ``_refresh_sensor_faucets`` only made
the full-length stream visible, so during replay / evaluation the stream stayed
at its full 40+ cm length and visibly passed through the cup bottom.

This pins the new behaviour: ``_refresh_sensor_faucets`` truncates the stream
to the dwelling cup the moment the faucet settles, keeps it intercepted while
that same cup stays in the zone (no repeated forward), and restores it once the
cup leaves.
"""

from __future__ import annotations

from typing import Any

import pytest

from bench.tasks.wa02 import environment as wa02_mod
from bench.tasks.wa04 import environment as wa04_mod
from bench.tasks.wa05 import environment as wa05_mod

TASKS = [
    (wa02_mod, wa02_mod.WA02ServeHotColdWaterCups),
    (wa04_mod, wa04_mod.WA04ServeUniqueTemperatureCupAmongThree),
    (wa05_mod, wa05_mod.WA05FillOneCupAndMatchTemperatureButton),
]

TEMPERATURES = ("hot", "cold")


class _StubSensorEnv:
    def __init__(self, threshold: int) -> None:
        self._sensor_faucet_layout: dict[str, Any] = {
            "zone_settle_steps": threshold
        }
        self._sensor_zone_entry_steps = {
            temperature: 0 for temperature in TEMPERATURES
        }
        self._active_sensor_faucets: set[str] = set()
        self._sensor_cup_by_temperature: dict[str, str | None] = {
            temperature: None for temperature in TEMPERATURES
        }
        self._sensor_faucet_affinity_by_cup: dict[str, str] = {}
        self._sensor_stream_intercepted_for: dict[str, str | None] = {
            temperature: None for temperature in TEMPERATURES
        }
        self._in_zone_for: dict[tuple[str, str], bool] = {}
        self.visibility_log: list[tuple[str, bool]] = []
        self.restore_log: list[str] = []
        self.intercept_calls: list[tuple[str, str]] = []

    def _cup_is_in_sensor_zone(self, cup_id: str, temperature: str) -> bool:
        return bool(self._in_zone_for.get((cup_id, temperature), False))

    def _set_sensor_stream_visibility(self, temperature: str, *, visible: bool) -> None:
        self.visibility_log.append((temperature, visible))

    def _restore_sensor_stream_geometry(self, temperature: str) -> dict[str, Any]:
        self.restore_log.append(temperature)
        self._sensor_stream_intercepted_for[temperature] = None
        return {"temperature": temperature, "restored": True}

    def set_water_stream_intercept(self, cup_id: str, temperature: str) -> dict[str, Any]:
        self.intercept_calls.append((cup_id, temperature))
        self._sensor_stream_intercepted_for[temperature] = cup_id
        return {"intercepted": True, "cup_id": cup_id, "temperature": temperature}


def _make_env(cls: Any, threshold: int = 2) -> _StubSensorEnv:
    env = _StubSensorEnv(threshold)
    env._sensor_zone_settle_steps = cls._sensor_zone_settle_steps.__get__(env)
    env._refresh_sensor_faucets = cls._refresh_sensor_faucets.__get__(env)
    return env


@pytest.mark.parametrize(
    "cls",
    [entry[1] for entry in TASKS],
    ids=[entry[0].__name__.split(".")[-1] for entry in TASKS],
)
def test_faucet_activate_intercepts_stream_once(cls: Any) -> None:
    env = _make_env(cls)
    env._in_zone_for[("cup_A", "cold")] = True

    env._refresh_sensor_faucets()  # step 1: dwell not yet satisfied
    assert env.intercept_calls == []
    assert "cold" not in env._active_sensor_faucets

    env._refresh_sensor_faucets()  # step 2: settles -> activate + intercept
    assert "cold" in env._active_sensor_faucets
    assert env._sensor_cup_by_temperature["cold"] == "cup_A"
    assert env.intercept_calls == [("cup_A", "cold")]
    assert ("cold", True) in env.visibility_log

    env._refresh_sensor_faucets()  # step 3+: same cup stays -> no repeat forward
    assert env.intercept_calls == [("cup_A", "cold")]


@pytest.mark.parametrize(
    "cls",
    [entry[1] for entry in TASKS],
    ids=[entry[0].__name__.split(".")[-1] for entry in TASKS],
)
def test_faucet_restores_stream_after_cup_exits(cls: Any) -> None:
    env = _make_env(cls)
    env._in_zone_for[("cup_A", "hot")] = True
    env._refresh_sensor_faucets()
    env._refresh_sensor_faucets()
    assert "hot" in env._active_sensor_faucets
    assert env.intercept_calls == [("cup_A", "hot")]

    env._in_zone_for[("cup_A", "hot")] = False
    env._refresh_sensor_faucets()
    assert "hot" not in env._active_sensor_faucets
    assert "hot" in env.restore_log
    assert ("hot", False) in env.visibility_log
    assert env._sensor_stream_intercepted_for["hot"] is None