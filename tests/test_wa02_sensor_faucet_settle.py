"""Coverage for WA-02 sensor-faucet zone dwell detection."""

from __future__ import annotations

import pytest

from bench.tasks.wa02 import environment as env_mod
from bench.tasks.wa02.config import load_task_config, resolve_action_config


class _StubSensorEnv:
    """Minimal stand-in exposing the sensor-zone counters used by dwell logic."""

    def __init__(self) -> None:
        self._sensor_faucet_layout: dict[str, object] = {"zone_settle_steps": 20}
        self._sensor_zone_entry_steps: dict[str, int] = {"hot": 0, "cold": 0}
        self._active_sensor_faucets: set[str] = set()
        self._sensor_cup_by_temperature: dict[str, str | None] = {
            "hot": None,
            "cold": None,
        }
        self._sensor_faucet_affinity_by_cup: dict[str, str] = {}
        self._in_zone = False

    def _restore_sensor_stream_geometry(self, temperature: str) -> dict[str, object]:
        return {"temperature": temperature, "restored": True}

    def _set_sensor_stream_visibility(self, temperature: str, *, visible: bool) -> None:
        del temperature, visible

    def _cup_is_in_sensor_zone(self, cup_id: str, temperature: str) -> bool:
        del cup_id, temperature
        return self._in_zone


def _make_env(threshold: int) -> _StubSensorEnv:
    stub = _StubSensorEnv()
    stub._sensor_faucet_layout = {"zone_settle_steps": threshold}
    cls = env_mod.WA02ServeHotColdWaterCups
    stub._sensor_zone_settle_steps = cls._sensor_zone_settle_steps.__get__(stub)
    stub._advance_sensor_zone = cls._advance_sensor_zone.__get__(stub)
    return stub


def test_faucet_activates_only_after_dwell_threshold() -> None:
    env = _make_env(threshold=20)
    env._in_zone = True
    for step in range(19):
        in_zone, activated = env._advance_sensor_zone("cup_A", "hot")
        assert in_zone is True
        assert activated is False
        assert "hot" not in env._active_sensor_faucets
        assert env._sensor_zone_entry_steps["hot"] == step + 1
    in_zone, activated = env._advance_sensor_zone("cup_A", "hot")
    assert in_zone is True
    assert activated is True
    assert "hot" in env._active_sensor_faucets
    assert env._sensor_cup_by_temperature["hot"] == "cup_A"


def test_fast_pass_through_zone_never_activates() -> None:
    env = _make_env(threshold=20)
    env._in_zone = True
    for _ in range(3):
        env._advance_sensor_zone("cup_A", "hot")
    env._in_zone = False
    in_zone, activated = env._advance_sensor_zone("cup_A", "hot")
    assert in_zone is False
    assert activated is False
    assert "hot" not in env._active_sensor_faucets
    assert env._sensor_zone_entry_steps["hot"] == 0


def test_leaving_zone_resets_dwell_before_reactivation() -> None:
    env = _make_env(threshold=10)
    env._in_zone = True
    for _ in range(8):
        env._advance_sensor_zone("cup_A", "cold")
    env._in_zone = False
    env._advance_sensor_zone("cup_A", "cold")
    env._in_zone = True
    # Dwell must start over from zero.
    for _ in range(9):
        _, activated = env._advance_sensor_zone("cup_A", "cold")
        assert activated is False
    _, activated = env._advance_sensor_zone("cup_A", "cold")
    assert activated is True


def test_zero_threshold_keeps_legacy_immediate_activation() -> None:
    env = _make_env(threshold=0)
    env._in_zone = True
    in_zone, activated = env._advance_sensor_zone("cup_A", "hot")
    assert in_zone is True
    assert activated is True


def test_dwell_is_per_temperature() -> None:
    env = _make_env(threshold=5)
    env._in_zone = True
    for _ in range(5):
        _, activated = env._advance_sensor_zone("cup_A", "hot")
    assert "hot" in env._active_sensor_faucets
    # Cold zone shares no state with hot.
    assert "cold" not in env._active_sensor_faucets
    assert env._sensor_zone_entry_steps["cold"] == 0


def test_affinity_blocks_neighboring_faucet_without_waiting() -> None:
    env = _make_env(threshold=2)
    env._sensor_faucet_affinity_by_cup["cup_A"] = "cold"
    env._in_zone = True

    for _ in range(4):
        in_zone, activated = env._advance_sensor_zone("cup_A", "hot")
        assert in_zone is False
        assert activated is False

    assert "hot" not in env._active_sensor_faucets
    assert env._sensor_zone_entry_steps["hot"] == 0

    for _ in range(2):
        in_zone, activated = env._advance_sensor_zone("cup_A", "cold")
    assert in_zone is True
    assert activated is True
    assert "cold" in env._active_sensor_faucets


def test_wa02_uses_short_fill_stage_settles_and_sensor_debounce() -> None:
    actions = resolve_action_config(config=load_task_config())

    assert actions.cup_pick_metadata["pregrasp_settle_steps"] == 30
    assert actions.cup_pick_metadata_by_cup["cup_A"][
        "grasp_approach_sector"
    ] == {
        "center_deg": -90.0,
        "half_width_deg": 90.0,
        "frame": "world",
        "angle_reference": "cup_center_to_pregrasp",
    }
    assert actions.cup_pick_metadata_by_cup["cup_B"][
        "grasp_approach_sector"
    ] == actions.cup_pick_metadata["grasp_approach_sector"]
    for metadata in actions.held_fill_metadata_by_temperature.values():
        assert metadata["entry_settle_steps"] == 1
        assert metadata["under_water_settle_steps"] == 1

    layout = env_mod._load_sensor_faucet_layout(None)
    assert layout["zone_settle_steps"] == 2


@pytest.mark.parametrize("bad", ("", "soon", -1, "x"))
def test_invalid_threshold_falls_back_to_zero(bad: object) -> None:
    env = _make_env(threshold=20)
    env._sensor_faucet_layout = {"zone_settle_steps": bad}
    assert env._sensor_zone_settle_steps() == 0
