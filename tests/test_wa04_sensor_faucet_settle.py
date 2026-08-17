"""WA-04 parity coverage for sensor-faucet dwell detection."""

from __future__ import annotations

from bench.tasks.wa04 import environment as env_mod
from bench.tasks.wa04.config import load_task_config, resolve_action_config


class _StubSensorEnv:
    def __init__(self, threshold: int) -> None:
        self._sensor_faucet_layout: dict[str, object] = {
            "zone_settle_steps": threshold
        }
        self._sensor_zone_entry_steps = {"hot": 0, "cold": 0}
        self._active_sensor_faucets: set[str] = set()
        self._sensor_cup_by_temperature = {"hot": None, "cold": None}
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
    env = _StubSensorEnv(threshold)
    cls = env_mod.WA04ServeUniqueTemperatureCupAmongThree
    env._sensor_zone_settle_steps = cls._sensor_zone_settle_steps.__get__(env)
    env._advance_sensor_zone = cls._advance_sensor_zone.__get__(env)
    return env


def test_wa04_faucet_requires_dwell_and_resets_after_exit() -> None:
    env = _make_env(2)
    env._in_zone = True
    assert env._advance_sensor_zone("cup_C", "cold") == (True, False)
    assert env._advance_sensor_zone("cup_C", "cold") == (True, True)
    assert env._sensor_cup_by_temperature["cold"] == "cup_C"

    env._in_zone = False
    assert env._advance_sensor_zone("cup_C", "cold") == (False, False)
    assert env._sensor_zone_entry_steps["cold"] == 0


def test_wa04_affinity_blocks_neighboring_faucet() -> None:
    env = _make_env(2)
    env._sensor_faucet_affinity_by_cup["cup_C"] = "cold"
    env._in_zone = True

    assert env._advance_sensor_zone("cup_C", "hot") == (False, False)
    assert env._advance_sensor_zone("cup_C", "hot") == (False, False)
    assert "hot" not in env._active_sensor_faucets

    assert env._advance_sensor_zone("cup_C", "cold") == (True, False)
    assert env._advance_sensor_zone("cup_C", "cold") == (True, True)


def test_wa04_uses_wa02_fill_and_service_settle_profile() -> None:
    actions = resolve_action_config(config=load_task_config())

    assert actions.cup_pick_metadata["pregrasp_settle_steps"] == 30
    for metadata in actions.held_fill_metadata_by_temperature.values():
        assert metadata["entry_settle_steps"] == 1
        assert metadata["under_water_settle_steps"] == 1
    for metadata in actions.service_place_metadata_by_slot.values():
        assert metadata["entry_settle_steps"] == 2
        assert metadata["place_settle_steps"] == 3
        assert metadata["clearance_settle_steps"] == 2
        assert metadata["retreat_settle_steps"] == 1
    assert actions.service_place_metadata_by_slot["slot_1"]["release_steps"] == 8
    assert actions.service_place_metadata_by_slot["slot_1"]["release_settle_steps"] == 2

    layout = env_mod._load_sensor_faucet_layout(None)
    assert layout["zone_settle_steps"] == 2
