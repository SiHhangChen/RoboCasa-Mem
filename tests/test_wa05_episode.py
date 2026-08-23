import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wa05.episode import resolve_episode


def test_expert_temperature_is_seed_balanced() -> None:
    values = [
        resolve_episode(EpisodeRequest(task_id="WA-05", seed=seed)).facts[
            "expert_fill_temperature"
        ]
        for seed in range(8)
    ]
    assert values == ["hot", "cold"] * 4


def test_prompt_does_not_reveal_episode_temperature_or_highlight() -> None:
    for seed in (0, 1):
        episode = resolve_episode(EpisodeRequest(task_id="WA-05", seed=seed))
        prompt = str(episode.public["prompt"]).lower()
        assert "highlight" not in prompt
        assert "高亮" not in prompt
        assert "expert" not in prompt
        assert "either hot or cold" in prompt


def test_episode_is_single_cup_single_slot() -> None:
    episode = resolve_episode(EpisodeRequest(task_id="WA-05", seed=3))
    assert episode.facts["cup_id"] == "cup_A"
    assert episode.facts["service_slot"] == "slot_1"
    assert episode.max_horizon == 4000


def test_cup_configs_use_wa01_scale_without_constructing_simulator() -> None:
    pytest.importorskip("mujoco")
    from bench.tasks.wa05.domain.geometry import (
        resolve_cup_region_constraints,
        resolve_cup_sampling_regions,
    )
    from bench.tasks.wa05.environment import (
        DEFAULT_CUP_REGION_CONSTRAINTS,
        DEFAULT_CUP_SAMPLING_REGIONS,
        WA05FillOneCupAndMatchTemperatureButton,
        WATER_CUP_OBJECT_SCALE,
    )

    # Exercise the task-owned object configuration without requiring a MuJoCo
    # renderer or a full kitchen reset.
    environment = WA05FillOneCupAndMatchTemperatureButton.__new__(
        WA05FillOneCupAndMatchTemperatureButton
    )
    environment._cup_counter_reset_region = {"offset": (0.0, 0.0, 0.0)}
    environment._cup_sampling_regions = resolve_cup_sampling_regions(
        DEFAULT_CUP_SAMPLING_REGIONS
    )
    environment._cup_region_constraints = resolve_cup_region_constraints(
        DEFAULT_CUP_REGION_CONSTRAINTS
    )
    environment.counter = SimpleNamespace(rot=0.0)

    configs = environment._get_obj_cfgs()

    assert configs
    assert all(
        config["object_scale"] == WATER_CUP_OBJECT_SCALE for config in configs
    )


def test_cup_grasp_candidates_have_wr03_style_body_offset() -> None:
    resource = (
        Path(__file__).parents[1]
        / "bench/tasks/wa05/resources/poses/water_cup_grasp_candidates.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))

    candidates = payload["candidates"]
    assert candidates
    assert all(
        candidate["eef_site_in_object"]["p"] == [0.005, 0.0, 0.018]
        for candidate in candidates
    )


def test_wa05_runtime_passes_wr03_style_gripper_physics() -> None:
    from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
    from bench.tasks.contracts import EpisodeRequest

    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-05", seed=0))
    api = resolve_builtin_task_plugin("WA-05").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)

    assert descriptor.environment.env_kwargs["gripper_force_limit"] == pytest.approx(20.0)
    assert descriptor.environment.env_kwargs["gripper_pad_friction"] == pytest.approx(
        (2.0, 0.05, 0.0001)
    )
