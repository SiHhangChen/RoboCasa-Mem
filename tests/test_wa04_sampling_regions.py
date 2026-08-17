from __future__ import annotations

import pytest

from bench.tasks import TaskCatalog, resolve_builtin_task_plugin
from bench.tasks.contracts import EpisodeRequest
from bench.tasks.wa04.config import load_task_config
from bench.tasks.wa04.domain.geometry import (
    resolve_cup_region_constraints,
    resolve_cup_sampling_regions,
    validate_cup_region_layout,
)


def _resolved_regions_and_constraints():
    config = load_task_config()
    objects = dict(config.initial_state.objects)
    regions = resolve_cup_sampling_regions(dict(objects["cups"]))
    constraints = resolve_cup_region_constraints(
        dict(objects["cup_region_constraints"])
    )
    return regions, constraints


def test_wa04_sampling_region_is_inset_from_legal_boundary() -> None:
    regions, constraints = _resolved_regions_and_constraints()

    assert constraints.sampling_inset_xy == pytest.approx((0.003, 0.003))
    legal = regions["cup_A"]
    sampled = legal.inset(constraints.sampling_inset_xy)

    assert legal.size_xy == pytest.approx((0.120, 0.120))
    assert legal.x_bounds == pytest.approx((0.461, 0.581))
    assert sampled.size_xy == pytest.approx((0.114, 0.114))
    assert sampled.x_bounds == pytest.approx((0.464, 0.578))
    assert sampled.y_bounds == pytest.approx((-0.527, -0.413))


def test_wa04_sampling_inset_cannot_consume_legal_region() -> None:
    regions, constraints = _resolved_regions_and_constraints()
    invalid = type(constraints)(
        sink_exterior_local_x_max=constraints.sink_exterior_local_x_max,
        minimum_sink_clearance=constraints.minimum_sink_clearance,
        minimum_inter_region_gap=constraints.minimum_inter_region_gap,
        sampling_inset_xy=(0.060, 0.003),
    )

    with pytest.raises(ValueError, match="invalid for cup_A"):
        validate_cup_region_layout(regions, invalid)


def test_wa04_runtime_descriptor_passes_sampling_inset_to_environment() -> None:
    episode = TaskCatalog.resolve(EpisodeRequest(task_id="WA-04", seed=0))
    api = resolve_builtin_task_plugin("WA-04").load_runtime_api()
    descriptor = api.build_descriptor(episode, False)

    assert descriptor.environment.env_kwargs["cup_region_constraints"][
        "sampling_inset_xy"
    ] == pytest.approx([0.003, 0.003])
