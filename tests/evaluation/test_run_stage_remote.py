from __future__ import annotations

import pytest

from scripts.eval.run_stage_remote import _select_seed_shard


def test_select_seed_shards_are_disjoint_and_complete() -> None:
    seeds = tuple(range(1050, 1100))
    shards = tuple(
        _select_seed_shard(seeds, index=index, count=3)
        for index in range(3)
    )

    assert set(shards[0]).isdisjoint(shards[1])
    assert set(shards[0]).isdisjoint(shards[2])
    assert set(shards[1]).isdisjoint(shards[2])
    assert sorted(seed for shard in shards for seed in shard) == list(seeds)


@pytest.mark.parametrize(
    ("index", "count"),
    ((None, 3), (0, None), (-1, 3), (3, 3), (0, 0)),
)
def test_select_seed_shard_rejects_invalid_configuration(
    index: int | None,
    count: int | None,
) -> None:
    with pytest.raises(ValueError):
        _select_seed_shard((1050, 1051), index=index, count=count)
