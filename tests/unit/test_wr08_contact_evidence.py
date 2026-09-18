"""Contact-index equivalence without requiring a simulator installation."""

from types import SimpleNamespace

import pytest

from bench.tasks.wr08.contact_evidence import ContactIndex


@pytest.mark.parametrize("pairs", ([], [(0, 1)], [(1, 0)], [(0, 0)], [(0, 1), (0, 2)]))
def test_index_matches_bidirectional_contact_scan(pairs):
    names = {"box": 0, "plate": 1, "finger": 2, "other": 3}
    lookups = []
    sim = SimpleNamespace(
        model=SimpleNamespace(geom_name2id=lambda name: lookups.append(name) or names[name]),
        data=SimpleNamespace(
            ncon=len(pairs),
            contact=[SimpleNamespace(geom1=a, geom2=b) for a, b in pairs]
            + [SimpleNamespace(geom1=0, geom2=3)],
        ),
    )
    ids = {}
    index = ContactIndex(sim, ids)
    for first in (["box"], ["plate"], ["box", "finger"], []):
        for second in (["plate"], ["box"], ["other"], [], None):
            expected = any(
                (
                    a in {names[n] for n in first}
                    and (second is None or b in {names[n] for n in second})
                )
                or (
                    b in {names[n] for n in first}
                    and (second is None or a in {names[n] for n in second})
                )
                for a, b in pairs
            )
            assert index.check(SimpleNamespace(contact_geoms=first), second) is expected
    assert len(lookups) == len(ids)
    assert index.check("box", "other") is False
    # A fresh state rescans contacts but retains static name-to-ID resolution.
    sim.data.contact[0] = SimpleNamespace(geom1=0, geom2=3)
    sim.data.ncon = 1
    assert ContactIndex(sim, ids).check("box", "other")
    assert len(lookups) == len(ids)
