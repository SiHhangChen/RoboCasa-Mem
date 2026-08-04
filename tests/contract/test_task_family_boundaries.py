from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.contract
def test_families_do_not_import_leaf_tasks_or_hold_resources() -> None:
    root = Path("bench/tasks/families")
    for source in root.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "bench.tasks.new_" not in text
        assert "bench.tasks.tm" not in text
        assert "bench.tasks.ts" not in text
        assert "bench.tasks.wa" not in text
        assert "bench.tasks.wr" not in text
        assert "bench.tasks.wx" not in text
        assert "/resources/" not in text


@pytest.mark.contract
def test_family_contracts_are_parameterized() -> None:
    from bench.tasks.families.cabinet_memory import CabinetMemorySpec, MemoryRelation
    from bench.tasks.families.drawer_inspection import DrawerInspectionSpec, InspectionState
    from bench.tasks.families.interleaved_task import EventGraph
    from bench.tasks.families.shelf_return import OriginalLevels, ShelfReturnSpec
    from bench.tasks.families.sugar_count import SugarCountSpec, target_names

    assert target_names(5, 2) == ("item_0", "item_1")
    assert InspectionState().complete(DrawerInspectionSpec(("a", "b"), 2)) is False
    assert MemoryRelation((("can", "upper"),)).level_for("can") == "upper"
    assert OriginalLevels((("mug", "middle"),)).level_for("mug") == "middle"
    assert ShelfReturnSpec(("mug",), ("middle",))
    assert EventGraph(("a", "b"), (("a", "b"),)).legal_next({"a"}) == ("b",)
    assert CabinetMemorySpec(("can",), ("upper",))
    assert SugarCountSpec(2, 3)
