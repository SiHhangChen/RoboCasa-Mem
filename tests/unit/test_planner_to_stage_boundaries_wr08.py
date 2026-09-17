from __future__ import annotations

import pytest

from scripts.annotate.planner_to_stage_boundaries_wr08 import (
    _holding_navigation_steps,
)


def test_holding_navigation_counts_rotate_and_translate_once() -> None:
    details = {
        "steps": 132,
        "holding_navigation": {
            "phases": [
                {"phase": "rotate", "steps": 141},
                {"phase": "translate", "steps": 132},
            ]
        },
    }
    assert _holding_navigation_steps(details, path="stage") == (
        273,
        {"rotate_steps": 141, "translate_steps": 132},
    )


def test_holding_navigation_rejects_nonduplicated_outer_count() -> None:
    details = {
        "steps": 273,
        "holding_navigation": {
            "phases": [
                {"phase": "rotate", "steps": 141},
                {"phase": "translate", "steps": 132},
            ]
        },
    }
    with pytest.raises(ValueError, match="outer steps"):
        _holding_navigation_steps(details, path="stage")
