from __future__ import annotations

from types import SimpleNamespace

from bench.execution.planning import PickPlaceConfig
from bench.tasks.ts03.execution import geometry


def _selector(monkeypatch, outcomes, *, preferred_yaw: float = 0.0):
    generated = []

    def make_payload(env, obs, **kwargs):
        generated.append(kwargs)
        return {"yaw": kwargs["grasp_yaw_offset_deg"]}

    def dry_run(payload, *, object_name):
        passed, lengths = outcomes[payload["yaw"]]
        return {
            "pass": passed,
            "stages": [
                {"stage": name, "pass": True, "plan": {"trajectory_len": length}}
                for name, length in zip(("pregrasp", "grasp", "lift"), lengths)
            ],
        }

    monkeypatch.setattr(geometry, "make_object_pick_target_poses", make_payload)
    monkeypatch.setattr(
        geometry, "preferred_tray_grasp_yaw_offset_deg", lambda env, name: preferred_yaw
    )
    request = {
        "env": SimpleNamespace(sugar_names=("sugar0", "sugar2")),
        "obs": {},
        "object_name": "sugar2",
        "pick_place_config": PickPlaceConfig(),
        "lift_height": 0.24,
        "kwargs": {"grasp_yaw_offsets_deg": [0, 90, 180, 270]},
        "dry_run_pick_sequence": dry_run,
    }
    return geometry.build_pick_payload_selector(), request, generated


def test_pick_selects_shortest_reachable_yaw_not_tray_side_preference(monkeypatch):
    # Seed-1 recorded lengths: preference 0° had the longest plan, 180° the shortest.
    selector, request, generated = _selector(
        monkeypatch,
        {
            0.0: (True, [32, 14, 15]),
            90.0: (True, [20, 16, 17]),
            180.0: (True, [17, 14, 15]),
            270.0: (True, [26, 16, 15]),
        },
        preferred_yaw=0.0,
    )
    result = selector(request)
    assert result["selection"]["mode"] == "four_direction_best_plan"
    assert result["selection"]["selected_yaw_offset_deg"] == 180.0
    assert result["payload"] == {"yaw": 180.0}
    assert [entry["grasp_yaw_offset_deg"] for entry in generated] == [0.0, 90.0, 180.0, 270.0]


def test_pick_uses_tray_side_preference_only_to_break_equal_plans(monkeypatch):
    selector, request, _ = _selector(
        monkeypatch,
        {
            0.0: (True, [20, 10, 10]),
            90.0: (True, [20, 10, 10]),
            180.0: (True, [20, 10, 10]),
            270.0: (True, [20, 10, 10]),
        },
        preferred_yaw=270.0,
    )
    result = selector(request)
    assert result["selection"]["selected_yaw_offset_deg"] == 270.0
