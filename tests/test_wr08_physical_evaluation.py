"""Read-only two-carton physics checks and policy/recording separation."""

from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.wr08.domain.contracts import ITEM_IDS, SOURCE_SLOTS, TARGET_SLOTS
from bench.tasks.wr08.domain.episode import resolve_episode_plan
from bench.tasks.wr08.domain.semantics import SemanticState, record_initial_sources


@pytest.fixture
def physical_env(monkeypatch):
    from bench.execution import planning
    from bench.execution.manipulation import object_interactions
    from bench.tasks.wr08.environment import WR08MatchMilkCartonsToColoredSugarPlates, OU

    plan = resolve_episode_plan(seed=0)
    env = object.__new__(WR08MatchMilkCartonsToColoredSugarPlates)
    env.require_expert_protocol = False
    env._wr08_evaluator = None
    env._physical_observer = None
    env._physical_event_diagnostics = {}
    env._semantic_state = SemanticState()
    record_initial_sources(env._semantic_state, plan.source_slot_by_item, plan.source_color_by_item)
    for name in (
        "source_slot_by_item", "source_color_by_item", "target_slot_by_color",
        "source_plate_color_by_slot", "target_plate_color_by_slot",
    ):
        setattr(env, name, dict(getattr(plan, name)))
    env._transport_order = plan.transport_order
    env.target_item_offset_xy_by_item = {item: np.zeros(2) for item in ITEM_IDS}
    env.plate_threshold = 0.16
    env.physical_event_dwell_steps = 3
    env.terminal_verification_steps = 1
    env.timestep = 0
    names = [*ITEM_IDS]
    names += [env.source_plate_name(slot) for slot in SOURCE_SLOTS]
    names += [env.target_plate_name(slot) for slot in TARGET_SLOTS]
    positions = np.array([
        [0.0, 0.0, 0.98], [0.0, 0.0, 0.98],
        [-3.0, 0.0, 0.93], [-2.0, 0.0, 0.93], [-1.0, 0.0, 0.93],
        [1.0, 0.0, 0.93], [2.0, 0.0, 0.93], [3.0, 0.0, 0.93],
    ])
    env.obj_body_id = {name: index for index, name in enumerate(names)}
    env.objects = {name: name for name in names}
    env.sim = SimpleNamespace(data=SimpleNamespace(body_xpos=positions))
    env._source_plate_centers = {
        slot: env._obj_pos(env.source_plate_name(slot)).copy() for slot in SOURCE_SLOTS
    }
    env._target_plate_centers = {
        slot: env._obj_pos(env.target_plate_name(slot)).copy() for slot in TARGET_SLOTS
    }
    env._source_surface_z = env._target_surface_z = 0.93
    for item in ITEM_IDS:
        positions[env.obj_body_id[item], :2] = env._source_plate_centers[plan.source_slot_by_item[item]][:2]
    env._evaluation_initial_item_positions = {item: env._obj_pos(item).copy() for item in ITEM_IDS}
    state = SimpleNamespace(
        qpos=np.array([0.04, 0.04]), held=set(), contact_known=True, finite=True,
        plate_contacts={item: set() for item in ITEM_IDS}, counter_contacts=set(),
    )
    env.robots = [SimpleNamespace(get_gripper_joint_positions=lambda arm: state.qpos)]
    env.check_contact = lambda item, plate: plate in state.plate_contacts[item]
    monkeypatch.setattr(env, "_object_collision_diagnostics", lambda: {
        item: [{"enabled": True}] for item in ITEM_IDS
    })
    monkeypatch.setattr(object_interactions, "get_object_hold_status", lambda env, item, **kw: {
        "held": item in state.held and state.finite,
        "contact_and_grasp": {"contact": item in state.held} if state.contact_known else {},
        "checks": {"finite_pose": state.finite, "grasp_or_contact": item in state.held},
    })
    monkeypatch.setattr(OU, "check_obj_any_counter_contact", lambda env, item: item in state.counter_contacts)
    monkeypatch.setattr(OU, "gripper_obj_far", lambda env, item, **kw: item not in state.held)

    def unexpected_geometry(*args, **kwargs):
        raise AssertionError("Policy placement must depend on actual contact")

    monkeypatch.setattr(planning, "object_reference_pos", unexpected_geometry)
    monkeypatch.setattr(OU, "check_obj_in_receptacle", unexpected_geometry)
    return env, state


def _pick(env, state, item):
    state.qpos[:] = 0.02
    state.held.add(item)
    state.plate_contacts[item].clear()
    env.sim.data.body_xpos[env.obj_body_id[item], 2] += 0.03


def _place(env, state, item, slot):
    name = env.target_plate_name(slot)
    env.sim.data.body_xpos[env.obj_body_id[item]] = env._obj_pos(name) + np.array([0.0, 0.0, 0.05])
    state.plate_contacts[item] = {name}
    state.held.discard(item)
    state.qpos[:] = 0.04


@pytest.mark.parametrize("item", ITEM_IDS)
@pytest.mark.parametrize(
    "held,gripper,lift,lateral,expected",
    [(True, 0.02, 0.03, 0.0, True), (False, 0.02, 0.03, 0.0, False),
     (True, 0.04, 0.03, 0.0, False), (True, 0.02, 0.01, 0.0, False),
     (True, 0.02, 0.0, 0.10, False)],
)
def test_each_carton_requires_closed_gripper_contact_and_lift(
    physical_env, item, held, gripper, lift, lateral, expected,
):
    env, state = physical_env
    state.held = {item} if held else set()
    state.qpos[:] = gripper
    env.sim.data.body_xpos[env.obj_body_id[item]] += [lateral, 0.0, lift]
    assert env._policy_grasp_evidence(item)["picked_up"] is expected


@pytest.mark.parametrize("fault", ("unknown_contact", "nonfinite_pose", "invalid_gripper"))
def test_unavailable_grasp_evidence_cannot_prove_release(physical_env, fault):
    env, state = physical_env
    if fault == "unknown_contact":
        state.contact_known = False
    elif fault == "nonfinite_pose":
        state.finite = False
    else:
        state.qpos[:] = np.nan
    for item in ITEM_IDS:
        evidence = env._policy_grasp_evidence(item)
        assert not evidence["picked_up"] and not evidence["released"]


@pytest.mark.parametrize("item", ITEM_IDS)
@pytest.mark.parametrize("slot", TARGET_SLOTS)
def test_all_target_plates_use_contact_without_geometry_filters(physical_env, item, slot):
    env, state = physical_env
    _place(env, state, item, slot)
    # Body position is deliberately outside the old 16 cm / bottom-height checks.
    env.sim.data.body_xpos[env.obj_body_id[item]] += [0.20, 0.0, 0.10]
    env.target_item_offset_xy_by_item[item] = np.array([-0.04, 0.0])
    assert env.detect_target_slot(item) is None
    snapshot = env.evaluation_snapshot()
    assert snapshot.item_placed_by_item[item]
    assert snapshot.target_slot_attempted_by_item[item] == slot
    assert snapshot.item_matched_by_item[item] is (slot == env.target_slot_for_item(item))
    assert not any(snapshot.memory_latched_by_item.values())


@pytest.mark.parametrize("case", ("airborne", "floor", "held", "no_contact"))
def test_xy_alignment_does_not_prove_placement(physical_env, case):
    env, state = physical_env
    for item in ITEM_IDS:
        _place(env, state, item, env.target_slot_for_item(item))
        if case == "held":
            state.held.add(item)
        else:
            state.plate_contacts[item].clear()
            if case == "airborne":
                env.sim.data.body_xpos[env.obj_body_id[item], 2] += 0.30
            elif case == "floor":
                env.sim.data.body_xpos[env.obj_body_id[item], 2] = 0.05
    snapshot = env.evaluation_snapshot()
    assert not any(snapshot.item_placed_by_item.values())
    assert not any(snapshot.item_matched_by_item.values())
    assert not snapshot.target_slot_attempted_by_item


def test_multiple_target_contacts_are_ambiguous(physical_env):
    env, state = physical_env
    for item in ITEM_IDS:
        state.plate_contacts[item] = {env.target_plate_name(slot) for slot in TARGET_SLOTS}
    snapshot = env.evaluation_snapshot()
    assert all(snapshot.item_placed_by_item.values())
    assert not any(snapshot.item_matched_by_item.values())
    assert not snapshot.target_slot_attempted_by_item


@pytest.mark.parametrize("destination", ("source_plate", "counter"))
def test_other_support_counts_as_putdown_without_a_target_attempt(physical_env, destination):
    env, state = physical_env
    for item in ITEM_IDS:
        if destination == "counter":
            state.counter_contacts.add(item)
        else:
            state.plate_contacts[item] = {env.source_plate_name(env.source_slot_by_item[item])}
    snapshot = env.evaluation_snapshot()
    assert all(snapshot.item_placed_by_item.values())
    assert not any(snapshot.item_matched_by_item.values())
    assert not snapshot.target_slot_attempted_by_item


def test_contact_query_errors_fail_closed(physical_env, monkeypatch):
    env, state = physical_env

    def fail(*args):
        raise RuntimeError("contact data unavailable")

    monkeypatch.setattr(env, "check_contact", fail)
    snapshot = env.evaluation_snapshot()
    assert not any(snapshot.item_placed_by_item.values())
    for item in ITEM_IDS:
        assert "contact data unavailable" in snapshot.details["physical"]["support_by_item"][item]["reason"]


def test_policy_history_does_not_create_recording_events(physical_env):
    env, state = physical_env
    evaluator = env.reset_evaluation()
    for index, item in enumerate(ITEM_IDS[::-1]):
        env.timestep = 3 * index + 1
        _pick(env, state, item)
        assert not env._check_success()
        _place(env, state, item, env.target_slot_for_item(item))
        env.timestep += 1
        assert not env._check_success()
        env.timestep += 1
        assert env._check_success() is (index == 1)
    assert evaluator.result().stop_execution
    env._update_task_state()
    assert not any(env._semantic_state.item_picked_by_item.values())
    assert not any(env._semantic_state.source_color_memory_latched_by_item.values())
    assert not any(env._semantic_state.item_placed_by_item.values())
    assert not env._semantic_state.protocol_violation


def test_policy_snapshot_is_built_once_per_state(physical_env, monkeypatch):
    env, _ = physical_env
    original = env._policy_evaluation_snapshot
    calls = []

    def counted(**kwargs):
        calls.append(env.timestep)
        return original(**kwargs)

    monkeypatch.setattr(env, "_policy_evaluation_snapshot", counted)
    env.reset_evaluation()
    snapshot = env.evaluation_snapshot()
    for _ in range(5):
        assert env.evaluation_snapshot() is snapshot
        env._check_success()
        env.task_diagnostics()
    assert calls == [0]
    env.timestep += 1
    assert env.evaluation_snapshot() is not snapshot
    assert calls == [0, 1]
    env.invalidate_policy_evidence()
    env.evaluation_snapshot()
    assert calls == [0, 1, 1]
    env.reset_evaluation()
    env.evaluation_snapshot()
    assert calls == [0, 1, 1, 1]


def test_direct_simulation_edits_invalidate_snapshot_key(physical_env):
    env, _ = physical_env
    env.sim.data.qpos = np.zeros(2)
    env.sim.data.time = 0.0
    initial = env.evaluation_snapshot()
    env.sim.data.qpos[0] = 1.0
    moved = env.evaluation_snapshot()
    assert moved is not initial
    env.sim.data.time = 0.01
    assert env.evaluation_snapshot() is not moved


@pytest.mark.parametrize("positions", ([0.02, 0.02], [0.035, 0.035], [0.6, 0.6], [-0.1, -0.1], [np.nan, 0.02]))
@pytest.mark.parametrize("contact", (False, True, "error"))
def test_contact_reuse_preserves_hold_diagnostics(positions, contact):
    from bench.execution.manipulation.object_interactions import get_object_hold_status

    calls = []

    def check(*args):
        calls.append(args)
        if contact == "error":
            raise RuntimeError("contact unavailable")
        return contact

    env = SimpleNamespace(
        objects={"box": "box"}, obj_body_id={"box": 0}, check_contact=check,
        robots=[SimpleNamespace(gripper={"right": "gripper"}, eef_site_id={"right": 0})],
        sim=SimpleNamespace(
            model=SimpleNamespace(get_joint_qpos_addr=lambda name: int(name.endswith("2"))),
            data=SimpleNamespace(
                qpos=np.array(positions), body_xpos=np.array([[0.0, 0.0, 1.0]]),
                site_xpos=np.array([[0.0, 0.0, 1.05]]),
            ),
        ),
    )
    legacy = get_object_hold_status(env, "box")
    assert len(calls) == 3
    calls.clear()
    assert get_object_hold_status(env, "box", contact_checker=check) == legacy
    assert len(calls) == 1


@pytest.mark.parametrize("split", ("train", "evaluation"))
def test_expert_attachment_reenables_recording_observer_in_either_split(
    physical_env, monkeypatch, split,
):
    from bench.tasks.wr08 import runtime
    from bench.tasks.wr08.expert import operations

    env, _ = physical_env
    descriptor = SimpleNamespace(
        episode=SimpleNamespace(split=split), episode_plan=resolve_episode_plan(seed=0),
        config=SimpleNamespace(initial_state=SimpleNamespace(robot_base=None, objects={}), manipulation={}),
        navigation=None, resources={"planner_script": "unused"}, execute=False,
    )
    monkeypatch.setattr(runtime, "initialize_episode_state", lambda env, obs, **kw: (obs, {}))
    monkeypatch.setattr(runtime, "resolve_episode_targets", lambda *args: {})
    env.require_expert_protocol = True
    _, _, metadata, _ = runtime._initialize(descriptor, env, {})
    evaluator = metadata["task_evaluator"]
    assert not env.require_expert_protocol and env._physical_observer is None
    assembly = SimpleNamespace(controller=SimpleNamespace())
    monkeypatch.setattr(operations, "build_navigation_assembly", lambda *a, **kw: assembly)
    monkeypatch.setattr(operations, "build_manipulation_assembly", lambda *a, **kw: assembly)
    monkeypatch.setattr(operations, "build_stage_handlers", lambda *a: {})
    operations.build_operation_handlers(descriptor, env, {})
    assert env.require_expert_protocol and env._physical_observer is not None
    assert env._wr08_evaluator is evaluator
    assert evaluator.result().details["evaluation"]["mode"] == "expert_recording"
    assert not evaluator.result().stop_execution


@pytest.mark.parametrize("offset,expected", [([0, 0, 0], True), ([0.2, 0, 0], False), ([0, 0, 0.1], False)])
def test_expert_placement_keeps_its_geometric_checks(physical_env, monkeypatch, offset, expected):
    from bench.execution import planning

    env, state = physical_env
    env.require_expert_protocol = True
    monkeypatch.setattr(planning, "object_reference_pos", lambda env, item, **kw: env._obj_pos(item) - [0, 0, 0.05])
    for item in ITEM_IDS:
        slot = env.target_slot_for_item(item)
        _place(env, state, item, slot)
        env.sim.data.body_xpos[env.obj_body_id[item]] += offset
        assert env.item_at_target(item, slot) is expected
