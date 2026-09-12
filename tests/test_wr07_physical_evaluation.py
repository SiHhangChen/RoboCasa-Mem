"""Read-only WR-07 physics checks and policy/expert runtime separation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.wr07.domain.contracts import ITEM_ID, SOURCE_SLOTS, TARGET_SLOTS
from bench.tasks.wr07.domain.episode import resolve_episode_plan
from bench.tasks.wr07.domain.semantics import (
    PhysicalState,
    SemanticState,
    evaluate_success,
    latch_source_color_memory,
    record_initial_source,
    record_item_picked,
    record_target_attempt,
    record_target_placement,
)


@pytest.fixture
def physical_env(monkeypatch):
    from bench.execution import planning
    from bench.execution.manipulation import object_interactions
    from bench.tasks.wr07.environment import WR07MatchMilkCartonToColoredSugarPlate, OU

    plan = resolve_episode_plan(seed=0)
    env = object.__new__(WR07MatchMilkCartonToColoredSugarPlate)
    env.require_expert_protocol = False
    env._wr07_evaluator = None
    env._semantic_state = SemanticState()
    record_initial_source(env._semantic_state, plan.source_slot_by_item[ITEM_ID],
                          plan.source_color_by_item[ITEM_ID])
    env.source_slot_by_item = dict(plan.source_slot_by_item)
    env.source_color_by_item = dict(plan.source_color_by_item)
    env.target_slot_by_color = dict(plan.target_slot_by_color)
    env.source_plate_color_by_slot = dict(plan.source_plate_color_by_slot)
    env.target_plate_color_by_slot = dict(plan.target_plate_color_by_slot)
    env.target_item_offset_xy_by_item = {ITEM_ID: np.zeros(2)}
    env.plate_threshold = 0.16
    env.timestep = 0

    names = [ITEM_ID]
    names += [env.source_plate_name(slot) for slot in SOURCE_SLOTS]
    names += [env.target_plate_name(slot) for slot in TARGET_SLOTS]
    positions = np.array([
        [0.0, 0.0, 0.98],
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
    env._source_surface_z = 0.93
    positions[0, :2] = env._source_plate_centers[plan.source_slot_by_item[ITEM_ID]][:2]
    env._evaluation_initial_item_position = positions[0].copy()
    state = SimpleNamespace(
        qpos=np.array([0.04, 0.04]), held=False, contact_known=True,
        finite=True, plate_contacts=set(), counter_contact=False,
    )
    env.robots = [SimpleNamespace(get_gripper_joint_positions=lambda arm: state.qpos)]
    env.check_contact = lambda item, plate: item == ITEM_ID and plate in state.plate_contacts
    monkeypatch.setattr(env, "_item_released", lambda: not state.held)
    monkeypatch.setattr(object_interactions, "get_object_hold_status", lambda *a, **kw: {
        "held": state.held and state.finite,
        "contact_and_grasp": {"contact": state.held} if state.contact_known else {},
        "checks": {"finite_pose": state.finite, "grasp_or_contact": state.held},
    })
    def unexpected_geometry_filter(*args, **kwargs):
        raise AssertionError("Policy placement must use contact, not XY/bottom geometry")

    monkeypatch.setattr(planning, "object_reference_pos", unexpected_geometry_filter)
    monkeypatch.setattr(OU, "check_obj_in_receptacle", unexpected_geometry_filter)
    monkeypatch.setattr(OU, "check_obj_any_counter_contact", lambda *a: state.counter_contact)
    return env, state


def _place(env, state, slot):
    name = env.target_plate_name(slot)
    env.sim.data.body_xpos[0] = env._obj_pos(name) + np.array([0.0, 0.0, 0.05])
    state.plate_contacts = {name}
    state.held = False
    state.qpos = np.array([0.04, 0.04])


@pytest.mark.parametrize(
    ("held", "gripper", "lift", "lateral", "expected"),
    ((True, 0.02, 0.03, 0.0, True),
     (False, 0.02, 0.03, 0.0, False),
     (True, 0.04, 0.03, 0.0, False),
     (True, 0.02, 0.01, 0.0, False),
     (True, 0.02, 0.0, 0.10, False),
     (False, 0.04, -0.8, 1.0, False)),
)
def test_pickup_requires_closed_gripper_grasp_and_lift(
    physical_env, held, gripper, lift, lateral, expected,
):
    env, state = physical_env
    state.held = held
    state.qpos[:] = gripper
    env.sim.data.body_xpos[0] += np.array([lateral, 0.0, lift])
    assert env._policy_grasp_evidence()["picked_up"] is expected


@pytest.mark.parametrize("fault", ("unknown_contact", "nonfinite_pose", "invalid_gripper"))
def test_unavailable_grasp_evidence_does_not_imply_release(physical_env, fault):
    env, state = physical_env
    if fault == "unknown_contact":
        state.contact_known = False
    elif fault == "nonfinite_pose":
        state.finite = False
    else:
        state.qpos[:] = np.nan
    evidence = env._policy_grasp_evidence()
    assert not evidence["picked_up"] and not evidence["released"]


@pytest.mark.parametrize("offset", ((0.20, 0.0, 0.0), (0.0, 0.0, 0.10), (0.20, 0.0, 0.10)))
def test_matching_uses_contact_without_center_distance_or_bottom_height(physical_env, offset):
    env, state = physical_env
    target = env.target_slot_for_source_color()
    _place(env, state, target)
    # The contact oracle is authoritative even when body reference positions
    # fall outside the former 16 cm XY / 6 cm bottom-height filters.
    env.sim.data.body_xpos[0] += np.asarray(offset)
    env.target_item_offset_xy_by_item[ITEM_ID] = np.array([-0.04, 0.0])
    if offset[0]:
        assert env.detect_target_slot() is None
    snapshot = env.evaluation_snapshot()
    assert snapshot.item_placed and snapshot.target_color_matches
    assert snapshot.target_slot_attempted == target
    assert not snapshot.source_color_memory_latched
    assert not snapshot.domain_success
    assert not env._semantic_state.item_placed
    assert not env._semantic_state.protocol_violation


@pytest.mark.parametrize("case", ("midair", "floor", "held", "no_contact"))
def test_target_xy_alone_does_not_prove_a_placement(physical_env, case):
    env, state = physical_env
    _place(env, state, env.target_slot_for_source_color())
    if case == "midair":
        env.sim.data.body_xpos[0, 2] += 0.3
        state.plate_contacts.clear()
    elif case == "floor":
        env.sim.data.body_xpos[0, 2] = 0.05
        state.plate_contacts.clear()
    elif case == "held":
        state.held = True
        state.qpos[:] = 0.02
    else:
        state.plate_contacts.clear()
    snapshot = env.evaluation_snapshot()
    assert not snapshot.item_placed
    assert not snapshot.target_color_matches


def test_contact_with_multiple_target_plates_is_not_a_unique_match(physical_env):
    env, state = physical_env
    _place(env, state, env.target_slot_for_source_color())
    state.plate_contacts = {env.target_plate_name(slot) for slot in TARGET_SLOTS}
    snapshot = env.evaluation_snapshot()
    assert snapshot.item_placed
    assert snapshot.target_slot_attempted is None
    assert not snapshot.target_color_matches


def test_failed_contact_query_does_not_imply_placement(physical_env, monkeypatch):
    env, state = physical_env
    _place(env, state, env.target_slot_for_source_color())

    def failed_contact(*args):
        raise RuntimeError("contact data unavailable")

    monkeypatch.setattr(env, "check_contact", failed_contact)
    snapshot = env.evaluation_snapshot()
    assert not snapshot.item_placed
    assert not snapshot.target_color_matches
    assert "contact data unavailable" in snapshot.details["physical"]["support"]["reason"]


@pytest.mark.parametrize(
    ("offset", "expected"),
    (((0.0, 0.0, 0.0), True), ((0.20, 0.0, 0.0), False), ((0.0, 0.0, 0.10), False)),
)
def test_expert_placement_geometry_is_unchanged(physical_env, monkeypatch, offset, expected):
    from bench.execution import planning

    env, state = physical_env
    env.require_expert_protocol = True
    target = env.target_slot_for_source_color()
    _place(env, state, target)
    env.sim.data.body_xpos[0] += np.asarray(offset)
    monkeypatch.setattr(planning, "object_reference_pos", lambda *a, **kw:
                        env._obj_pos(ITEM_ID) - np.array([0.0, 0.0, 0.05]))
    assert env.item_at_target(target) is expected


@pytest.mark.parametrize("destination", ("wrong_plate", "counter", "source_plate"))
def test_putdown_is_independent_of_target_color(physical_env, destination):
    env, state = physical_env
    if destination == "wrong_plate":
        wrong = next(slot for slot in TARGET_SLOTS if slot != env.target_slot_for_source_color())
        _place(env, state, wrong)
    elif destination == "counter":
        state.counter_contact = True
    else:
        state.plate_contacts = {env.source_plate_name(env.source_slot_by_item[ITEM_ID])}
    snapshot = env.evaluation_snapshot()
    assert snapshot.item_placed
    assert not snapshot.target_color_matches


@pytest.mark.parametrize("placement_offset", ((0.0, 0.0, 0.0), (0.20, 0.0, 0.10)))
def test_environment_success_and_diagnostics_follow_physical_evaluator(
    physical_env, placement_offset,
):
    env, state = physical_env
    evaluator = env.reset_evaluation()
    env.timestep = 1
    state.qpos[:] = 0.02
    state.held = True
    env.sim.data.body_xpos[0, 2] += 0.03
    assert not env._check_success()
    assert evaluator.result().stages["item_picked"].qualified
    _place(env, state, env.target_slot_for_source_color())
    env.sim.data.body_xpos[0] += np.asarray(placement_offset)
    for step in (2, 3):
        env.timestep = step
        assert env._check_success() is (step == 3)
        assert env.task_diagnostics()["success"] is (step == 3)
    # Policy observations must not synthesize any expert events.
    env._update_task_state()
    assert not env._semantic_state.item_picked
    assert not env._semantic_state.source_color_memory_latched
    assert not env._semantic_state.item_placed
    assert not env._semantic_state.protocol_violation


@pytest.mark.parametrize("matched", (False, True))
def test_physical_policy_terminal_result_cannot_be_revised(physical_env, matched):
    env, state = physical_env
    env.reset_evaluation()
    env.timestep = 1
    state.qpos[:] = 0.02
    state.held = True
    env.sim.data.body_xpos[0, 2] += 0.03
    assert not env._check_success()

    correct = env.target_slot_for_source_color()
    wrong = next(slot for slot in TARGET_SLOTS if slot != correct)
    _place(env, state, correct if matched else wrong)
    env.timestep = 2
    assert not env.task_diagnostics()["terminal"]
    env.timestep = 3
    assert env._check_success() is matched
    terminal = env.task_diagnostics()
    assert terminal["terminal"]
    assert terminal["stop_execution"]
    assert terminal["failure_code"] == (None if matched else "wrong_target_placement")
    assert terminal["evaluation"]["terminal_step"] == 3

    # Defensive check for late readers; the rollout itself stops at step 3.
    state.held = True
    state.plate_contacts.clear()
    env.timestep = 4
    assert env._check_success() is matched
    _place(env, state, wrong if matched else correct)
    env.timestep = 5
    assert env.task_diagnostics() == terminal
    assert not env._semantic_state.source_color_memory_latched
    assert not env._semantic_state.protocol_violation


@pytest.mark.parametrize("split", ("train", "evaluation"))
def test_runtime_selects_policy_but_explicit_expert_attachment_overrides_it(
    physical_env, monkeypatch, split,
):
    from bench.tasks.wr07 import runtime
    from bench.tasks.wr07.expert import operations

    env, _ = physical_env
    descriptor = SimpleNamespace(
        episode=SimpleNamespace(split=split), episode_plan=SimpleNamespace(seed=0),
        config=SimpleNamespace(initial_state=SimpleNamespace(robot_base=None, objects={}),
                               manipulation={}),
        navigation=None, resources={"planner_script": "unused"}, execute=False,
    )
    monkeypatch.setattr(runtime, "initialize_episode_state", lambda env, obs, **kw: (obs, {}))
    monkeypatch.setattr(runtime, "resolve_episode_targets", lambda *a: {})
    env.require_expert_protocol = True
    _, _, metadata, _ = runtime._initialize(descriptor, env, {})
    evaluator = metadata["task_evaluator"]
    assert not env.require_expert_protocol
    assert evaluator.result().details["evaluation"]["mode"] == "policy_physical"

    assembly = SimpleNamespace(controller=SimpleNamespace())
    monkeypatch.setattr(operations, "build_navigation_assembly", lambda *a, **kw: assembly)
    monkeypatch.setattr(operations, "build_manipulation_assembly", lambda *a, **kw: assembly)
    monkeypatch.setattr(operations, "build_stage_handlers", lambda *a: {})
    operations.build_operation_handlers(descriptor, env, {})
    assert env.require_expert_protocol
    assert env._wr07_evaluator is evaluator
    assert evaluator.result().details["evaluation"]["mode"] == "expert_recording"


@pytest.mark.parametrize("remember", (False, True))
def test_expert_semantics_still_need_explicit_memory_latch(remember):
    state = SemanticState()
    record_initial_source(state, "source_left", "red")
    record_item_picked(state, physically_picked=True)
    if remember:
        assert latch_source_color_memory(state)
    record_target_attempt(state, "target_right", matches=True)
    record_target_placement(state, matches=True)
    physical = PhysicalState({}, {"target_right": True}, True, True)
    assert evaluate_success(state, physical, target_slot="target_right").success is remember
    assert state.source_color_memory_latched is remember
    assert state.protocol_violation is not remember
