"""Exercise WR-08 reset methods with real MuJoCo free joints."""

from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.wr08.domain.contracts import ITEM_IDS

pytestmark = pytest.mark.simulator


@pytest.fixture
def env():
    pytest.importorskip("mujoco")
    from bench.env.robocasa_runtime import bootstrap_local_robocasa_sources

    bootstrap_local_robocasa_sources()
    from robosuite.utils.binding_utils import MjSim
    from bench.tasks.wr08.environment import WR08MatchMilkCartonsToColoredSugarPlates

    sim = MjSim.from_xml_string("""
    <mujoco><worldbody>
      <body name="source_plate_left" pos="1 2 0.93" quat="0.70710678 0 0 0.70710678">
        <freejoint name="left_joint"/><geom size="0.15 0.15 0.015" type="box"/>
      </body>
      <body name="source_plate_right" pos="2 3 0.97">
        <freejoint name="right_joint"/><geom size="0.15 0.15 0.025" type="box"/>
      </body>
      <body name="boxed_food_a" pos="0 0 0.01">
        <freejoint name="a_joint"/><geom size="0.03 0.03 0.06" type="box"/>
      </body>
      <body name="boxed_food_b" pos="0 0 9">
        <freejoint name="b_joint"/><geom size="0.03 0.03 0.09" type="box"/>
      </body>
    </worldbody></mujoco>
    """)
    instance = WR08MatchMilkCartonsToColoredSugarPlates.__new__(
        WR08MatchMilkCartonsToColoredSugarPlates
    )
    instance.sim = sim
    instance.objects = {
        name: SimpleNamespace(
            joints=[joint], top_offset=np.array([0, 0, half_z]),
            bottom_offset=np.array([0, 0, -half_z]),
        )
        for name, joint, half_z in (
            ("source_plate_left", "left_joint", 0.015),
            ("source_plate_right", "right_joint", 0.025),
            ("boxed_food_a", "a_joint", 0.06),
            ("boxed_food_b", "b_joint", 0.09),
        )
    }
    instance.obj_body_id = {name: sim.model.body_name2id(name) for name in instance.objects}
    sim.forward()
    instance.source_slot_by_item = dict(zip(ITEM_IDS, ("source_left", "source_right")))
    instance._source_plate_centers = {
        slot: instance._obj_pos(instance.source_plate_name(slot)).copy()
        for slot in instance.source_slot_by_item.values()
    }
    instance.source_item_offset_xy_by_item = {
        ITEM_IDS[0]: np.array([0.03, -0.02]), ITEM_IDS[1]: np.array([0.04, 0.01]),
    }
    instance.source_item_yaw_by_item = dict(zip(ITEM_IDS, (1.2, -0.7)))
    instance.carton_material_by_item = {item: "satin" for item in ITEM_IDS}
    sim.data.qvel[:] = 3.0
    return instance


def test_cartons_use_their_own_tray_height_and_start_without_momentum(env):
    env._apply_milk_carton_poses()
    np.testing.assert_allclose(env._obj_pos(ITEM_IDS[0]), [1.02, 2.03, 1.005])
    np.testing.assert_allclose(env._obj_pos(ITEM_IDS[1]), [2.04, 3.01, 1.085])
    for item in ITEM_IDS:
        joint = env.objects[item].joints[0]
        yaw = env.source_item_yaw_by_item[item]
        np.testing.assert_allclose(
            env.sim.data.get_joint_qpos(joint)[3:],
            [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
        )
        np.testing.assert_array_equal(env.sim.data.get_joint_qvel(joint), np.zeros(6))
    np.testing.assert_array_equal(env.sim.data.get_joint_qvel("left_joint"), np.full(6, 3.0))


def test_episode_configuration_does_not_reposition_settled_cartons(env):
    env._apply_milk_carton_poses()
    env.source_plate_color_by_slot = {"source_left": "red", "source_right": "blue"}
    env.target_plate_color_by_slot = {"target_left": "blue", "target_right": "red"}
    env.source_color_by_item = dict(zip(ITEM_IDS, ("red", "blue")))
    env.target_slot_by_color = {"red": "target_right", "blue": "target_left"}
    env.target_item_offset_xy_by_item = {item: np.zeros(2) for item in ITEM_IDS}
    env.target_item_yaw_by_item = {item: 0.0 for item in ITEM_IDS}
    plan = SimpleNamespace(**{
        name: getattr(env, name)
        for name in (
            "source_plate_color_by_slot", "target_plate_color_by_slot",
            "source_slot_by_item", "source_color_by_item", "target_slot_by_color",
            "source_item_offset_xy_by_item", "target_item_offset_xy_by_item",
            "source_item_yaw_by_item", "target_item_yaw_by_item",
            "carton_material_by_item",
        )
    }, transport_order=ITEM_IDS[::-1])
    env.sim.data.qpos[:] += 0.01
    env.sim.data.qvel[:] = 2.0
    before = env.sim.data.qpos.copy()
    env.configure_episode(plan)
    np.testing.assert_array_equal(env.sim.data.qpos, before)
    np.testing.assert_array_equal(env.sim.data.qvel, np.full(env.sim.model.nv, 2.0))
    assert env._transport_order == ITEM_IDS[::-1]


def test_pick_preflight_uses_virtual_arm_start_without_changing_physics(monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    from bench.env.robocasa_runtime import bootstrap_local_robocasa_sources

    bootstrap_local_robocasa_sources()
    from robosuite.utils.binding_utils import MjSim
    from bench.execution.manipulation.planning_bridge import PlannedStageExecutor
    from bench.tasks.wr08.execution.pick_controller import WorldAnchoredPickExecutor

    sim = MjSim.from_xml_string("""
    <mujoco><worldbody>
      <body name="arm"><joint name="arm_joint" type="hinge"/>
        <geom size="0.1" mass="1"/>
      </body>
      <body name="carton" pos="1 0 1"><freejoint name="carton_joint"/>
        <geom size="0.04" mass="0.1"/>
      </body>
    </worldbody></mujoco>
    """)
    env = SimpleNamespace(sim=sim, robots=[SimpleNamespace(
        composite_controller=SimpleNamespace(part_controllers={
            "right": SimpleNamespace(qpos_index=[0]),
        }),
    )])
    obs = {"robot0_base_pos": [0, 0, 0], "robot0_base_quat": [0, 0, 0, 1]}
    executor = WorldAnchoredPickExecutor(
        env, obs, driver=object(), planner=lambda payload: {},
        runtime_config=SimpleNamespace(execute=True),
    )
    sim.data.qvel[:] = 0.25
    sim.forward()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    before = np.empty(mujoco.mj_stateSize(sim.model._model, state_spec))
    mujoco.mj_getState(sim.model._model, sim.data._data, before, state_spec)
    carton_before = sim.data.get_joint_qpos("carton_joint").copy()

    def check(self, trajectory, monitor, summary):
        assert sim.data.qpos[0] == 0.5
        np.testing.assert_array_equal(sim.data.get_joint_qpos("carton_joint"), carton_before)
        raise RuntimeError("preflight probe")

    monkeypatch.setattr(PlannedStageExecutor, "_preflight_if_enabled", check)
    with executor.pick_frame(obs):
        executor._virtual_arm_qpos = np.array([0.5])
        with pytest.raises(RuntimeError, match="preflight probe"):
            executor._preflight_if_enabled([[0.7]], {}, {})
    after = np.empty_like(before)
    mujoco.mj_getState(sim.model._model, sim.data._data, after, state_spec)
    np.testing.assert_array_equal(after, before)
