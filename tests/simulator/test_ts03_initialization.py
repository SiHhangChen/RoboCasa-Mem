from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.ts03.domain.episode import resolve_episode_plan
from bench.tasks.ts03.domain.geometry import sample_sugar_cube_yaws_for_seed
from bench.tasks.ts03.initialization import apply_initial_object_state


pytestmark = pytest.mark.simulator


@pytest.fixture
def env():
    pytest.importorskip("mujoco")
    from bench.env.robocasa_runtime import bootstrap_local_robocasa_sources

    bootstrap_local_robocasa_sources()
    from robosuite.utils.binding_utils import MjSim

    sim = MjSim.from_xml_string("""
    <mujoco><worldbody>
      <body name="sugar_plate" pos="1 2 0.93" quat="0.70710678 0 0 0.70710678">
        <freejoint name="plate_joint"/><geom type="box" size="0.15 0.1 0.01"/>
      </body>
      <body name="sugar0" pos="0 0 0.95" euler="0.1 0.2 0.4">
        <freejoint name="sugar0_joint"/><geom type="box" size="0.01 0.01 0.01"/>
      </body>
      <body name="sugar1" pos="0 0 0.95" euler="-0.2 0.1 -0.7">
        <freejoint name="sugar1_joint"/><geom type="box" size="0.01 0.01 0.01"/>
      </body>
      <body name="sugar2" pos="0 0 0.95">
        <freejoint name="sugar2_joint"/><geom type="box" size="0.01 0.01 0.01"/>
      </body>
    </worldbody></mujoco>
    """)
    objects = {
        "sugar_plate": SimpleNamespace(joints=["plate_joint"], size=(0.3, 0.2, 0.02)),
        **{
            f"sugar{i}": SimpleNamespace(joints=[f"sugar{i}_joint"], size=(0.02, 0.02, 0.02))
            for i in range(3)
        },
    }
    instance = SimpleNamespace(
        sim=sim, objects=objects, sugar_plate_name="sugar_plate",
        obj_body_id={name: sim.model.body_name2id(name) for name in objects},
        object_placements={name: (None, None, obj) for name, obj in objects.items()},
        original_sugar_poses={},
    )
    instance._get_observations = lambda **kwargs: {
        name: sim.data.get_joint_qpos(obj.joints[0]).copy()
        for name, obj in objects.items()
    }
    sim.forward()
    return instance


def test_reset_applies_sampled_world_yaw_and_rotated_tray_offsets(env):
    episode = resolve_episode_plan(seed=7)
    plate_before = env.sim.data.get_joint_qpos("plate_joint").copy()
    before_rotations = {
        name: env.sim.data.body_xmat[env.obj_body_id[name]].reshape(3, 3).copy()
        for name in episode.sugar_names
    }
    plate_rotation = env.sim.data.body_xmat[env.obj_body_id["sugar_plate"]].reshape(3, 3).copy()
    observation, summary = apply_initial_object_state(env, episode=episode)
    yaws = sample_sugar_cube_yaws_for_seed(count=3, seed=7)
    assert summary["sugar_cube_yaws_rad"] == list(yaws)
    assert summary["sugar_plate_sampling_radius_m"] == 0.005
    np.testing.assert_array_equal(env.sim.data.get_joint_qpos("plate_joint"), plate_before)

    for name, yaw, offset in zip(episode.sugar_names, yaws, env.sugar_plate_offsets_xy, strict=True):
        qpos = observation[name]
        expected_position = plate_before[:3] + plate_rotation @ np.asarray([*offset, 0.0])
        np.testing.assert_allclose(qpos[:3], expected_position, atol=1e-7)
        rotation = env.sim.data.body_xmat[env.obj_body_id[name]].reshape(3, 3)
        assert np.arctan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(yaw, abs=1e-7)
        np.testing.assert_allclose(rotation[2, :], before_rotations[name][2, :], atol=1e-7)
        assert np.linalg.norm(qpos[3:]) == pytest.approx(1.0, abs=1e-7)
        np.testing.assert_allclose(summary["objects"][name]["after_quat_wxyz"], qpos[3:])
        np.testing.assert_allclose(env.object_placements[name][1], qpos[3:])
        np.testing.assert_allclose(env.original_sugar_poses[name], qpos[:3])


def test_same_seed_reuses_positions_and_absolute_yaws(env):
    episode = resolve_episode_plan(seed=7)
    first, _ = apply_initial_object_state(env, episode=episode)
    second, _ = apply_initial_object_state(env, episode=episode)
    for name in episode.sugar_names:
        np.testing.assert_allclose(first[name], second[name], atol=1e-7)


def test_different_seeds_change_positions_and_yaws(env):
    first, _ = apply_initial_object_state(env, episode=resolve_episode_plan(seed=7))
    second, _ = apply_initial_object_state(env, episode=resolve_episode_plan(seed=11))
    assert not np.allclose(first["sugar0"][:3], second["sugar0"][:3])
    assert not np.allclose(first["sugar0"][3:], second["sugar0"][3:])


def test_yaw_configuration_does_not_change_xy_samples(env):
    episode = resolve_episode_plan(seed=7)
    first, _ = apply_initial_object_state(env, episode=episode)
    second, _ = apply_initial_object_state(
        env, episode=episode, settings={"sugar_cube_yaw_half_range_deg": 0.0},
    )
    for name in episode.sugar_names:
        np.testing.assert_array_equal(first[name][:3], second[name][:3])
        rotation = env.sim.data.body_xmat[env.obj_body_id[name]].reshape(3, 3)
        assert np.arctan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(0.0, abs=1e-7)
