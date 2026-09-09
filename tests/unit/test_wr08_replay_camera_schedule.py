"""Camera elision must not change physics or any observation the policy reads."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analysis import replay_wr08_actions_on_demand as replay


class Observable:
    def __init__(self, sensor):
        self._sensor = sensor

    def set_sensor(self, sensor):
        self._sensor = sensor


class CameraEnv:
    def __init__(self):
        self.camera_names = ("left", "right", "wrist")
        self._observables = {}
        self.sim = SimpleNamespace(data=SimpleNamespace(
            time=0.0, qpos=np.zeros(2), qvel=np.zeros(2), act=np.zeros(0), ctrl=np.zeros(1),
        ))
        self.rendered_at = []
        for camera in self.camera_names:
            self._observables[f"{camera}_image"] = Observable(self.make_sensor(camera))

    def make_sensor(self, camera):
        def sensor(cache):
            self.rendered_at.append((camera, self.sim.data.time))
            cache[f"{camera}_depth"] = np.full((2, 2), self.sim.data.time)
            return np.full((2, 2, 3), round(self.sim.data.time * 20), dtype=np.uint8)

        sensor.__modality__ = "image"
        return sensor

    def observe(self):
        cache = {}
        for camera in self.camera_names:
            cache[f"{camera}_image"] = self._observables[f"{camera}_image"]._sensor(cache)
        return cache

    def step(self, action):
        data = self.sim.data
        data.time += 0.05
        data.ctrl[:] = action
        data.qvel += action[0] * 0.01
        data.qpos += data.qvel * 0.05
        return self.observe()


def policy_observation(env, obs):
    return SimpleNamespace(state=env.sim.data.qpos.copy(),
                           images={camera: obs[f"{camera}_image"] for camera in env.camera_names})


def replay_fake(every_step):
    env = CameraEnv()
    obs = env.observe()
    originals = {k: v._sensor for k, v in env._observables.items()}
    original_step = env.step
    schedule = replay.CameraSchedule(env, obs, stride=3, horizon=8, every_step=every_step)
    schedule.record_input(policy_observation(env, obs))
    consumed = []
    for step in range(1, 9):
        obs = env.step(np.array([step * 0.01]))
        if step % 3 == 0:
            schedule.record_input(policy_observation(env, obs))
            consumed.append(obs)
    final = obs
    schedule.close()
    schedule.close()
    assert env.step == original_step
    assert all(env._observables[k]._sensor is original for k, original in originals.items())
    return schedule, env, consumed, final


def test_sparse_and_full_have_identical_physics_inputs_and_final_images():
    sparse, sparse_env, sparse_inputs, sparse_final = replay_fake(False)
    full, full_env, full_inputs, full_final = replay_fake(True)
    assert sparse.metadata["physics_steps"] == full.metadata["physics_steps"] == 8
    assert sparse.metadata["observation_calls"] == full.metadata["observation_calls"] == 3
    assert sparse.metadata["physics_sha256"] == full.metadata["physics_sha256"]
    assert sparse.metadata["consumed_observation_sha256"] == full.metadata["consumed_observation_sha256"]
    assert sparse.metadata["render_calls"] == 9
    assert sparse.metadata["elided_calls"] == 15
    assert full.metadata["render_calls"] == 24 and full.metadata["elided_calls"] == 0
    for a, b in zip([*sparse_inputs, sparse_final], [*full_inputs, full_final]):
        assert set(a) == set(b)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(sparse_env.sim.data.qpos, full_env.sim.data.qpos)


def test_elided_frame_preserves_depth_cache_and_sensor_modality():
    env = CameraEnv()
    initial = env.observe()
    schedule = replay.CameraSchedule(env, initial, stride=3, horizon=8, every_step=False)
    obs = env.step(np.array([0.1]))
    assert schedule.elided_calls == 3
    for camera in env.camera_names:
        np.testing.assert_array_equal(obs[f"{camera}_depth"], initial[f"{camera}_depth"])
        assert env._observables[f"{camera}_image"]._sensor.__modality__ == "image"
    with pytest.raises(RuntimeError, match="unscheduled camera observation"):
        schedule.record_input(policy_observation(env, obs))
    schedule.close()


@pytest.mark.parametrize("missing", ["sensor", "initial_frame"])
def test_missing_camera_does_not_partially_install_schedule(missing):
    env = CameraEnv()
    obs = env.observe()
    if missing == "sensor":
        del env._observables["wrist_image"]
    else:
        del obs["wrist_image"]
    originals = {k: v._sensor for k, v in env._observables.items()}
    original_step = env.step
    with pytest.raises(ValueError, match="required camera observable missing"):
        replay.CameraSchedule(env, obs, stride=3, horizon=8, every_step=False)
    assert env.step == original_step
    assert all(env._observables[k]._sensor is original for k, original in originals.items())


@pytest.mark.parametrize("stride,horizon", [(0, 8), (-1, 8), (3, 0), (3, -1)])
def test_invalid_schedule_is_rejected(stride, horizon):
    env = CameraEnv()
    with pytest.raises(ValueError, match="must be positive"):
        replay.CameraSchedule(env, env.observe(), stride=stride, horizon=horizon, every_step=False)


@pytest.mark.parametrize("extra,error", [(["--record-video"], "cannot record per-step video"),
                                       (["--task-id", "WR-07"], "scoped to WR-08")])
def test_cli_rejects_unvalidated_modes_before_simulation(monkeypatch, tmp_path, extra, error):
    monkeypatch.setattr("sys.argv", ["replay", "--task-id", "WR-08", "--dataset-root", str(tmp_path),
                                    "--output-run", str(tmp_path / "run"), *extra])
    with pytest.raises(ValueError, match=error):
        replay.main()
    assert not (tmp_path / "run").exists()
