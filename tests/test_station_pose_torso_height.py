from types import SimpleNamespace

import numpy as np
import pytest

from bench.execution.navigation.station_poses import (
    StationPoseSampler,
    StationPoseSelector,
    station_sample_torso_height,
)
from scripts.planning.calibrate_station_base_poses import (
    clipped_torso_target,
    current_pose_sample,
    set_current_yaw_translation_command,
    teleop_translation_world_vector,
)


def test_calibration_sample_keeps_base_z_and_torso_height_separate():
    env = SimpleNamespace(
        sim=SimpleNamespace(
            data=SimpleNamespace(
                qpos=np.array([0.0, 0.24], dtype=float),
                time=1.5,
            )
        )
    )
    torso = SimpleNamespace(qpos_index=np.array([1], dtype=int))
    obs = {
        "robot0_base_pos": np.array([1.0, -2.0, 0.7], dtype=float),
        "robot0_base_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    }

    sample = current_pose_sample(
        env,
        obs,
        "shelf_upper_station",
        3,
        torso=torso,
    )

    assert sample["base_pos"] == [1.0, -2.0, 0.7]
    assert sample["torso_height"] == pytest.approx(0.24)


def test_torso_target_is_clipped_inside_joint_limit_margin():
    env = SimpleNamespace(
        sim=SimpleNamespace(
            model=SimpleNamespace(
                jnt_range=np.array(
                    [
                        [-1.0, 1.0],
                        [0.0, 0.35],
                    ],
                    dtype=float,
                )
            )
        )
    )
    torso = SimpleNamespace(joint_index=np.array([1], dtype=int))

    target, clipped = clipped_torso_target(
        env,
        torso,
        0.35,
        margin=0.005,
    )

    assert clipped is True
    assert target == pytest.approx(0.345)


@pytest.mark.parametrize(
    ("yaw", "forward", "left", "expected"),
    [
        (0.0, 1.0, 0.0, [1.0, 0.0]),
        (np.pi / 2.0, 1.0, 0.0, [0.0, 1.0]),
        (np.pi, 1.0, 0.0, [-1.0, 0.0]),
        (np.pi, 0.0, 1.0, [0.0, -1.0]),
    ],
)
def test_calibration_teleop_translation_uses_current_base_yaw(
    yaw,
    forward,
    left,
    expected,
):
    np.testing.assert_allclose(
        teleop_translation_world_vector(yaw, forward=forward, left=left),
        expected,
        atol=1e-12,
    )


def test_calibration_forward_at_pi_yaw_sends_negative_raw_x_after_conversion():
    command = np.zeros(3, dtype=float)

    set_current_yaw_translation_command(
        command,
        env=SimpleNamespace(),
        obs={"yaw": np.pi},
        forward=1.0,
        speed=0.45,
        base_yaw_fn=lambda obs: float(obs["yaw"]),
        world_xy_to_controller_action_fn=lambda env, vec: np.asarray(vec, dtype=float),
    )

    np.testing.assert_allclose(command, [-0.45, 0.0, 0.0], atol=1e-12)


def test_station_sampler_preserves_torso_height_on_pose_and_summary():
    payload = {
        "stations": {
            "shelf_upper_station": {
                "samples": [
                    {
                        "sample_index": 0,
                        "base_pos": [1.0, -2.0, 0.7],
                        "base_yaw": 0.5,
                        "torso_height": 0.28,
                    }
                ]
            }
        }
    }
    sampler = StationPoseSampler(
        payload=payload,
        selector=StationPoseSelector(policy="first"),
    )

    pose = sampler.sample("shelf_upper_station")

    assert pose.position == pytest.approx((1.0, -2.0, 0.7))
    assert pose.torso_height == pytest.approx(0.28)
    assert sampler.last_summary("shelf_upper_station")["torso_height"] == pytest.approx(
        0.28
    )


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"torso_height": 0.2}, 0.2),
        ({"torso_qpos": [0.3]}, 0.3),
        ({"base_pos": [1.0, 2.0, 0.7]}, None),
        ({"torso_height": float("nan")}, None),
    ],
)
def test_station_sample_torso_height_is_backward_compatible(sample, expected):
    assert station_sample_torso_height(sample) == expected
