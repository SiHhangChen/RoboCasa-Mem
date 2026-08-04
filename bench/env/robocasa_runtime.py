"""RoboCasa runtime helpers owned by bench.

Scripts that need real RoboCasa environments opt in to this module explicitly.
Importing ``bench`` or ``bench.env`` does not load RoboCasa.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bench.runtime.contracts import CONTROL_FREQUENCY_HZ
from bench.runtime.data_contract import PANDA_ARM_JOINT_LIMITS_RAD


PANDA_QPOS_LOW = [limits[0] for limits in PANDA_ARM_JOINT_LIMITS_RAD]
PANDA_QPOS_HIGH = [limits[1] for limits in PANDA_ARM_JOINT_LIMITS_RAD]


def bootstrap_local_robocasa_sources(repo_root: Path | None = None) -> None:
    """Prefer checked-out RoboCasa / robosuite sources when run from repo root."""
    root = repo_root or Path(__file__).resolve().parents[2]
    for project in ("robosuite", "robocasa"):
        source_root = root / project
        package_init = source_root / project / "__init__.py"
        if package_init.exists() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))


def register_bench_tasks() -> None:
    """Import bench-owned RoboCasa task classes for robosuite registration."""
    bootstrap_local_robocasa_sources()
    from bench.tasks import register_builtin_robocasa_tasks

    register_builtin_robocasa_tasks()


def make_joint_position_controller_config() -> dict[str, Any]:
    bootstrap_local_robocasa_sources()
    from robosuite.controllers import load_composite_controller_config

    config = copy.deepcopy(load_composite_controller_config(robot="PandaOmron"))
    body_parts = config["body_parts"]
    right_key: str | tuple[str, str] | None = "right" if "right" in body_parts else None
    if right_key is None and "arms" in body_parts and "right" in body_parts["arms"]:
        right_key = ("arms", "right")
    if right_key is None:
        raise KeyError(
            f"cannot find right arm controller in body_parts keys {list(body_parts)}"
        )

    right = copy.deepcopy(
        body_parts[right_key]
        if isinstance(right_key, str)
        else body_parts[right_key[0]][right_key[1]]
    )
    right.update(
        {
            "type": "JOINT_POSITION",
            "input_type": "absolute",
            "input_max": PANDA_QPOS_HIGH,
            "input_min": PANDA_QPOS_LOW,
            "output_max": PANDA_QPOS_HIGH,
            "output_min": PANDA_QPOS_LOW,
            "kp": 50,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "qpos_limits": [PANDA_QPOS_LOW, PANDA_QPOS_HIGH],
            "interpolation": None,
            "ramp_ratio": 0.2,
        }
    )
    right.setdefault("gripper", {"type": "GRIP"})
    if isinstance(right_key, str):
        body_parts[right_key] = right
    else:
        body_parts[right_key[0]][right_key[1]] = right
    return config


def make_robocasa_env(
    *,
    env_name: str,
    split: str,
    seed: int,
    controller_config: dict[str, Any],
    env_kwargs: dict[str, Any] | None = None,
    camera_names: list[str] | tuple[str, ...] = (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ),
    camera_widths: int = 224,
    camera_heights: int = 224,
    layout_id: int | None = None,
    style_id: int | None = None,
    has_renderer: bool = False,
    has_offscreen_renderer: bool = True,
    use_camera_obs: bool = True,
    camera_depths: bool = False,
    control_freq: int = CONTROL_FREQUENCY_HZ,
):
    bootstrap_local_robocasa_sources()
    register_bench_tasks()
    import robosuite
    # The v2 data contract is OpenCV/top-left.  Set the RoboSuite macro before
    # constructing any environment so RGB, depth and calibration share one
    # image convention; recorders and evaluators must not flip frames again.
    try:
        from robosuite import macros
    except ImportError:
        # Lightweight callers may provide a robosuite make stub.  Real
        # RoboSuite always exposes macros; the stub does not need convention
        # configuration because it never renders an image.
        macros = None
    if macros is not None:
        macros.IMAGE_CONVENTION = "opencv"

    # bench-del1 又是重复的layout和style的设置
    if split == "target":
        obj_instance_split = "target"
        layout_ids = None
        style_ids = None
        layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))
    elif split == "pretrain":
        obj_instance_split = "pretrain"
        layout_ids = -2
        style_ids = -2
        layout_and_style_ids = None
    elif split == "all":
        obj_instance_split = None
        layout_ids = -3
        style_ids = -3
        layout_and_style_ids = None
    else:
        raise ValueError(f"unsupported split {split!r}")

    if layout_id is not None:
        layout_ids = int(layout_id)
        layout_and_style_ids = None
    if style_id is not None:
        style_ids = int(style_id)
        layout_and_style_ids = None
    # bench-del1 

    extra_env_kwargs = dict(env_kwargs or {})
    return robosuite.make(
        env_name=env_name,
        robots="PandaOmron",
        controller_configs=controller_config,
        camera_names=list(camera_names),
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=bool(has_renderer),
        has_offscreen_renderer=bool(has_offscreen_renderer),
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=bool(use_camera_obs),
        camera_depths=bool(camera_depths),
        control_freq=int(control_freq),
        seed=seed,
        obj_instance_split=obj_instance_split,
        generative_textures=None,
        randomize_cameras=False,
        layout_and_style_ids=layout_and_style_ids,
        layout_ids=layout_ids,
        style_ids=style_ids,
        translucent_robot=False,
        **extra_env_kwargs,
    )


def reset_env_with_initial_arm_qpos(env: Any) -> dict[str, Any]:
    """Reset env, then undo absolute JOINT_POSITION's zero-action reset settle."""
    env.reset()
    robot = env.robots[0]
    initial_qpos = np.array(robot.init_qpos, dtype=float)
    robot.set_robot_joint_positions(initial_qpos)
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    env.sim.forward()

    right = robot.composite_controller.part_controllers.get("right")
    if right is not None:
        right.update_initial_joints(initial_qpos)
        right.reset_goal()

    robot.composite_controller.reset()
    return env._get_observations(force_update=True)


def qpos_for_controller(env: Any, controller: Any) -> np.ndarray:
    return np.array(env.sim.data.qpos[controller.qpos_index], dtype=float)


def build_env_meta(
    *,
    env_name: str,
    controller_configs: dict[str, Any],
    camera_names: list[str] | tuple[str, ...],
    camera_width: int,
    camera_height: int,
    split: str,
    layout_id: int | None,
    style_id: int | None,
    env_kwargs: dict[str, Any] | None = None,
    camera_depths: bool = False,
    control_freq: int = CONTROL_FREQUENCY_HZ,
) -> dict[str, Any]:
    """Build environment metadata for the actual controller used to collect data."""
    bootstrap_local_robocasa_sources()
    try:
        import robocasa as _robocasa
    except Exception:
        _robocasa = None
    try:
        import robosuite as _robosuite
    except Exception:
        _robosuite = None

    kwargs_out: dict[str, Any] = {
        "env_name": env_name,
        "robots": "PandaOmron",
        "controller_configs": copy.deepcopy(controller_configs),
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": True,
        "camera_depths": bool(camera_depths),
        "translucent_robot": False,
        "camera_names": list(camera_names),
        "camera_heights": int(camera_height),
        "camera_widths": int(camera_width),
        "control_freq": int(control_freq),
    }
    #bench-del2
    if split == "target":
        kwargs_out["obj_instance_split"] = "target"
        kwargs_out["layout_and_style_ids"] = list(zip(range(1, 11), range(1, 11)))
    elif split == "pretrain":
        kwargs_out["obj_instance_split"] = "pretrain"
        kwargs_out["layout_ids"] = -2
        kwargs_out["style_ids"] = -2
    elif split == "all":
        kwargs_out["layout_ids"] = -3
        kwargs_out["style_ids"] = -3
    if layout_id is not None:
        kwargs_out["layout_ids"] = int(layout_id)
        kwargs_out.pop("layout_and_style_ids", None)
    if style_id is not None:
        kwargs_out["style_ids"] = int(style_id)
        kwargs_out.pop("layout_and_style_ids", None)
    #bench-del2
    for key, value in (env_kwargs or {}).items():
        kwargs_out.setdefault(key, copy.deepcopy(value))

    return {
        "env_name": env_name,
        "type": 1,
        "env_version": "" if _robocasa is None else getattr(_robocasa, "__version__", ""),
        "robosuite_version": "" if _robosuite is None else getattr(_robosuite, "__version__", ""),
        "env_kwargs": kwargs_out,
    }


__all__ = [
    "CONTROL_FREQUENCY_HZ",
    "PANDA_QPOS_HIGH",
    "PANDA_QPOS_LOW",
    "build_env_meta",
    "bootstrap_local_robocasa_sources",
    "make_joint_position_controller_config",
    "make_robocasa_env",
    "qpos_for_controller",
    "register_bench_tasks",
    "reset_env_with_initial_arm_qpos",
]
