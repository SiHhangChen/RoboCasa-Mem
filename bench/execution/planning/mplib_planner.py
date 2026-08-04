"""MPLib planning entry points.

This module owns the third-party MPLib call boundary. It accepts a plain plan
request dict and returns a plain result dict. It must not import task implementations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = [
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "robosuite",
    REPOSITORY_ROOT / "robocasa",
]
for source_path in SOURCE_PATHS:
    source_text = str(source_path)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

import numpy as np

from bench.execution.planning.config import MplibPlannerConfig
from bench.execution.planning.model_alignment import (
    build_joint_mapping,
    matrix_from_json_pose,
    matrix_from_pos_quat_wxyz,
    pose7_from_matrix_wxyz,
    pose_error,
    pose_from_matrix,
    rotation_error_deg,
    trajectory_for_robocasa,
)

DEFAULT_PLANNER_ARGS = MplibPlannerConfig().planner_args()


def _filter_conda_pythonpath_warning(stderr: str) -> str:
    """Drop conda-run noise from planner stderr while preserving real errors."""
    filtered = []
    skip_next_pythonpath_line = False
    for line in str(stderr).splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "WARNING: overwriting environment variables set in the machine":
            skip_next_pythonpath_line = True
            continue
        if skip_next_pythonpath_line and stripped == "overwriting variable ['PYTHONPATH']":
            skip_next_pythonpath_line = False
            continue
        skip_next_pythonpath_line = False
        filtered.append(line)
    return "".join(filtered)


def planner_args(payload: dict[str, Any]) -> dict[str, Any]:
    values = dict(DEFAULT_PLANNER_ARGS)
    values.update(dict(payload.get("planner_args") or {}))
    return values


def discover_model_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    panda_urdf_dir = (
        REPOSITORY_ROOT
        / "robosuite/robosuite/models/assets/bullet_data/panda_description/urdf"
    )
    repo_urdfs = [
        panda_urdf_dir / "panda_arm_hand.urdf",
        panda_urdf_dir / "panda_arm.urdf",
    ]
    for urdf in repo_urdfs:
        srdf = urdf.with_suffix(".srdf")
        candidates.append(
            {
                "source": "repo",
                "urdf": str(urdf),
                "srdf": str(srdf),
                "urdf_exists": urdf.exists(),
                "srdf_exists": srdf.exists(),
                "srdf_mode": "file" if srdf.exists() else "missing",
            }
        )
    try:
        import mani_skill

        panda_dir = Path(mani_skill.__file__).resolve().parent / "assets/robots/panda"
        for urdf in sorted(panda_dir.glob("*.urdf"), key=_panda_urdf_preference_key):
            srdf = urdf.with_suffix(".srdf")
            urdf_has_capsule = _urdf_contains_unsupported_capsule(urdf)
            candidates.append(
                {
                    "source": "mani_skill",
                    "urdf": str(urdf),
                    "srdf": str(srdf),
                    "urdf_exists": urdf.exists(),
                    "srdf_exists": srdf.exists(),
                    "srdf_mode": "file" if srdf.exists() else "missing",
                    "urdf_has_unsupported_capsule": urdf_has_capsule,
                }
            )
    except Exception as exc:
        candidates.append({"source": "mani_skill", "error": f"{type(exc).__name__}: {exc}"})
    return candidates


def _panda_urdf_preference_key(path: Path) -> tuple[int, str]:
    """Prefer standard Panda arm/hand models over task-tool variants."""
    preferred = {
        "panda_v2.urdf": 0,
        "panda_v3.urdf": 1,
        "panda_v2_gripper.urdf": 2,
        "mobile_panda_single_arm.urdf": 3,
        "panda_stick.urdf": 4,
        "panda_bucket.urdf": 5,
        "panda_pinch.urdf": 6,
        "mobile_panda_dual_arm.urdf": 7,
    }
    return (preferred.get(path.name, 100), path.name)


def _urdf_contains_unsupported_capsule(path: Path) -> bool:
    """Detect URDF geometry tags not accepted by MPLib's urdfdom parser."""
    try:
        return "<capsule" in path.read_text(errors="ignore")
    except OSError:
        return False


def instantiate_planner(
    mplib_module: Any,
    candidates: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any], str]:
    errors = []
    for candidate in candidates:
        if candidate.get("error"):
            errors.append(
                {
                    "source": candidate.get("source"),
                    "urdf": candidate.get("urdf"),
                    "srdf": candidate.get("srdf"),
                    "move_group": None,
                    "error": candidate["error"],
                }
            )
            continue
        if not candidate.get("urdf_exists"):
            errors.append(
                {
                    "source": candidate.get("source"),
                    "urdf": candidate.get("urdf"),
                    "srdf": candidate.get("srdf"),
                    "move_group": None,
                    "error": "skipped_missing_urdf",
                }
            )
            continue
        if not candidate.get("srdf_exists"):
            errors.append(
                {
                    "source": candidate.get("source"),
                    "urdf": candidate.get("urdf"),
                    "srdf": candidate.get("srdf"),
                    "move_group": None,
                    "error": "skipped_missing_srdf",
                }
            )
            continue
        if candidate.get("urdf_has_unsupported_capsule", False):
            errors.append(
                {
                    "urdf": candidate.get("urdf"),
                    "srdf": candidate.get("srdf"),
                    "move_group": None,
                    "error": "skipped_unsupported_urdf_capsule_geometry",
                }
            )
            continue
        for move_group in ("panda_hand_tcp", "panda_hand", "panda_link8"):
            try:
                planner = mplib_module.Planner(
                    urdf=candidate["urdf"],
                    srdf=candidate["srdf"],
                    move_group=move_group,
                )
                selected = dict(candidate)
                selected["srdf_mode"] = "file"
                selected["move_group_for_backend_init"] = move_group
                return planner, selected, "mplib_pinocchio"
            except Exception as exc:
                errors.append(
                    {
                        "urdf": candidate.get("urdf"),
                        "srdf": candidate.get("srdf"),
                        "move_group": move_group,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    raise RuntimeError("could not initialize MPlib Planner; errors=" + json.dumps(errors))


def candidate_link_names(link_names: list[str]) -> list[str]:
    preferred = [
        "panda_hand_tcp",
        "panda_hand",
        "panda_link8",
        "panda_link7",
        "panda_leftfinger",
        "panda_rightfinger",
        "panda_leftfinger_pad",
        "panda_rightfinger_pad",
    ]
    candidates = [name for name in preferred if name in link_names]
    for name in link_names:
        low = name.lower()
        if name not in candidates and any(
            token in low for token in ("tcp", "hand", "finger", "link8")
        ):
            candidates.append(name)
    return candidates


def qpos_for_mplib(
    planner: Any,
    sample: dict[str, Any],
    mapping: dict[str, Any],
    robocasa_joint_names: list[str],
) -> np.ndarray:
    sample_joint_values = np.array(
        sample["right_arm_qpos"] + sample["right_gripper_qpos"],
        dtype=float,
    )
    robocasa_values = {
        name: float(value)
        for name, value in zip(robocasa_joint_names, sample_joint_values)
    }
    qpos = np.zeros(len(planner.user_joint_names), dtype=float)
    mapping_by_mplib = {entry["mplib_joint_name"]: entry for entry in mapping["entries"]}
    for i, mplib_name in enumerate(planner.user_joint_names):
        entry = mapping_by_mplib.get(mplib_name)
        if entry and entry.get("status") == "ok":
            value = robocasa_values[entry["robocasa_joint_name"]]
            if entry.get("value_transform") == "absolute_opening_distance":
                value = abs(value)
            qpos[i] = value
        elif str(mplib_name).startswith("panda_finger_joint"):
            qpos[i] = 0.04
    return qpos


def evaluate_tcp_candidates(
    planner: Any,
    samples: list[dict[str, Any]],
    mapping: dict[str, Any],
    robocasa_joint_names: list[str],
) -> list[dict[str, Any]]:
    link_candidates = candidate_link_names(list(planner.user_link_names))
    link_name_to_idx = {name: idx for idx, name in enumerate(planner.user_link_names)}
    results = []
    for link_name in link_candidates:
        per_sample = []
        offsets = []
        for sample in samples:
            qpos = qpos_for_mplib(planner, sample, mapping, robocasa_joint_names)
            planner.pinocchio_model.compute_forward_kinematics(qpos)
            pose7 = np.array(
                planner.pinocchio_model.get_link_pose(link_name_to_idx[link_name]),
                dtype=float,
            )
            fk_pose = matrix_from_pos_quat_wxyz(pose7[:3], pose7[3:])
            target_pose = matrix_from_json_pose(sample["eef_site_in_base"])
            pos_err, ori_err = pose_error(fk_pose, target_pose)
            offset = np.linalg.inv(fk_pose) @ target_pose
            offsets.append(offset)
            per_sample.append(
                {
                    "sample": sample["name"],
                    "fk_pose": pose_from_matrix(fk_pose),
                    "target_eef_in_base": sample["eef_site_in_base"],
                    "position_error_m": pos_err,
                    "orientation_error_deg": ori_err,
                    "qpos_used": qpos.astype(float).tolist(),
                }
            )
        position_errors = [item["position_error_m"] for item in per_sample]
        orientation_errors = [item["orientation_error_deg"] for item in per_sample]
        offset_translations = np.array([offset[:3, 3] for offset in offsets], dtype=float)
        offset_mean_translation = np.mean(offset_translations, axis=0)
        offset_translation_variation = float(
            np.max(np.linalg.norm(offset_translations - offset_mean_translation, axis=1))
        )
        offset_rotation_variations = [
            rotation_error_deg(offsets[0][:3, :3], offset[:3, :3])
            for offset in offsets
        ]
        offset_mean = np.eye(4)
        offset_mean[:3, :3] = offsets[0][:3, :3]
        offset_mean[:3, 3] = offset_mean_translation
        results.append(
            {
                "candidate_link": link_name,
                "per_sample": per_sample,
                "max_position_error_m": float(np.max(position_errors)),
                "mean_position_error_m": float(np.mean(position_errors)),
                "max_orientation_error_deg": float(np.max(orientation_errors)),
                "mean_orientation_error_deg": float(np.mean(orientation_errors)),
                "fixed_offset": {
                    "mean_pose": pose_from_matrix(offset_mean),
                    "translation_variation_m": offset_translation_variation,
                    "rotation_variation_deg": float(np.max(offset_rotation_variations)),
                },
            }
        )
    results.sort(
        key=lambda item: (
            item["max_position_error_m"],
            item["max_orientation_error_deg"],
            item["fixed_offset"]["translation_variation_m"],
        )
    )
    return results


def select_tcp_candidate(
    tcp_candidates: list[dict[str, Any]],
    args: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    forced_link = str(args.get("force_tcp_candidate_link", "") or "").strip()
    if forced_link:
        forced = [
            item for item in tcp_candidates if item["candidate_link"] == forced_link
        ]
        if forced:
            return forced[0], "forced"
        return None, f"forced_tcp_candidate_not_found:{forced_link}"

    direct_candidates = [
        item
        for item in tcp_candidates
        if item["max_position_error_m"] < args["direct_pos_threshold"]
        and item["max_orientation_error_deg"] < args["direct_ori_threshold_deg"]
    ]
    offset_candidates = [
        item
        for item in tcp_candidates
        if item["fixed_offset"]["translation_variation_m"]
        < args["offset_pos_variation_threshold"]
        and item["fixed_offset"]["rotation_variation_deg"]
        < args["offset_ori_variation_threshold_deg"]
    ]
    if direct_candidates:
        return direct_candidates[0], "direct"
    if offset_candidates:
        offset_candidates.sort(
            key=lambda item: (
                item["fixed_offset"]["translation_variation_m"],
                item["fixed_offset"]["rotation_variation_deg"],
                item["mean_position_error_m"],
            )
        )
        return offset_candidates[0], "stable_fixed_offset"
    return None, None


def collision_scene_summary(collision_scene: dict[str, Any] | None) -> dict[str, Any] | None:
    if collision_scene is None:
        return None
    return {key: value for key, value in collision_scene.items() if key != "points"}


def update_planner_point_cloud(
    planner: Any,
    collision_scene: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = collision_scene_summary(collision_scene) or {"enabled": False}
    summary["update_success"] = False
    summary["use_point_cloud"] = False
    if not collision_scene or not bool(collision_scene.get("enabled", False)):
        return summary
    if collision_scene.get("frame") != "robot_base":
        summary["reason"] = f"unsupported_point_cloud_frame:{collision_scene.get('frame')!r}"
        return summary

    points = np.array(collision_scene.get("points", []), dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) == 0
        or not np.all(np.isfinite(points))
    ):
        summary["reason"] = "invalid_or_empty_points"
        return summary

    resolution = float(
        collision_scene.get(
            "point_radius",
            collision_scene.get("resolution", collision_scene.get("spacing", 0.02)),
        )
    )
    errors = []
    for kwargs in ({"resolution": resolution}, {"radius": resolution}, {}):
        try:
            planner.update_point_cloud(points, **kwargs)
            summary["update_success"] = True
            summary["use_point_cloud"] = True
            summary["applied_point_count"] = int(len(points))
            summary["api_kwargs"] = kwargs
            return summary
        except TypeError as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            summary["reason"] = f"{type(exc).__name__}: {exc}"
            summary["errors"] = errors
            return summary

    summary["reason"] = "update_point_cloud_signature_mismatch"
    summary["errors"] = errors
    return summary


def plan_tcp_target_in_mplib(payload: dict[str, Any]) -> dict[str, Any]:
    import mplib

    args = planner_args(payload)
    model_candidates = payload.get("model_candidates") or discover_model_candidates()
    planner, selected_model, fk_backend = instantiate_planner(mplib, model_candidates)
    robocasa_joint_names = (
        payload["robocasa"]["right_arm_joint_names"]
        + payload["robocasa"]["right_gripper_joint_names"]
    )
    joint_mapping = build_joint_mapping(
        list(planner.user_joint_names),
        payload["robocasa"]["right_arm_joint_names"],
        payload["robocasa"]["right_gripper_joint_names"],
    )
    tcp_candidates = evaluate_tcp_candidates(
        planner,
        payload["samples"],
        joint_mapping,
        robocasa_joint_names,
    )
    selected_tcp, selected_reason = select_tcp_candidate(tcp_candidates, args)
    if selected_tcp is not None and selected_tcp["candidate_link"] != planner.move_group:
        planner = mplib.Planner(
            urdf=selected_model["urdf"],
            srdf=selected_model["srdf"],
            move_group=selected_tcp["candidate_link"],
        )

    point_cloud_summary = update_planner_point_cloud(planner, payload.get("collision_scene"))
    use_point_cloud = bool(point_cloud_summary.get("use_point_cloud", False))
    point_cloud_required = bool(payload.get("collision_scene", {}).get("enabled", False))
    point_cloud_failed = bool(point_cloud_required and not use_point_cloud)

    if point_cloud_failed:
        plan_result = {
            "status": "Point cloud update failed",
            "attempts": [],
            "collision_scene": point_cloud_summary,
        }
    elif selected_tcp is None:
        plan_result = {
            "status": "No TCP candidate",
            "attempts": [],
        }
    else:
        plan_result = _plan_with_selected_tcp(
            planner,
            payload,
            args,
            selected_tcp=selected_tcp,
            joint_mapping=joint_mapping,
            robocasa_joint_names=robocasa_joint_names,
            use_point_cloud=use_point_cloud,
        )

    return {
        "environment": {
            "collector": payload.get("collector_environment", {}),
            "analyzer": {
                "sys_executable": sys.executable,
                "pythonpath_head": sys.path[:5],
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "mplib_file": getattr(mplib, "__file__", None),
                "fk_backend": fk_backend,
            },
        },
        "model_files": {
            "candidates": model_candidates,
            "selected": selected_model,
            "mplib_user_link_names": list(planner.user_link_names),
            "mplib_user_joint_names": list(planner.user_joint_names),
            "mplib_move_group_joint_names": list(planner.robot.get_move_group_joint_names()),
        },
        "robocasa": payload["robocasa"],
        "alignment_execution": payload.get("alignment_execution", {}),
        "target": payload["target"],
        "collision_scene": point_cloud_summary,
        "joint_mapping": joint_mapping,
        "tcp_candidates": tcp_candidates,
        "selected_tcp_candidate": _selected_tcp_summary(selected_tcp, selected_reason),
        "plan": plan_result,
    }


def _plan_with_selected_tcp(
    planner: Any,
    payload: dict[str, Any],
    args: dict[str, Any],
    *,
    selected_tcp: dict[str, Any],
    joint_mapping: dict[str, Any],
    robocasa_joint_names: list[str],
    use_point_cloud: bool,
) -> dict[str, Any]:
    current_sample = payload["samples"][-1]
    trajectory_start_qpos = np.array(
        qpos_for_mplib(planner, current_sample, joint_mapping, robocasa_joint_names),
        dtype=float,
    )
    target_eef = matrix_from_json_pose(payload["target"]["target_eef_site_in_base"])
    fixed_offset = matrix_from_json_pose(selected_tcp["fixed_offset"]["mean_pose"])
    target_link = target_eef @ np.linalg.inv(fixed_offset)
    target_pose7 = pose7_from_matrix_wxyz(target_link)

    attempts = []
    result = planner.plan_screw(
        np.array(target_pose7, dtype=float),
        trajectory_start_qpos.copy(),
        qpos_step=args["qpos_step"],
        time_step=args["time_step"],
        use_point_cloud=use_point_cloud,
        use_attach=False,
        wrt_world=False,
        verbose=False,
    )
    attempts.append(
        {
            "method": "plan_screw",
            "status": result.get("status"),
            "use_point_cloud": use_point_cloud,
        }
    )
    if result.get("status") != "Success":
        preserve_stages = {
            str(value).strip().lower()
            for value in args.get("preserve_ik_branch_stages", [])
            if str(value).strip()
        }
        target_stage = str((payload.get("target") or {}).get("stage", "")).lower()
        if target_stage in preserve_stages:
            result = _plan_qpos_to_pose_nearest_ik(
                planner,
                target_pose7=np.array(target_pose7, dtype=float),
                trajectory_start_qpos=trajectory_start_qpos,
                args=args,
                use_point_cloud=use_point_cloud,
            )
            fallback_method = "plan_qpos_to_nearest_ik"
        else:
            result = planner.plan_qpos_to_pose(
                np.array(target_pose7, dtype=float),
                trajectory_start_qpos.copy(),
                time_step=args["time_step"],
                rrt_range=args["rrt_range"],
                planning_time=args["planning_time"],
                use_point_cloud=use_point_cloud,
                use_attach=False,
                wrt_world=False,
                verbose=False,
            )
            fallback_method = "plan_qpos_to_pose"
        attempts.append(
            {
                "method": fallback_method,
                "status": result.get("status"),
                "use_point_cloud": use_point_cloud,
                "ik_branch_selection": result.get("ik_branch_selection"),
            }
        )

    if result.get("status") != "Success":
        return {"status": result.get("status", "unknown"), "attempts": attempts}

    positions = np.array(result["position"], dtype=float)
    move_group_joint_names = list(planner.robot.get_move_group_joint_names())
    trajectory = trajectory_for_robocasa(
        positions,
        move_group_joint_names,
        joint_mapping,
        payload["robocasa"]["right_arm_joint_names"],
    )
    return {
        "status": "Success",
        "attempts": attempts,
        "trajectory": trajectory,
        "trajectory_len": len(trajectory),
        "duration": float(result.get("duration", 0.0)),
        "move_group_joint_names": move_group_joint_names,
        "target_mplib_link_pose": pose_from_matrix(target_link),
        "target_mplib_pose7_wxyz": target_pose7,
        "start_full_qpos": trajectory_start_qpos.astype(float).tolist(),
    }


def _plan_qpos_to_pose_nearest_ik(
    planner: Any,
    *,
    target_pose7: np.ndarray,
    trajectory_start_qpos: np.ndarray,
    args: dict[str, Any],
    use_point_cloud: bool,
) -> dict[str, Any]:
    """Plan to the IK solution closest to the current joint branch."""
    ik_status, solutions = planner.IK(
        target_pose7.copy(),
        trajectory_start_qpos.copy(),
        n_init_qpos=int(args.get("nearest_ik_samples", 64)),
        threshold=float(args.get("nearest_ik_threshold", 1e-3)),
    )
    if ik_status != "Success" or not solutions:
        return {
            "status": ik_status,
            "ik_branch_selection": {
                "mode": "nearest_current_qpos",
                "candidate_count": int(len(solutions or [])),
            },
        }
    indices = np.array(planner.move_group_joint_indices, dtype=int)
    start = np.array(trajectory_start_qpos, dtype=float)
    ranked = []
    for solution in solutions:
        candidate = np.array(solution, dtype=float)
        delta = candidate[indices] - start[indices]
        ranked.append(
            (
                float(np.linalg.norm(delta)),
                float(np.max(np.abs(delta))),
                candidate,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]))
    distance, max_delta, selected = ranked[0]
    max_allowed_delta = float(args.get("nearest_ik_max_joint_delta", 2.0))
    selection = {
        "mode": "nearest_current_qpos",
        "candidate_count": int(len(ranked)),
        "selected_l2_joint_delta": distance,
        "selected_max_joint_delta": max_delta,
        "max_allowed_joint_delta": max_allowed_delta,
    }
    if max_delta > max_allowed_delta:
        return {
            "status": "Nearest IK branch exceeds joint-delta limit",
            "ik_branch_selection": selection,
        }
    result = planner.plan_qpos_to_qpos(
        [selected],
        start.copy(),
        time_step=args["time_step"],
        rrt_range=args["rrt_range"],
        planning_time=args["planning_time"],
        use_point_cloud=use_point_cloud,
        use_attach=False,
        verbose=False,
    )
    result["ik_branch_selection"] = selection
    return result


def _selected_tcp_summary(
    selected_tcp: dict[str, Any] | None,
    selected_reason: str | None,
) -> dict[str, Any] | None:
    if selected_tcp is None:
        return None
    return {
        "candidate_link": selected_tcp["candidate_link"],
        "reason": selected_reason,
        "max_position_error_m": selected_tcp["max_position_error_m"],
        "max_orientation_error_deg": selected_tcp["max_orientation_error_deg"],
        "fixed_offset": selected_tcp["fixed_offset"],
    }


def plan_tcp_delta_in_mplib(payload: dict[str, Any]) -> dict[str, Any]:
    return plan_tcp_target_in_mplib(payload)


def run_mplib_planner_subprocess(
    *,
    script_path: Path,
    plan_request: dict[str, Any],
    conda_env: str = "membench",
    extra_python_paths: list[Path] | None = None,
    timeout_s: float | None = None,
) -> tuple[int, dict[str, Any] | None]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir="/tmp") as handle:
        json.dump(plan_request, handle)
        request_path = Path(handle.name)
    try:
        requested_env = str(conda_env).strip()
        active_env_names = {
            str(os.environ.get("CONDA_DEFAULT_ENV", "")).strip(),
            Path(sys.prefix).name,
        }
        if requested_env in active_env_names:
            cmd = [
                sys.executable,
                str(script_path),
                "--plan-json",
                str(request_path),
            ]
        else:
            cmd = [
                "conda",
                "run",
                "-n",
                requested_env,
                "python",
                str(script_path),
                "--plan-json",
                str(request_path),
            ]
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = "/tmp/matplotlib"
        local_paths = [*SOURCE_PATHS, *(extra_python_paths or [])]
        local_pythonpath = os.pathsep.join(str(path) for path in local_paths)
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = local_pythonpath + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = local_pythonpath
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPOSITORY_ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stderr:
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
                stderr = _filter_conda_pythonpath_warning(stderr)
                if stderr:
                    print(stderr, end="", file=sys.stderr)
            return -1, {
                "plan": {
                    "status": "planner_subprocess_timeout",
                    "attempts": [],
                    "timeout_s": timeout_s,
                },
                "subprocess_timeout_s": timeout_s,
            }
        if proc.stderr:
            stderr = _filter_conda_pythonpath_warning(proc.stderr)
            if stderr:
                print(stderr, end="", file=sys.stderr)
        if not proc.stdout:
            return int(proc.returncode), None
        try:
            return int(proc.returncode), json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(proc.stdout, end="", file=sys.stderr)
            return int(proc.returncode), None
    finally:
        try:
            request_path.unlink()
        except OSError:
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.plan_json.read_text())
    result = plan_tcp_target_in_mplib(payload)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0 if (result.get("plan") or {}).get("status") == "Success" else 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


__all__ = [
    "DEFAULT_PLANNER_ARGS",
    "candidate_link_names",
    "collision_scene_summary",
    "discover_model_candidates",
    "evaluate_tcp_candidates",
    "instantiate_planner",
    "plan_tcp_delta_in_mplib",
    "plan_tcp_target_in_mplib",
    "planner_args",
    "qpos_for_mplib",
    "run_mplib_planner_subprocess",
    "select_tcp_candidate",
    "update_planner_point_cloud",
]


if __name__ == "__main__":
    raise SystemExit(main())
