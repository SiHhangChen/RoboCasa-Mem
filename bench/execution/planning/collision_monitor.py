"""MuJoCo contact monitoring for planned RoboCasa arm/base trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from bench.execution.planning.static_scene_point_cloud import parse_keyword_csv


ROBOT_GEOM_EXCLUDE_KEYWORDS = (
    "base",
    "wheel",
    "torso",
    "head",
    "camera",
    "caster",
)


def object_contact_geoms(env: Any, obj_name: str | None) -> set[str]:
    if obj_name is None:
        return set()
    obj = (getattr(env, "objects", {}) or {}).get(obj_name)
    if obj is None:
        return set()
    return _existing_geom_names(env, getattr(obj, "contact_geoms", []) or [])


def _eef_site_matrix(env: Any) -> np.ndarray:
    robot = env.robots[0]
    site_id = robot.eef_site_id["right"]
    mat = np.eye(4, dtype=float)
    mat[:3, 3] = np.array(env.sim.data.site_xpos[site_id], dtype=float)
    mat[:3, :3] = np.array(env.sim.data.site_xmat[site_id].reshape(3, 3), dtype=float)
    return mat


def _object_collision_radius(
    env: Any,
    obj_name: str,
    *,
    inflate: float = 0.0,
) -> tuple[float, str]:
    obj = (getattr(env, "objects", {}) or {}).get(obj_name)
    if obj is None:
        return float(max(0.01, inflate)), "object_unavailable"
    pos = None
    try:
        pos = np.array(env.sim.data.body_xpos[env.obj_body_id[obj_name]], dtype=float)
    except Exception:
        pos = None
    if pos is not None:
        try:
            quat_wxyz = np.array(env.sim.data.body_xquat[env.obj_body_id[obj_name]], dtype=float)
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            points = np.array(obj.get_bbox_points(trans=pos, rot=quat_xyzw), dtype=float)
            if points.shape == (8, 3) and np.all(np.isfinite(points)):
                center = np.mean(points, axis=0)
                radius = float(np.max(np.linalg.norm(points - center, axis=1)) + inflate)
                return radius, "object_bbox"
        except Exception:
            pass
    size = getattr(obj, "size", None)
    try:
        arr = np.array(size, dtype=float).reshape(-1)
        if arr.size >= 3 and np.all(np.isfinite(arr)):
            return float(np.linalg.norm(0.5 * arr[:3]) + inflate), "object_size"
    except Exception:
        pass
    radius = getattr(obj, "horizontal_radius", None)
    try:
        return float(max(float(radius), 0.01) + inflate), "object_horizontal_radius"
    except Exception:
        return float(0.04 + inflate), "fallback_radius"


def _fixture_aabb(env: Any, fixture: Any) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    try:
        points = np.array(fixture.get_bbox_points(), dtype=float)
        if points.shape == (8, 3) and np.all(np.isfinite(points)):
            return np.min(points, axis=0), np.max(points, axis=0), "fixture.get_bbox_points"
    except Exception:
        pass
    pos = None
    size = None
    try:
        pos = np.array(getattr(fixture, "pos", None), dtype=float).reshape(-1)
    except Exception:
        pos = None
    try:
        size = np.array(getattr(fixture, "size", None), dtype=float).reshape(-1)
    except Exception:
        size = None
    if pos is None or pos.size < 2 or size is None or size.size < 3:
        root_body = getattr(fixture, "root_body", None) or getattr(fixture, "name", None)
        if root_body is not None:
            try:
                pos = np.array(env.sim.data.get_body_xpos(root_body), dtype=float).reshape(-1)
            except Exception:
                pos = None
    if pos is None or pos.size < 2 or size is None or size.size < 3:
        return None, None, "unavailable"
    if pos.size == 2:
        pos = np.array([pos[0], pos[1], 0.0], dtype=float)
    half = 0.5 * size[:3]
    return pos[:3] - half, pos[:3] + half, "fixture.pos_size_aabb"


def _existing_geom_names(env: Any, names: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    existing = set()
    for name in names:
        try:
            env.sim.model.geom_name2id(name)
            existing.add(str(name))
        except Exception:
            pass
    return existing


def _fixture_matches(name: str, fixture: Any, include: list[str], exclude: list[str]) -> bool:
    haystack = f"{name.lower()} {fixture.__class__.__name__.lower()}"
    if any(keyword in haystack for keyword in exclude):
        return False
    if not include or "*" in include or "all" in include:
        return True
    return any(keyword in haystack for keyword in include)


def _fixture_allowed(name: str, fixture: Any, allowed: list[str]) -> bool:
    if not allowed:
        return False
    if "*" in allowed or "all" in allowed:
        return True
    haystack = f"{name.lower()} {fixture.__class__.__name__.lower()}"
    return any(keyword in haystack for keyword in allowed)


def static_fixture_contact_geoms(
    env: Any,
    *,
    fixture_keywords: str | list[str] | tuple[str, ...] = "cabinet",
    exclude_keywords: str | list[str] | tuple[str, ...] = "floor,wall",
) -> tuple[set[str], list[dict[str, Any]]]:
    include = parse_keyword_csv(fixture_keywords)
    exclude = parse_keyword_csv(exclude_keywords)
    geoms: set[str] = set()
    fixtures = []
    for fixture_name, fixture in (getattr(env, "fixtures", {}) or {}).items():
        if not _fixture_matches(str(fixture_name), fixture, include, exclude):
            continue
        fixture_geoms = _existing_geom_names(env, getattr(fixture, "contact_geoms", []) or [])
        if not fixture_geoms:
            continue
        geoms.update(fixture_geoms)
        fixtures.append(
            {
                "name": str(fixture_name),
                "class": fixture.__class__.__name__,
                "contact_geom_count": int(len(fixture_geoms)),
            }
        )
    return geoms, fixtures


def allowed_fixture_contact_geoms(
    env: Any,
    *,
    fixture_keywords: str | list[str] | tuple[str, ...] = "cabinet",
    exclude_keywords: str | list[str] | tuple[str, ...] = "floor,wall",
    allowed_fixture_keywords: str | list[str] | tuple[str, ...] = "",
) -> tuple[set[str], list[dict[str, Any]]]:
    include = parse_keyword_csv(fixture_keywords)
    exclude = parse_keyword_csv(exclude_keywords)
    allowed = parse_keyword_csv(allowed_fixture_keywords)
    geoms: set[str] = set()
    fixtures = []
    if not allowed:
        return geoms, fixtures
    for fixture_name, fixture in (getattr(env, "fixtures", {}) or {}).items():
        name = str(fixture_name)
        if not _fixture_matches(name, fixture, include, exclude):
            continue
        if not _fixture_allowed(name, fixture, allowed):
            continue
        fixture_geoms = _existing_geom_names(env, getattr(fixture, "contact_geoms", []) or [])
        geoms.update(fixture_geoms)
        fixtures.append(
            {
                "name": name,
                "class": fixture.__class__.__name__,
                "contact_geom_count": int(len(fixture_geoms)),
            }
        )
    return geoms, fixtures


def robot_arm_contact_geoms(env: Any) -> set[str]:
    robot = env.robots[0]
    candidates = set(getattr(robot.robot_model, "contact_geoms", []) or [])
    right_gripper = getattr(robot, "gripper", {}).get("right")
    if right_gripper is not None:
        candidates.update(getattr(right_gripper, "contact_geoms", []) or [])

    geoms = set()
    for geom in candidates:
        lower = str(geom).lower()
        if any(keyword in lower for keyword in ROBOT_GEOM_EXCLUDE_KEYWORDS):
            continue
        geoms.add(str(geom))
    return _existing_geom_names(env, geoms)


def robot_base_contact_geoms(env: Any) -> set[str]:
    robot = env.robots[0]
    geoms = set()
    base_model = getattr(robot.robot_model, "base", None)
    if base_model is not None:
        geoms.update(getattr(base_model, "contact_geoms", []) or [])
    for geom in getattr(robot.robot_model, "contact_geoms", []) or []:
        lower = str(geom).lower()
        if any(keyword in lower for keyword in ("base", "mobile", "pedestal", "wheel", "caster")):
            geoms.add(str(geom))
    return _existing_geom_names(env, geoms)


def contact_pairs_between(
    env: Any,
    robot_geoms: set[str],
    obstacle_geoms: set[str],
) -> list[dict[str, Any]]:
    pairs = []
    if not robot_geoms or not obstacle_geoms:
        return pairs
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1 = env.sim.model.geom_id2name(contact.geom1)
        geom2 = env.sim.model.geom_id2name(contact.geom2)
        if geom1 is None or geom2 is None:
            continue
        hit = (geom1 in robot_geoms and geom2 in obstacle_geoms) or (
            geom2 in robot_geoms and geom1 in obstacle_geoms
        )
        if not hit:
            continue
        pairs.append(
            {
                "geom1": geom1,
                "geom2": geom2,
                "dist": float(contact.dist),
                "pos": np.array(contact.pos, dtype=float).tolist(),
            }
        )
    return pairs


def make_collision_monitor(
    env: Any,
    *,
    enabled: bool,
    fixture_keywords: str | list[str] | tuple[str, ...] = "cabinet",
    exclude_keywords: str | list[str] | tuple[str, ...] = "floor,wall",
    abort_on_collision: bool = True,
    holding_object: str | None = None,
    attached_object_proxy: bool = True,
    attached_object_proxy_inflate: float = 0.02,
    holding_allowed_fixture_keywords: str | list[str] | tuple[str, ...] = "",
    protected_object_names: str | list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    robot_geoms = robot_arm_contact_geoms(env)
    fixture_obstacle_geoms, fixtures = static_fixture_contact_geoms(
        env,
        fixture_keywords=fixture_keywords,
        exclude_keywords=exclude_keywords,
    )
    holding_allowed_geoms, holding_allowed_fixtures = allowed_fixture_contact_geoms(
        env,
        fixture_keywords=fixture_keywords,
        exclude_keywords=exclude_keywords,
        allowed_fixture_keywords=holding_allowed_fixture_keywords,
    )
    held_geoms = object_contact_geoms(env, holding_object)
    protected_names = _object_name_sequence(protected_object_names)
    protected_objects = []
    protected_geoms: set[str] = set()
    for object_name in protected_names:
        object_geoms = object_contact_geoms(env, object_name)
        protected_geoms.update(object_geoms)
        protected_objects.append(
            {
                "name": object_name,
                "contact_geom_count": int(len(object_geoms)),
            }
        )
    robot_obstacle_geoms = set(fixture_obstacle_geoms) | protected_geoms
    held_obstacle_geoms = set(fixture_obstacle_geoms) - set(holding_allowed_geoms)
    proxy_summary = _attached_object_proxy_summary(
        env,
        holding_object,
        enabled=bool(holding_object is not None and attached_object_proxy),
        fixture_keywords=fixture_keywords,
        exclude_keywords=exclude_keywords,
        inflate=float(attached_object_proxy_inflate),
        allowed_fixture_keywords=holding_allowed_fixture_keywords,
    )
    return {
        "enabled": True,
        "abort_on_collision": bool(abort_on_collision),
        "fixture_keywords": parse_keyword_csv(fixture_keywords),
        "exclude_keywords": parse_keyword_csv(exclude_keywords),
        "holding_allowed_fixture_keywords": parse_keyword_csv(holding_allowed_fixture_keywords),
        "holding_allowed_fixtures": holding_allowed_fixtures,
        "robot_geoms": sorted(robot_geoms),
        "held_object": holding_object,
        "held_object_geoms": sorted(held_geoms),
        "attached_object_proxy": proxy_summary,
        "obstacle_geoms": sorted(robot_obstacle_geoms),
        "fixture_obstacle_geoms": sorted(fixture_obstacle_geoms),
        "protected_object_names": protected_names,
        "protected_objects": protected_objects,
        "protected_object_geoms": sorted(protected_geoms),
        "held_obstacle_geoms": sorted(held_obstacle_geoms),
        "holding_allowed_obstacle_geoms": sorted(holding_allowed_geoms),
        "fixtures": fixtures,
    }


def _attached_object_proxy_summary(
    env: Any,
    holding_object: str | None,
    *,
    enabled: bool,
    fixture_keywords: str | list[str] | tuple[str, ...],
    exclude_keywords: str | list[str] | tuple[str, ...],
    inflate: float,
    allowed_fixture_keywords: str | list[str] | tuple[str, ...] = "",
) -> dict[str, Any] | None:
    if not enabled or holding_object is None:
        return None
    radius, radius_source = _object_collision_radius(env, holding_object, inflate=float(inflate))
    fixture_aabbs = []
    allowed_fixture_aabbs = []
    include = parse_keyword_csv(fixture_keywords)
    exclude = parse_keyword_csv(exclude_keywords)
    allowed = parse_keyword_csv(allowed_fixture_keywords)
    for fixture_name, fixture in (getattr(env, "fixtures", {}) or {}).items():
        name = str(fixture_name)
        if not _fixture_matches(name, fixture, include, exclude):
            continue
        bbox_min, bbox_max, bbox_source = _fixture_aabb(env, fixture)
        if bbox_min is None or bbox_max is None:
            continue
        record = {
            "name": name,
            "class": fixture.__class__.__name__,
            "bbox_source": bbox_source,
            "bbox_min": bbox_min.astype(float).tolist(),
            "bbox_max": bbox_max.astype(float).tolist(),
        }
        if _fixture_allowed(name, fixture, allowed):
            allowed_fixture_aabbs.append(record)
        else:
            fixture_aabbs.append(record)
    try:
        object_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[holding_object]], dtype=float)
        eef_to_world = _eef_site_matrix(env)
        center_offset_eef = (np.linalg.inv(eef_to_world) @ np.r_[object_pos, 1.0])[:3]
        return {
            "enabled": True,
            "object_name": holding_object,
            "radius": float(radius),
            "radius_source": radius_source,
            "center_offset_eef": center_offset_eef.astype(float).tolist(),
            "allowed_fixture_keywords": allowed,
            "allowed_fixture_aabb_count": int(len(allowed_fixture_aabbs)),
            "allowed_fixture_names": [
                str(item.get("name")) for item in allowed_fixture_aabbs
            ],
            "fixture_aabbs": fixture_aabbs,
        }
    except Exception as exc:
        return {
            "enabled": False,
            "object_name": holding_object,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def make_base_path_monitor(
    env: Any,
    *,
    enabled: bool,
    fixture_keywords: str | list[str] | tuple[str, ...] = "cabinet",
    exclude_keywords: str | list[str] | tuple[str, ...] = "floor,wall",
    holding_object: str | None = None,
    abort_on_collision: bool = True,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    base_geoms = robot_base_contact_geoms(env)
    obstacle_geoms, fixtures = static_fixture_contact_geoms(
        env,
        fixture_keywords=fixture_keywords,
        exclude_keywords=exclude_keywords,
    )
    held_geoms = object_contact_geoms(env, holding_object)
    return {
        "enabled": True,
        "abort_on_collision": bool(abort_on_collision),
        "fixture_keywords": parse_keyword_csv(fixture_keywords),
        "exclude_keywords": parse_keyword_csv(exclude_keywords),
        "base_geoms": sorted(base_geoms),
        "held_object": holding_object,
        "held_object_geoms": sorted(held_geoms),
        "obstacle_geoms": sorted(obstacle_geoms),
        "fixtures": fixtures,
    }


def base_path_monitor_summary(monitor: dict[str, Any] | None) -> dict[str, Any] | None:
    if monitor is None:
        return None
    if not monitor.get("enabled", False):
        return {"enabled": False}
    return {
        "enabled": True,
        "abort_on_collision": bool(monitor.get("abort_on_collision", True)),
        "fixture_keywords": monitor.get("fixture_keywords", []),
        "exclude_keywords": monitor.get("exclude_keywords", []),
        "base_geom_count": len(monitor.get("base_geoms", [])),
        "held_object": monitor.get("held_object"),
        "held_object_geom_count": len(monitor.get("held_object_geoms", [])),
        "obstacle_geom_count": len(monitor.get("obstacle_geoms", [])),
        "fixtures": monitor.get("fixtures", []),
    }


def check_base_path_collision(env: Any, monitor: dict[str, Any] | None) -> dict[str, Any]:
    if monitor is None or not monitor.get("enabled", False):
        return {"enabled": False, "collision": False, "base_pairs": [], "held_object_pairs": []}
    obstacle_geoms = set(monitor.get("obstacle_geoms", []))
    base_pairs = contact_pairs_between(env, set(monitor.get("base_geoms", [])), obstacle_geoms)
    held_pairs = contact_pairs_between(
        env,
        set(monitor.get("held_object_geoms", [])),
        obstacle_geoms,
    )
    return {
        "enabled": True,
        "collision": bool(base_pairs or held_pairs),
        "base_pairs": base_pairs,
        "held_object_pairs": held_pairs,
    }


def collision_monitor_summary(monitor: dict[str, Any] | None) -> dict[str, Any] | None:
    if monitor is None:
        return None
    if not monitor.get("enabled", False):
        return {"enabled": False}
    return {
        "enabled": True,
        "abort_on_collision": bool(monitor.get("abort_on_collision", True)),
        "fixture_keywords": monitor.get("fixture_keywords", []),
        "exclude_keywords": monitor.get("exclude_keywords", []),
        "holding_allowed_fixture_keywords": monitor.get("holding_allowed_fixture_keywords", []),
        "holding_allowed_fixture_count": len(monitor.get("holding_allowed_fixtures", [])),
        "robot_geom_count": len(monitor.get("robot_geoms", [])),
        "held_object": monitor.get("held_object"),
        "held_object_geom_count": len(monitor.get("held_object_geoms", [])),
        "attached_object_proxy": _attached_object_proxy_summary_without_aabbs(monitor),
        "obstacle_geom_count": len(monitor.get("obstacle_geoms", [])),
        "fixture_obstacle_geom_count": len(monitor.get("fixture_obstacle_geoms", [])),
        "protected_object_names": monitor.get("protected_object_names", []),
        "protected_object_geom_count": len(monitor.get("protected_object_geoms", [])),
        "held_obstacle_geom_count": len(monitor.get("held_obstacle_geoms", [])),
        "fixtures": monitor.get("fixtures", []),
    }


def _attached_object_proxy_summary_without_aabbs(
    monitor: dict[str, Any],
) -> dict[str, Any] | None:
    proxy = monitor.get("attached_object_proxy")
    if proxy is None:
        return None
    return {key: value for key, value in proxy.items() if key != "fixture_aabbs"}


def _object_name_sequence(value: str | list[str] | tuple[str, ...]) -> list[str]:
    raw = value.replace(";", ",").split(",") if isinstance(value, str) else value
    return [str(item).strip() for item in raw if str(item).strip()]


def attached_object_proxy_collisions(
    env: Any,
    monitor: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if monitor is None or not monitor.get("enabled", False):
        return []
    proxy = monitor.get("attached_object_proxy") or {}
    if not proxy.get("enabled", False):
        return []
    eef_to_world = _eef_site_matrix(env)
    center_offset_eef = np.array(proxy.get("center_offset_eef", [0.0, 0.0, 0.0]), dtype=float)
    center_world = (eef_to_world @ np.r_[center_offset_eef, 1.0])[:3]
    radius = float(proxy.get("radius", 0.0))
    pairs = []
    for fixture in proxy.get("fixture_aabbs", []):
        bbox_min = np.array(fixture.get("bbox_min"), dtype=float)
        bbox_max = np.array(fixture.get("bbox_max"), dtype=float)
        closest = np.minimum(np.maximum(center_world, bbox_min), bbox_max)
        distance = float(np.linalg.norm(center_world - closest))
        if distance > radius:
            continue
        pairs.append(
            {
                "held_object": proxy.get("object_name"),
                "fixture": fixture.get("name"),
                "fixture_class": fixture.get("class"),
                "distance": distance,
                "radius": radius,
                "center_world": center_world.astype(float).tolist(),
                "closest_point": closest.astype(float).tolist(),
            }
        )
    return pairs


def check_monitor_collision(env: Any, monitor: dict[str, Any] | None) -> dict[str, Any]:
    if monitor is None or not monitor.get("enabled", False):
        return {
            "enabled": False,
            "collision": False,
            "pairs": [],
            "robot_pairs": [],
            "held_object_pairs": [],
            "attached_object_proxy_pairs": [],
        }
    robot_pairs = contact_pairs_between(
        env,
        set(monitor.get("robot_geoms", [])),
        set(monitor.get("obstacle_geoms", [])),
    )
    held_obstacle_geoms = set(
        monitor.get("held_obstacle_geoms", monitor.get("obstacle_geoms", []))
    )
    held_pairs = contact_pairs_between(
        env,
        set(monitor.get("held_object_geoms", [])),
        held_obstacle_geoms,
    )
    proxy_pairs = attached_object_proxy_collisions(env, monitor)
    pairs = [*robot_pairs, *held_pairs, *proxy_pairs]
    return {
        "enabled": True,
        "collision": bool(pairs),
        "pairs": pairs,
        "robot_pairs": robot_pairs,
        "held_object_pairs": held_pairs,
        "attached_object_proxy_pairs": proxy_pairs,
    }


def preflight_arm_trajectory_collisions(
    env: Any,
    trajectory: list[list[float]],
    *,
    max_joint_step: float,
    monitor: dict[str, Any] | None,
) -> dict[str, Any]:
    if monitor is None or not monitor.get("enabled", False):
        return {"enabled": False, "collision": False}
    if not trajectory:
        return {"enabled": True, "collision": False, "checked_steps": 0}

    right = env.robots[0].composite_controller.part_controllers["right"]
    state = env.sim.get_state()
    current = np.array(env.sim.data.qpos[right.qpos_index], dtype=float)
    checked_steps = 0
    try:
        for waypoint_index, target in enumerate(trajectory):
            target_qpos = np.array(target, dtype=float)
            delta = target_qpos - current
            max_delta = float(np.max(np.abs(delta)))
            substeps = max(1, int(np.ceil(max_delta / max_joint_step)))
            for substep_index in range(1, substeps + 1):
                waypoint = current + delta * (substep_index / substeps)
                env.sim.data.qpos[right.qpos_index] = waypoint
                env.sim.data.qvel[:] = 0.0
                env.sim.forward()
                checked_steps += 1
                collision = check_monitor_collision(env, monitor)
                if collision["collision"]:
                    collision.update(
                        {
                            "checked_steps": int(checked_steps),
                            "waypoint_index": int(waypoint_index),
                            "substep_index": int(substep_index),
                        }
                    )
                    return collision
            current = target_qpos
        return {"enabled": True, "collision": False, "checked_steps": int(checked_steps)}
    finally:
        env.sim.set_state(state)
        env.sim.forward()


__all__ = [
    "attached_object_proxy_collisions",
    "allowed_fixture_contact_geoms",
    "base_path_monitor_summary",
    "check_base_path_collision",
    "check_monitor_collision",
    "collision_monitor_summary",
    "contact_pairs_between",
    "make_base_path_monitor",
    "make_collision_monitor",
    "object_contact_geoms",
    "preflight_arm_trajectory_collisions",
    "robot_arm_contact_geoms",
    "robot_base_contact_geoms",
    "static_fixture_contact_geoms",
]
