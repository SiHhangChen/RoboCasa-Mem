"""Navigation controller — station sampling + path planning + base movement.

``NavigationController`` combines a ``StationSampler`` with a ``BaseDriver``
and has no dependency on a task-specific orchestration context.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

from bench.execution.contracts import StageKind, StageResult
from bench.execution.navigation.contracts import Pose

if TYPE_CHECKING:
    from bench.execution.navigation.grid_planner import StationPathPlanner


# BaseDriver Protocol


class BaseDriver(Protocol):
    """Low-level base movement — plan a path and execute it.

    Implementations own env stepping. During migration, ``BaseDriverAdapter``
    still wraps the previous source execution helpers.
    """

    def plan_and_move(
        self,
        target_xy: np.ndarray,
        target_yaw: float | None,
        *,
        station_name: str = "",
        purpose: str | None = None,
        holding: str | None = None,
        waypoints: Sequence[Sequence[float]] | None = None,
        terminal_target_xy: Sequence[float] | None = None,
        planning_summary: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> StageResult:
        """Plan a collision-free path to *(target_xy, target_yaw)* and execute."""
        ...


class BasePoseProvider(Protocol):
    """Provide the current mobile-base xy position."""

    def current_base_xy(self) -> np.ndarray:
        ...


class StationSampler(Protocol):
    """Sample a base pose for a station name."""

    def sample(self, station_name: str, *, purpose: str | None = None) -> Pose:
        ...


# BaseDriverAdapter (wraps existing src execution code)


class BaseDriverAdapter:
    """Bridge the ``BaseDriver`` Protocol to existing execution functions.

    Uses bench-owned base motion helpers to execute in a RoboCasa-like env.
    """

    def __init__(
        self,
        env: Any,
        obs: dict[str, Any],
        *,
        nav_config: Any = None,
        post_step_callback: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    ):
        self._env = env
        self._obs = obs
        self._nav_config = nav_config
        self._post_step_callback = post_step_callback
        self._manual_base_path_executor = None

    def current_base_xy(self) -> np.ndarray:
        """Return the latest base xy from the driver's current observation."""
        self.refresh_obs()
        value = self._obs["robot0_base_pos"]
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            raise ValueError("obs['robot0_base_pos'] does not contain a finite xy position")
        return arr[:2]

    def current_base_yaw(self) -> float:
        """Return the latest base yaw from the driver's current observation."""
        from bench.execution.navigation.base_motion import base_yaw

        self.refresh_obs()
        return float(base_yaw(self._obs))

    def refresh_obs(self) -> dict[str, Any]:
        """Refresh the driver's cached observation from the environment."""
        self._obs = self._env._get_observations(force_update=True)
        return self._obs

    def _base_path_monitor(self, holding: str | None) -> dict[str, Any] | None:
        """Create the runtime base collision monitor requested by nav config."""
        if self._nav_config is None or not bool(
            self._nav_config.validate_base_path_collisions
        ):
            return None
        from bench.execution.planning.collision_monitor import make_base_path_monitor

        return make_base_path_monitor(
            self._env,
            enabled=True,
            fixture_keywords=self._nav_config.base_path_fixture_keywords,
            exclude_keywords=self._nav_config.base_path_exclude_keywords,
            holding_object=holding,
            abort_on_collision=bool(self._nav_config.abort_on_base_path_collision),
        )

    def _manual_path_executor(self):
        if self._nav_config is None or not bool(
            getattr(self._nav_config, "manual_base_path_enabled", False)
        ):
            return None
        path_file = getattr(self._nav_config, "manual_base_path_file", None)
        if path_file is None:
            return None
        if self._manual_base_path_executor is None:
            from bench.execution.navigation.manual_base_path import ManualBasePathExecutor
            from bench.execution.navigation.velocity_profile import BaseVelocityLimits

            self._manual_base_path_executor = ManualBasePathExecutor(
                self._env,
                path_file=path_file,
                start_index=int(getattr(self._nav_config, "manual_base_path_start_index", 1)),
                xy_tolerance=float(
                    getattr(self._nav_config, "manual_base_path_xy_tolerance", 0.07)
                ),
                yaw_tolerance=float(
                    getattr(self._nav_config, "manual_base_path_yaw_tolerance", 0.08)
                ),
                segment_yaw_threshold=float(
                    getattr(self._nav_config, "manual_base_path_segment_yaw_threshold", 0.25)
                ),
                max_rotate_steps=int(
                    getattr(self._nav_config, "manual_base_path_max_rotate_steps", 350)
                ),
                max_translate_steps=int(
                    getattr(self._nav_config, "manual_base_path_max_translate_steps", 900)
                ),
                xy_gain=float(getattr(self._nav_config, "manual_base_path_xy_gain", 3.0)),
                yaw_gain=float(getattr(self._nav_config, "manual_base_path_yaw_gain", 2.0)),
                nearest_yaw_weight=float(
                    getattr(self._nav_config, "manual_base_path_nearest_yaw_weight", 0.20)
                ),
                rotate_xy_tolerance=float(
                    getattr(self._nav_config, "manual_base_path_rotate_xy_tolerance", 0.10)
                ),
                velocity_limits=BaseVelocityLimits.from_navigation_config(self._nav_config),
            )
        return self._manual_base_path_executor

    def plan_and_move(
        self,
        target_xy: np.ndarray,
        target_yaw: float | None,
        *,
        station_name: str = "",
        purpose: str | None = None,
        holding: str | None = None,
        waypoints: Sequence[Sequence[float]] | None = None,
        terminal_target_xy: Sequence[float] | None = None,
        planning_summary: dict[str, Any] | None = None,
        **move_overrides: Any,
    ) -> StageResult:
        """Execute a base movement to the target pose."""
        use_manual_path = move_overrides.pop("use_manual_path", None)
        manual_executor = (
            self._manual_path_executor()
            if use_manual_path is not False
            else None
        )
        if manual_executor is not None:
            self.refresh_obs()
            gripper_action_override = move_overrides.get("gripper_action")
            xy_tolerance_override = move_overrides.get("xy_tolerance")
            yaw_tolerance_override = move_overrides.get("yaw_tolerance")
            terminal_tolerance_only = bool(
                move_overrides.get("terminal_tolerance_only", False)
            )
            original_xy_tolerance = manual_executor.xy_tolerance
            original_yaw_tolerance = manual_executor.yaw_tolerance
            if not terminal_tolerance_only:
                if xy_tolerance_override is not None:
                    manual_executor.xy_tolerance = float(xy_tolerance_override)
                if yaw_tolerance_override is not None:
                    manual_executor.yaw_tolerance = float(yaw_tolerance_override)
            try:
                self._obs, nav = manual_executor.move_to(
                    self._obs,
                    station_name,
                    purpose=purpose,
                    holding=holding,
                    gripper_action=(
                        None
                        if gripper_action_override is None
                        else float(gripper_action_override)
                    ),
                    force_terminal_settle=bool(
                        move_overrides.get("force_terminal_settle", False)
                    ),
                    terminal_xy_tolerance=(
                        float(xy_tolerance_override)
                        if terminal_tolerance_only
                        and xy_tolerance_override is not None
                        else None
                    ),
                    terminal_yaw_tolerance=(
                        float(yaw_tolerance_override)
                        if terminal_tolerance_only
                        and yaw_tolerance_override is not None
                        else None
                    ),
                )
            finally:
                manual_executor.xy_tolerance = original_xy_tolerance
                manual_executor.yaw_tolerance = original_yaw_tolerance
            nav["terminal_tolerance_only"] = terminal_tolerance_only
            if xy_tolerance_override is not None:
                nav["xy_tolerance_override"] = float(xy_tolerance_override)
            if yaw_tolerance_override is not None:
                nav["yaw_tolerance_override"] = float(yaw_tolerance_override)
            nav["requested_target_xy"] = np.array(target_xy, dtype=float)[:2].tolist()
            nav["requested_target_yaw"] = None if target_yaw is None else float(target_yaw)
            if planning_summary is not None:
                nav["station_path"] = planning_summary
            if station_name:
                nav["station_name"] = station_name
            return StageResult(
                index=-1,
                kind=StageKind.NAVIGATE,
                success=bool(nav.get("pass", False)),
                details=nav,
                failure_code=nav.get("reason") if not nav.get("pass") else None,
            )

        from bench.execution.navigation.base_motion import (
            move_base_along_waypoints,
            move_base_to_pose,
            qpos_for_controller,
        )
        from bench.execution.navigation.velocity_profile import BaseVelocityLimits

        velocity_limits = (
            BaseVelocityLimits.from_navigation_config(self._nav_config)
            if self._nav_config is not None
            else None
        )

        right = self._env.robots[0].composite_controller.part_controllers["right"]
        kwargs = {
            "tolerance": 0.06,
            "yaw_tolerance": 0.20,
            "max_steps": 500,
            "kinematic": False,
            "action_frame": "base",
            "holding_drop_tolerance": 0.04,
        }
        if self._nav_config is not None:
            kwargs.update(self._nav_config.base_move_kwargs())
        kwargs.update(
            {
                key: value
                for key, value in move_overrides.items()
                if key in kwargs and value is not None
            }
        )
        # Workflow stages use the same ``xy_tolerance`` metadata key for both
        # manual-path and direct navigation.  The direct base primitive calls
        # that argument ``tolerance``, so translate the public override here.
        # Without this mapping a stage that explicitly requests tight docking
        # silently keeps the configured default tolerance.
        xy_tolerance_override = move_overrides.get("xy_tolerance")
        if xy_tolerance_override is not None:
            kwargs["tolerance"] = float(xy_tolerance_override)
        yaw_tolerance_override = move_overrides.get("yaw_tolerance")
        gripper_action_override = move_overrides.get("gripper_action")
        resolved_gripper_action = (
            float(gripper_action_override)
            if gripper_action_override is not None
            else (1.0 if holding else -1.0)
        )

        arm_qpos = qpos_for_controller(self._env, right)
        path_monitor = self._base_path_monitor(holding)
        if waypoints is not None and len(waypoints) > 0:
            waypoint_kwargs = {}
            waypoint_tolerance = 0.10
            if self._nav_config is not None:
                waypoint_kwargs.update(self._nav_config.waypoint_follower_kwargs())
                waypoint_tolerance = float(self._nav_config.base_waypoint_tolerance)
            self._obs, nav = move_base_along_waypoints(
                self._env,
                self._obs,
                waypoints_xy=[np.array(point, dtype=float)[:2].tolist() for point in waypoints],
                terminal_target_xy=terminal_target_xy,
                target_yaw=target_yaw,
                gripper_action=resolved_gripper_action,
                arm_qpos=arm_qpos,
                tolerance=float(kwargs["tolerance"]),
                yaw_tolerance=float(kwargs["yaw_tolerance"]),
                max_steps=int(kwargs["max_steps"]),
                waypoint_tolerance=waypoint_tolerance,
                action_frame=str(kwargs["action_frame"]),
                holding_object_name=holding,
                holding_drop_tolerance=float(kwargs["holding_drop_tolerance"]),
                path_monitor=path_monitor,
                kinematic=bool(kwargs["kinematic"]),
                velocity_limits=velocity_limits,
                post_step_callback=self._post_step_callback,
                **waypoint_kwargs,
            )
        else:
            translation_action_sign = 1.0
            if self._nav_config is not None:
                translation_action_sign = float(
                    self._nav_config.base_translation_action_sign
                )
            self._obs, nav = move_base_to_pose(
                self._env,
                self._obs,
                target_xy=np.array(target_xy, dtype=float),
                target_yaw=target_yaw,
                gripper_action=resolved_gripper_action,
                arm_qpos=arm_qpos,
                translation_action_sign=translation_action_sign,
                path_monitor=path_monitor,
                holding_object_name=holding,
                velocity_limits=velocity_limits,
                post_step_callback=self._post_step_callback,
                **kwargs,
            )
        if (
            self._nav_config is not None
            and bool(getattr(self._nav_config, "base_snap_to_target_after_success", False))
            and bool(nav.get("pass", False))
        ):
            from bench.execution.navigation.base_motion import base_yaw, set_base_pose_kinematic

            before_xy = np.array(self._obs["robot0_base_pos"][:2], dtype=float)
            before_yaw = float(base_yaw(self._obs))
            self._obs = set_base_pose_kinematic(
                self._env,
                target_xy=np.array(target_xy, dtype=float),
                target_yaw=target_yaw,
            )
            after_xy = np.array(self._obs["robot0_base_pos"][:2], dtype=float)
            after_yaw = float(base_yaw(self._obs))
            nav["snap_to_target_after_success"] = {
                "enabled": True,
                "before_xy": before_xy.astype(float).tolist(),
                "before_yaw": before_yaw,
                "target_xy": np.array(target_xy, dtype=float)[:2].astype(float).tolist(),
                "target_yaw": None if target_yaw is None else float(target_yaw),
                "after_xy": after_xy.astype(float).tolist(),
                "after_yaw": after_yaw,
            }
        if planning_summary is not None:
            nav["station_path"] = planning_summary
        if xy_tolerance_override is not None:
            nav["xy_tolerance_override"] = float(xy_tolerance_override)
        if yaw_tolerance_override is not None:
            nav["yaw_tolerance_override"] = float(yaw_tolerance_override)
        if station_name:
            nav["station_name"] = station_name
        return StageResult(
            index=-1,
            kind=StageKind.NAVIGATE,
            success=bool(nav.get("pass", False)),
            details=nav,
            failure_code=nav.get("reason") if not nav.get("pass") else None,
        )


# New NavigationController (no ctx dependency)


class NavigationController:
    """Navigate to stations using sampling + base driver.

    Zero dependency on ``MobileManipulationContext``.
    """

    def __init__(
        self,
        station_sampler: StationSampler,
        base_driver: BaseDriver,
        *,
        path_planner: "StationPathPlanner | None" = None,
        base_pose_provider: BasePoseProvider | Callable[[], Sequence[float]] | None = None,
        path_include_provider: Callable[[str], Sequence[Sequence[float]]] | None = None,
    ):
        self._sampler = station_sampler
        self._driver = base_driver
        self._path_planner = path_planner
        self._base_pose_provider = base_pose_provider or (
            base_driver if hasattr(base_driver, "current_base_xy") else None
        )
        self._path_include_provider = path_include_provider

    @property
    def station_sampler(self) -> StationSampler:
        """Station sampler owned by this navigation assembly."""

        return self._sampler

    @property
    def base_driver(self) -> BaseDriver:
        """Low-level base driver owned by this navigation assembly."""

        return self._driver

    def sample_station_pose(
        self,
        station_name: str,
        *,
        purpose: str | None = None,
    ) -> Pose:
        """Resolve one task-owned station without moving the base."""

        return self._sampler.sample(station_name, purpose=purpose)

    def navigate_to(
        self,
        station_name: str,
        *,
        holding: str | None = None,
        purpose: str | None = None,
        **navigation_kwargs: Any,
    ) -> StageResult:
        """Sample a base pose for *station_name* and move there."""
        try:
            # import ipdb; ipdb.set_trace()
            pose = self._sampler.sample(station_name, purpose=purpose)
        except (KeyError, ValueError) as exc:
            return StageResult(
                index=-1,
                kind=StageKind.NAVIGATE,
                success=False,
                failure_code=f"sampling failed for {station_name!r}: {exc}",
            )

        target_xy = np.array(pose.position[:2], dtype=float)
        target_yaw = _pose_yaw(pose)
        station_summary = self._station_sample_summary(station_name)
        waypoints = None
        terminal_target_xy = None
        planning_summary = None

        if self._path_planner is not None:
            try:
                start_xy = self._current_base_xy()
                start_yaw = self._current_base_yaw()
                planning_summary = self._path_planner.plan(
                    start_xy=start_xy,
                    start_yaw=start_yaw,
                    target_xy=target_xy,
                    target_yaw=target_yaw,
                    station_name=station_name,
                    include_points=self._path_include_points(station_name),
                )
            except Exception as exc:
                return StageResult(
                    index=-1,
                    kind=StageKind.NAVIGATE,
                    success=False,
                    details={"station_pose": station_summary} if station_summary else {},
                    failure_code=f"path planning failed for {station_name!r}: {exc}",
                )
            if not bool(planning_summary.get("pass", False)):
                if station_summary is not None:
                    planning_summary["station_pose"] = station_summary
                return StageResult(
                    index=-1,
                    kind=StageKind.NAVIGATE,
                    success=False,
                    details={"station_path": planning_summary},
                    failure_code=planning_summary.get("reason", "station_path_planning_failed"),
                )
            waypoints = planning_summary.get("sparse_waypoints")
            terminal_target_xy = planning_summary.get("terminal_target_xy")

        if station_summary is not None:
            if planning_summary is None:
                planning_summary = {
                    "enabled": False,
                    "reason": "path_planner_not_configured",
                }
            planning_summary["station_pose"] = station_summary

        return self._driver.plan_and_move(
            target_xy,
            target_yaw,
            station_name=station_name,
            purpose=purpose,
            holding=holding,
            waypoints=waypoints,
            terminal_target_xy=terminal_target_xy,
            planning_summary=planning_summary,
            **navigation_kwargs,
        )

    def _current_base_xy(self) -> np.ndarray:
        provider = self._base_pose_provider
        if provider is None:
            raise RuntimeError("base_pose_provider is required when path_planner is configured")
        if callable(provider):
            value = provider()
        else:
            value = provider.current_base_xy()
        arr = np.array(value, dtype=float).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            raise ValueError("base_pose_provider returned an invalid xy position")
        return arr[:2]

    def _current_base_yaw(self) -> float | None:
        provider = self._base_pose_provider
        yaw_fn = getattr(provider, "current_base_yaw", None)
        if yaw_fn is None:
            return None
        yaw = float(yaw_fn())
        if not np.isfinite(yaw):
            raise ValueError("base_pose_provider returned an invalid yaw")
        return yaw

    def _path_include_points(self, station_name: str) -> Sequence[Sequence[float]]:
        if self._path_include_provider is not None:
            return self._path_include_provider(station_name)
        include_points = getattr(self._sampler, "include_points", None)
        if include_points is None:
            return ()
        return include_points()

    def _station_sample_summary(self, station_name: str) -> dict[str, Any] | None:
        summary_fn = getattr(self._sampler, "last_summary", None)
        if summary_fn is None:
            return None
        summary = summary_fn(station_name)
        return dict(summary) if isinstance(summary, dict) else None


def _dict_to_stage_result(raw: dict[str, Any], stage_kind: str) -> StageResult:
    return StageResult(
        index=-1,
        kind=StageKind(stage_kind),
        success=bool(raw.get("pass", False)),
        details=raw,
        failure_code=raw.get("reason") if not raw.get("pass") else None,
    )
def _pose_yaw(pose: Pose) -> float:
    if pose.quaternion is None:
        return 0.0
    x, y, z, w = np.array(pose.quaternion, dtype=float)
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )
