"""Compact one-drawer fixture owned by the TS02 package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import string_to_array

from robocasa.models.fixtures.fixture import Fixture
from robocasa.utils.object_utils import (
    get_fixture_to_point_rel_offset,
    get_pos_after_rel_offset,
)


SINGLE_DRAWER_ASSET = (
    Path(__file__).resolve().parent
    / "resources"
    / "assets"
    / "single_tabletop_drawer.xml"
)
SINGLE_DRAWER_DEFAULT_SIZE = (0.25, 0.30, 0.32)
SINGLE_DRAWER_FLOOR_CLEARANCE = 0.012


class SingleTabletopDrawer(Fixture):
    """A cabinet fixture with one independently sliding drawer."""

    def __init__(
        self,
        *,
        name: str,
        size: tuple[float, float, float] = SINGLE_DRAWER_DEFAULT_SIZE,
        pos: tuple[float, float, float] | None = None,
        rng: Any | None = None,
    ) -> None:
        super().__init__(
            xml=str(SINGLE_DRAWER_ASSET),
            name=name,
            duplicate_collision_geoms=True,
            joints=None,
            pos=pos,
            size=size,
            rng=rng,
        )
        self.size = np.asarray(size, dtype=float)

    def get_reset_region_names(self):
        return ("drawer",)

    def get_state(self):
        return {}

    def update_state(self, env: Any) -> None:
        region = self._regions.get("drawer")
        if region is None:
            return
        try:
            geom_pos = np.asarray(
                env.sim.data.get_geom_xpos(f"{self.naming_prefix}reg_drawer"),
                dtype=float,
            )
        except Exception:
            return
        halfsize = string_to_array(region["elem"].get("size"))
        self.update_regions(
            {
                "drawer": {
                    "pos": get_fixture_to_point_rel_offset(self, geom_pos),
                    "halfsize": halfsize,
                }
            },
            update_elem=False,
        )

    @property
    def rot(self) -> float:
        return float(self.euler[2])

    @property
    def door_joint_names(self) -> list[str]:
        return [f"{self.name}_drawer_slide"]

    @property
    def handle_name(self) -> str:
        return f"{self.name}_drawer_handle"

    def open_drawer(self, *, env: Any, min: float = 0.90, max: float = 1.0) -> None:
        self.set_joint_state(
            env=env,
            min=float(min),
            max=float(max),
            joint_names=self.door_joint_names,
        )

    def close_drawer(self, *, env: Any, min: float = 0.0, max: float = 0.0) -> None:
        self.set_joint_state(
            env=env,
            min=float(min),
            max=float(max),
            joint_names=self.door_joint_names,
        )

    def open_fraction(self, env: Any) -> float:
        state = self.get_joint_state(env, self.door_joint_names)
        return float(state.get(self.door_joint_names[0], 0.0))

    def set_world_pose(
        self,
        *,
        world_xyz: tuple[float, float, float],
        yaw: float,
    ) -> None:
        self.set_pos([float(value) for value in world_xyz])
        self.set_euler([0.0, 0.0, float(yaw)])

    def drawer_pose(self) -> np.ndarray:
        center = self.drawer_floor_pose()
        center[2] += 0.030
        return center[:3]

    def drawer_floor_pose(self) -> np.ndarray:
        points = self.get_int_sites(relative=False).get("drawer")
        if points is None:
            raise KeyError("single tabletop drawer has no drawer region")
        p0, px, py, pz = [np.asarray(point, dtype=float) for point in points[:4]]
        center = p0 + 0.5 * ((px - p0) + (py - p0) + (pz - p0))
        center[1] = (p0[1] + py[1]) * 0.5
        center[2] = p0[2] + SINGLE_DRAWER_FLOOR_CLEARANCE
        return center[:3]

    def front_pose(self, distance: float = 0.42) -> np.ndarray:
        pos = get_pos_after_rel_offset(
            self,
            np.asarray([0.0, -float(distance), 0.0], dtype=float),
        )
        quat_xyzw = T.mat2quat(T.euler2mat([0.0, 0.0, self.rot]))
        return np.asarray([pos[0], pos[1], pos[2], *quat_xyzw], dtype=float)


__all__ = ["SingleTabletopDrawer"]
