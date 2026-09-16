from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bench.tasks.ts03.materials import (
    SAUCEPAN_MATERIAL_PROFILES,
    apply_object_material,
)


def test_saucepan_material_skips_liquid_and_region_geoms() -> None:
    model = SimpleNamespace(
        body_parentid=np.asarray([0, 0]),
        geom_bodyid=np.asarray([1, 1, 1, 1]),
        geom_group=np.asarray([1, 1, 1, 0]),
        geom_type=np.asarray([7, 5, 5, 7]),
        geom_rgba=np.asarray(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.58, 0.37, 0.24, 0.92],
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.5],
            ],
            dtype=float,
        ),
        geom_matid=np.asarray([0, -1, -1, -1]),
        mat_specular=np.asarray([0.1], dtype=float),
        mat_shininess=np.asarray([0.1], dtype=float),
        mat_reflectance=np.asarray([0.0], dtype=float),
        geom_id2name=lambda geom_id: (
            "saucepan_vis",
            "saucepan_liquid",
            "saucepan_reg_int",
            "saucepan_collision",
        )[int(geom_id)],
    )
    env = SimpleNamespace(obj_body_id={"saucepan": 1}, sim=SimpleNamespace(model=model))
    liquid_before = model.geom_rgba[1].copy()
    region_before = model.geom_rgba[2].copy()
    collision_before = model.geom_rgba[3].copy()

    result = apply_object_material(
        env, "saucepan", "copper", SAUCEPAN_MATERIAL_PROFILES
    )

    assert result["applied"] is True
    assert result["visual_geom_count"] == 1
    assert np.allclose(model.geom_rgba[0], SAUCEPAN_MATERIAL_PROFILES["copper"]["rgba"])
    assert np.array_equal(model.geom_rgba[1], liquid_before)
    assert np.array_equal(model.geom_rgba[2], region_before)
    assert np.array_equal(model.geom_rgba[3], collision_before)
    assert model.mat_specular[0] == SAUCEPAN_MATERIAL_PROFILES["copper"]["specular"]
