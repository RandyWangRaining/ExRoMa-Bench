from __future__ import annotations

import pytest

from simforge import BakeType

from exroma_bench.sim.simforge_terrain import build_simforge_surface, terrain_profile


def test_default_profile_matches_high_quality_32_metre_surface() -> None:
    profile = terrain_profile(32.0)
    assert profile.scale == (32.0, 32.0, 3.2)
    assert profile.resolution_multiplier == 4
    assert profile.density == pytest.approx(0.16)
    assert profile.flat_area_size == pytest.approx(4.222425, rel=1e-5)


def test_profile_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError):
        terrain_profile(0.0)


@pytest.mark.parametrize(
    ("scene_name", "model_name"),
    (("procedural_moon", "moon_surface"), ("procedural_mars", "mars_surface")),
)
def test_foundry_surface_receives_high_quality_profile(
    scene_name: str,
    model_name: str,
) -> None:
    model = build_simforge_surface(scene_name=scene_name, size=32.0)
    operation = model.geo.ops[0]

    assert model.name() == model_name
    assert operation.scale == (32.0, 32.0, 3.2)
    assert operation.density == pytest.approx(0.16)
    assert operation.flat_area_size == pytest.approx(4.222425, rel=1e-5)
    assert model.texture_resolution[BakeType.ALBEDO] == 4096
    assert model.texture_resolution[BakeType.NORMAL] == 4096
    assert model.texture_resolution[BakeType.ROUGHNESS] == 2048
