"""Direct SimForge terrain construction without an SRB runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SimforgeTerrainProfile:
    size: float
    vertical_scale: float
    resolution_multiplier: int
    density: float
    flat_area_size: float

    @property
    def scale(self) -> tuple[float, float, float]:
        return (self.size, self.size, self.vertical_scale)


def terrain_profile(size: float = 32.0) -> SimforgeTerrainProfile:
    """Return the size-adaptive profile used by released ExRoMa terrains."""

    if size <= 0.0:
        raise ValueError("SimForge terrain size must be positive.")
    dynamic_resolution = max(
        1,
        min(
            10,
            2 * round(min(8.0, math.pow(size, 0.4)) / 2),
        ),
    )
    return SimforgeTerrainProfile(
        size=float(size),
        vertical_scale=0.1 * float(size),
        resolution_multiplier=dynamic_resolution,
        density=0.01 * dynamic_resolution**2,
        flat_area_size=0.8 * dynamic_resolution**1.2,
    )


def build_simforge_terrain_cfg(
    *,
    scene_name: str,
    prim_path: str,
    size: float = 32.0,
    seed: int = 0,
):
    """Build an Isaac Lab asset config backed directly by SimForge Foundry."""

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from simforge.integrations.isaaclab.spawner import SimforgeAssetCfg

    model = build_simforge_surface(scene_name=scene_name, size=size)

    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=SimforgeAssetCfg(
            assets=[model],
            seed=int(seed),
            use_cache=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )


def build_simforge_surface(*, scene_name: str, size: float = 32.0):
    """Construct and configure a Foundry surface without importing Isaac Lab."""

    from simforge import BakeType
    import simforge_foundry

    model_types = {
        "procedural_moon": simforge_foundry.MoonSurface,
        "procedural_mars": simforge_foundry.MarsSurface,
    }
    try:
        model = model_types[scene_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported SimForge terrain scene: {scene_name}") from exc

    profile = terrain_profile(size)
    surface_operation = model.geo.ops[0]
    surface_operation.scale = profile.scale
    surface_operation.density = profile.density
    surface_operation.flat_area_size = profile.flat_area_size
    model.texture_resolution = {
        BakeType.ALBEDO: profile.resolution_multiplier * 1024,
        BakeType.EMISSION: profile.resolution_multiplier * 128,
        BakeType.METALLIC: profile.resolution_multiplier * 256,
        BakeType.NORMAL: profile.resolution_multiplier * 1024,
        BakeType.ROUGHNESS: profile.resolution_multiplier * 512,
    }

    return model
