# Terrain Pipeline

ExRoMa uses SimForge directly and does not import Space Robotics Bench. The
scene layer constructs a `simforge_foundry.MoonSurface` or `MarsSurface`, sets
its geometry-node inputs and PBR bake resolutions, and passes it to
`SimforgeAssetCfg` from SimForge's public Isaac Lab integration.

## Default 32-metre profile

The default profile matches the high-quality planetary workspace used for
collection:

- Geometry scale: `32 x 32 x 3.2 m`.
- Albedo and normal maps: `4096 px`.
- Roughness map: `2048 px`.
- Metallic map: `1024 px`.
- Emission map: `512 px`.
- Surface density: `0.16`.
- Central flat work region: approximately `4.22 m`.

The profile scales deterministically with `--terrain-size`. Generated models
are cached by SimForge, so an existing terrain seed is normally loaded from
`~/.cache/simforge` without launching Blender again.

## Scene selection

Use direct SimForge generation:

```bash
exroma preview \
  --task stack_blocks_two \
  --scene procedural_moon \
  --terrain-size 32 \
  --terrain-seed 0
```

Use the portable baked fallback:

```bash
exroma preview --task stack_blocks_two --scene moon_surface
```

`--terrain-seed` is independent from the task `--seed`. This keeps the terrain
fixed while object placement and language prompts vary across demonstrations.
Both values, the terrain size, and the generator source are stored in episode
metadata and `collection_summary.json`.

## Illumination

Moon, Mars, and Earth presets define their own gravity, solar irradiance,
angular diameter, color temperature, and environment map. The star, Mars-sky,
and cloudy-sky textures are distributed in ExRoMa-Assets and loaded through
the normal asset resolver; no reference repository path is used.

The lunar worksite composes the NASA outpost at its native converted scale and
snaps its combined lower bound to the generated terrain mesh. This placement
step is implemented locally in ExRoMa and does not require an SRB environment.
The default Moon lighting pose reproduces the deterministic seed-zero reset
used for the reference collection scene, including its rotated star skydome.
