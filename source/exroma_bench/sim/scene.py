"""Isaac Lab-native scene construction without an SRB runtime dependency."""

from __future__ import annotations

import math
from types import SimpleNamespace

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from exroma_bench.paths import asset_path
from exroma_bench.robots.dual_piper_module_cfg import build_dual_piper_module_cfg
from exroma_bench.robots.mobile_aloha_camera_cfg import build_d435_camera_cfg
from exroma_bench.robots.pragyan_cfg import build_pragyan_cfg
from exroma_bench.tasks.benchmark_suite import add_benchmark_task_assets

from .domains import NASA_OUTPOST_SCALE, domain_for_scene
from .simforge_terrain import build_simforge_terrain_cfg
from .task_assets import add_beat_block_hammer, add_test_tube_rack


TABLE_X = 0.0
TABLE_Y = -0.85
TABLE_HEIGHT = 0.60
TABLE_LENGTH = 1.20
TABLE_WIDTH = 0.70
TABLE_THICKNESS = 0.06
TABLE_LEG_THICKNESS = 0.05
TABLE_LEG_HEIGHT = TABLE_HEIGHT - TABLE_THICKNESS
TABLE_LEG_Z = 0.5 * TABLE_LEG_HEIGHT
TABLE_LEG_X = 0.5 * TABLE_LENGTH - 0.5 * TABLE_LEG_THICKNESS
TABLE_LEG_Y = 0.5 * TABLE_WIDTH - 0.5 * TABLE_LEG_THICKNESS


def ensure_terrain_collisions(stage, terrain_path: str) -> tuple[int, int]:
    """Apply static triangle-mesh collision to baked terrain when absent."""

    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(terrain_path)
    if not root.IsValid():
        raise RuntimeError(f"Terrain prim does not exist: {terrain_path}")
    mesh_count = 0
    applied_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            applied_count += 1
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set("none")
    return mesh_count, applied_count


def _terrain_cfg(
    scene_name: str,
    *,
    size: float,
    seed: int,
) -> AssetBaseCfg:
    if scene_name == "ground_plane":
        spawn = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        )
        prim_path = "/World/GroundPlane"
    elif scene_name in {"procedural_moon", "procedural_mars"}:
        return build_simforge_terrain_cfg(
            scene_name=scene_name,
            prim_path="{ENV_REGEX_NS}/terrain",
            size=size,
            seed=seed,
        )
    else:
        paths = {
            "lunalab": "usd/scenery/lunalab.usdc",
            "oberpfaffenhofen": "usd/scenery/oberpfaffenhofen_test_site.usdc",
            "moon_surface": "usd/terrain/procedural_moon/moon_surface0.usdz",
            "mars_surface": "usd/terrain/procedural_mars/mars_surface0.usdz",
        }
        spawn = sim_utils.UsdFileCfg(
            usd_path=asset_path(paths[scene_name]).as_posix(),
        )
        prim_path = "{ENV_REGEX_NS}/terrain"
    return AssetBaseCfg(prim_path=prim_path, spawn=spawn)


def _table_part(name: str, pos, size, color) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.75,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _add_lunar_outpost(scene_cfg) -> None:
    """Add the collision-free NASA habitat as a distant lunar backdrop."""

    rotation = (0.991445, 0.0, 0.0, -0.130526)
    for index in (1, 2):
        setattr(
            scene_cfg,
            f"lunar_outpost_part_{index}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/lunar_outpost_part_{index}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=asset_path(
                        "usd/backgrounds/nasa_habitat_demonstration_unit/"
                        f"habitat_part_{index}.usdc"
                    ).as_posix(),
                    scale=(NASA_OUTPOST_SCALE,) * 3,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(7.0, -10.0, 0.0),
                    rot=rotation,
                ),
            ),
        )


def build_scene_cfg(
    *,
    scene_name: str,
    task_name: str,
    robot_start_y: float,
    num_envs: int = 1,
    spacing: float = 32.0,
    material_style: str = "lunar",
    enable_cameras: bool = False,
    terrain_seed: int = 0,
) -> InteractiveSceneCfg:
    domain = domain_for_scene(scene_name)
    mount_rot = (
        math.cos(math.radians(-90.0) * 0.5),
        0.0,
        0.0,
        math.sin(math.radians(-90.0) * 0.5),
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        terrain = _terrain_cfg(scene_name, size=spacing, seed=terrain_seed)
        sun = AssetBaseCfg(
            prim_path="/World/Sun",
            spawn=sim_utils.DistantLightCfg(
                intensity=domain.sun_intensity,
                angle=domain.sun_angle,
                color_temperature=domain.sun_color_temperature,
                enable_color_temperature=True,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                rot=domain.sun_orientation
            ),
        )
        sky = AssetBaseCfg(
            prim_path="/World/Sky",
            spawn=sim_utils.DomeLightCfg(
                intensity=domain.dome_intensity,
                texture_file=asset_path(domain.sky_texture).as_posix(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(rot=domain.sky_orientation),
        )
        rover = build_pragyan_cfg(pos=(0.0, robot_start_y, 0.25))
        dual_piper = build_dual_piper_module_cfg(
            pos=(0.0, robot_start_y, 0.67),
            rot=mount_rot,
        )
        work_table_top = _table_part(
            "work_table_top",
            (TABLE_X, TABLE_Y, TABLE_HEIGHT - 0.5 * TABLE_THICKNESS),
            (TABLE_LENGTH, TABLE_WIDTH, TABLE_THICKNESS),
            (0.46, 0.39, 0.31),
        )
        work_table_leg_fl = _table_part(
            "work_table_leg_fl",
            (TABLE_X + TABLE_LEG_X, TABLE_Y + TABLE_LEG_Y, TABLE_LEG_Z),
            (TABLE_LEG_THICKNESS, TABLE_LEG_THICKNESS, TABLE_LEG_HEIGHT),
            (0.28, 0.28, 0.27),
        )
        work_table_leg_fr = _table_part(
            "work_table_leg_fr",
            (TABLE_X + TABLE_LEG_X, TABLE_Y - TABLE_LEG_Y, TABLE_LEG_Z),
            (TABLE_LEG_THICKNESS, TABLE_LEG_THICKNESS, TABLE_LEG_HEIGHT),
            (0.28, 0.28, 0.27),
        )
        work_table_leg_rl = _table_part(
            "work_table_leg_rl",
            (TABLE_X - TABLE_LEG_X, TABLE_Y + TABLE_LEG_Y, TABLE_LEG_Z),
            (TABLE_LEG_THICKNESS, TABLE_LEG_THICKNESS, TABLE_LEG_HEIGHT),
            (0.28, 0.28, 0.27),
        )
        work_table_leg_rr = _table_part(
            "work_table_leg_rr",
            (TABLE_X - TABLE_LEG_X, TABLE_Y - TABLE_LEG_Y, TABLE_LEG_Z),
            (TABLE_LEG_THICKNESS, TABLE_LEG_THICKNESS, TABLE_LEG_HEIGHT),
            (0.28, 0.28, 0.27),
        )

    scene_cfg = SceneCfg(num_envs=num_envs, env_spacing=spacing, replicate_physics=False)
    if scene_name in {"moon_surface", "procedural_moon"}:
        _add_lunar_outpost(scene_cfg)
    if task_name in {"stack_blocks_two", "handover_block", "scan_object", "scan_rock"}:
        add_benchmark_task_assets(
            SimpleNamespace(scene=scene_cfg),
            task_name,
            table_x=TABLE_X,
            table_y=TABLE_Y,
            table_height=TABLE_HEIGHT,
            material_style=material_style,
        )
    elif task_name == "test_tube_rack":
        add_test_tube_rack(
            scene_cfg,
            table_x=TABLE_X,
            table_y=TABLE_Y,
            table_height=TABLE_HEIGHT,
            target_x=TABLE_X + 0.18,
            target_y=TABLE_Y,
        )
    elif task_name == "beat_block_hammer":
        add_beat_block_hammer(
            scene_cfg,
            table_x=TABLE_X,
            table_y=TABLE_Y,
            table_height=TABLE_HEIGHT,
        )
    else:
        raise ValueError(f"Unsupported ExRoMa task: {task_name}")
    if enable_cameras:
        scene_cfg.mast_camera = build_d435_camera_cfg(
            camera_name="mast",
            parent_link_name="camera_stand_2_Link",
            prim_path="/World/envs/env_0/dual_piper",
        )
        scene_cfg.front_left_camera = build_d435_camera_cfg(
            camera_name="front_left",
            parent_link_name="fl_link6",
            prim_path="/World/envs/env_0/dual_piper",
            wrist_mount_mode="piper",
        )
        scene_cfg.front_right_camera = build_d435_camera_cfg(
            camera_name="front_right",
            parent_link_name="fr_link6",
            prim_path="/World/envs/env_0/dual_piper",
            wrist_mount_mode="piper",
        )
    return scene_cfg
