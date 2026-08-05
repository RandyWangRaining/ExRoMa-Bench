"""RoboTwin assets and task geometry for the dual-PiPER benchmark suite."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.utils import clone
from pxr import Gf, Sdf, UsdGeom, UsdShade

from exroma_bench.paths import asset_root

from .switch_variants import load_switch_variant


ROBOTTWIN_USD_ROOT = asset_root() / "usd" / "robotwin"
ROBOTTWIN_URDF_ROOT = asset_root() / "urdf" / "robotwin"
LUNAR_TEXTURE_ROOT = asset_root() / "textures" / "robotwin_lunar"
LUNAR_TEXTURES = (
    LUNAR_TEXTURE_ROOT / "Sand_BaseColor.png",
    LUNAR_TEXTURE_ROOT / "Sand_Normal.png",
    LUNAR_TEXTURE_ROOT / "Sand_ORM.png",
)
OMNILRS_SCAN_ROCK_USD = asset_root() / "usd" / "rocks" / "omnilrs" / "rock4057.usdz"
OMNILRS_SCAN_ROCK_SCALE = 0.08
OMNILRS_SCAN_ROCK_SIZE = (0.083, 0.079, 0.080)
OMNILRS_SCAN_ROCK_ROOT_Z_OFFSET = 0.042

COLORS = {
    "red": (0.82, 0.06, 0.04),
    "green": (0.06, 0.62, 0.16),
    "blue": (0.05, 0.18, 0.82),
    "yellow": (0.92, 0.68, 0.06),
}

ROBOTTWIN_RIGID_SCALES = {
    "011_dustbin": 0.20,
    "024_scanner": 0.08,
    "047_mouse": 0.50,
    "048_stapler": 0.05,
    "057_toycar": 0.05,
    "063_tabletrashbin": 0.08,
    "073_rubikscube": 0.04,
    "075_bread": 0.04,
    "077_phone": 0.078,
    "081_playingcards": 0.05,
    "107_soap": 0.05,
    "112_tea-box": 0.05,
    "113_coffee-box": 0.05,
}

CABINET_OBJECT_CHOICES = tuple(
    name
    for name in ROBOTTWIN_RIGID_SCALES
    if name not in {"011_dustbin", "063_tabletrashbin"}
)


def _lunar_material(style: str):
    if style == "original":
        return None
    if style != "lunar":
        raise ValueError(f"Unsupported RoboTwin material style: {style}")
    missing = [path for path in LUNAR_TEXTURES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing lunar material textures: {missing}. "
            "Restore the ExRoMa asset bundle first."
        )
    # The textured UsdPreviewSurface is bound after all referenced USD/URDF
    # meshes exist on the live stage.
    return None


def _create_lunar_preview_material(stage, material_path: str):
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st_reader = UsdShade.Shader.Define(stage, f"{material_path}/Primvar")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_output = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    albedo = UsdShade.Shader.Define(stage, f"{material_path}/Albedo")
    albedo.CreateIdAttr("UsdUVTexture")
    albedo.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(LUNAR_TEXTURES[0].as_posix())
    )
    albedo.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    albedo.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    albedo.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    albedo.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
    albedo_output = albedo.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        albedo_output
    )

    normal = UsdShade.Shader.Define(stage, f"{material_path}/Normal")
    normal.CreateIdAttr("UsdUVTexture")
    normal.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(LUNAR_TEXTURES[1].as_posix())
    )
    normal.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    normal.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    normal.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    normal.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
    normal.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(2.0, 2.0, 2.0, 1.0)
    )
    normal.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(-1.0, -1.0, -1.0, 0.0)
    )
    normal_output = normal.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
        normal_output
    )

    orm = UsdShade.Shader.Define(stage, f"{material_path}/Orm")
    orm.CreateIdAttr("UsdUVTexture")
    orm.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(LUNAR_TEXTURES[2].as_posix())
    )
    orm.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    orm.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    orm.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    orm.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_output)
    roughness_output = orm.CreateOutput("g", Sdf.ValueTypeNames.Float)
    surface.GetInput("roughness").ConnectToSource(roughness_output)

    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    return material


def apply_benchmark_materials(
    stage,
    task_name: str,
    material_style: str,
    *,
    env_prim_path: str = "/World/envs/env_0",
) -> int:
    """Bind the lunar PBR map set to the RoboTwin table and visible task meshes."""

    _lunar_material(material_style)
    if material_style == "original":
        return 0
    roots_by_task = {
        "dump_bin_bigbin": ("benchmark_big_bin", "benchmark_small_bin"),
        "put_object_cabinet": (
            "benchmark_cabinet",
            "benchmark_cabinet_object",
        ),
        "turn_switch": ("benchmark_switch",),
        "scan_object": (
            "benchmark_scanner",
            "benchmark_scan_object",
        ),
        "scan_rock": ("benchmark_scanner",),
    }
    roots = ("work_table", *roots_by_task.get(task_name, ()))

    material = _create_lunar_preview_material(
        stage, f"{env_prim_path}/Looks/RobotwinLunarDust"
    )
    bound = 0
    for root_name in roots:
        root = stage.GetPrimAtPath(f"{env_prim_path}/{root_name}")
        if not root.IsValid():
            continue
        # URDF meshes are instance proxies. A strong root binding intentionally
        # overrides their original descendant bindings without editing instances.
        UsdShade.MaterialBindingAPI.Apply(root).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        bound += 1
    return bound


def _robotwin_usd(object_name: str) -> Path:
    path = ROBOTTWIN_USD_ROOT / object_name / "base0.usdc"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing converted RoboTwin asset: {path}. Run "
            "scripts/tools/download_robotwin_task_assets.py and convert the GLB pair."
        )
    return path


def _dynamic_box(
    name: str,
    *,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    mass: float,
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.8,
                dynamic_friction=1.4,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.48,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _robotwin_procedural_block(
    name: str,
    *,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    """Instantiate RoboTwin's original ProceduralBlock asset definition."""

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _target_marker(
    name: str,
    *,
    pos: tuple[float, float, float],
    radius: float,
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=0.003,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.5,
                opacity=0.65,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _rigid_robotwin_asset(
    name: str,
    object_name: str,
    *,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    mass: float,
    material_style: str,
) -> RigidObjectCfg:
    scale = ROBOTTWIN_RIGID_SCALES[object_name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_robotwin_usd(object_name).as_posix(),
            scale=(scale, scale, scale),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.20,
                angular_damping=0.50,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=_lunar_material(material_style),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _static_robotwin_asset(
    name: str,
    object_name: str,
    *,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    material_style: str,
) -> RigidObjectCfg:
    scale = ROBOTTWIN_RIGID_SCALES[object_name]
    usd_path = _robotwin_usd(object_name).as_posix()
    spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        scale=(scale, scale, scale),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        visual_material=_lunar_material(material_style),
    )
    asset_cfg = RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )
    return asset_cfg


@clone
def _spawn_invisible_collision_cuboid(
    prim_path: str,
    cfg: sim_utils.CuboidCfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    prim = sim_utils.spawn_cuboid(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    proxy_mesh = prim.GetStage().GetPrimAtPath(f"{prim_path}/geometry/mesh")
    if proxy_mesh.IsValid():
        UsdGeom.Imageable(proxy_mesh).MakeInvisible()
    return prim


def _static_omnilrs_ground_rock(
    name: str,
    *,
    pos: tuple[float, float, float],
) -> RigidObjectCfg:
    if not OMNILRS_SCAN_ROCK_USD.is_file():
        raise FileNotFoundError(
            f"Missing OmniLRS ground-rock asset: {OMNILRS_SCAN_ROCK_USD}. "
            "Run `exroma assets pull` to restore it."
        )
    spawn_cfg = sim_utils.CuboidCfg(
            size=OMNILRS_SCAN_ROCK_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        )
    spawn_cfg.func = _spawn_invisible_collision_cuboid
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _omnilrs_ground_rock_visual(
    name: str,
    *,
    parent_name: str,
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{parent_name}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=OMNILRS_SCAN_ROCK_USD.as_posix(),
            scale=(OMNILRS_SCAN_ROCK_SCALE,) * 3,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


def _articulated_robotwin_asset(
    name: str,
    *,
    urdf_path: Path,
    usd_dir: Path,
    scale: float,
    pos: tuple[float, float, float],
    rot: tuple[float, float, float, float],
    joint_names: str,
    joint_pos: dict[str, float],
    material_style: str,
    actuator_effort: float = 80.0,
    actuator_stiffness: float = 40.0,
    actuator_damping: float = 8.0,
) -> ArticulationCfg:
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"Missing RoboTwin articulation source: {urdf_path}. Run "
            "scripts/tools/download_robotwin_task_assets.py."
        )
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path.as_posix(),
            usd_dir=usd_dir.as_posix(),
            usd_file_name=f"{name}.usd",
            fix_base=True,
            merge_fixed_joints=True,
            joint_drive=None,
            link_density=500.0,
            make_instanceable=False,
            force_usd_conversion=False,
            scale=(scale, scale, scale),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            visual_material=_lunar_material(material_style),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "mechanism": ImplicitActuatorCfg(
                joint_names_expr=[joint_names],
                effort_limit_sim=actuator_effort,
                velocity_limit_sim=1.0,
                stiffness=actuator_stiffness,
                damping=actuator_damping,
            )
        },
        soft_joint_pos_limit_factor=0.98,
    )


def _set(scene, name: str, cfg) -> None:
    setattr(scene, name, cfg)


def _add_stack_two(
    scene,
    table_x: float,
    table_y: float,
    table_z: float,
    _material_style: str,
    _cabinet_object: str,
) -> None:
    # RoboTwin 1.0 blocks_stack_easy uses 5 cm, 10 g procedural cubes.
    block_size = 0.05
    for name, x, y, color in (
        ("benchmark_block_red", 0.21, 0.07, COLORS["red"]),
        ("benchmark_block_black", 0.21, 0.16, (0.01, 0.01, 0.01)),
    ):
        _set(
            scene,
            name,
            _dynamic_box(
                name,
                pos=(table_x + x, table_y + y, table_z + 0.5 * block_size),
                size=(block_size,) * 3,
                color=color,
                mass=0.01,
            ),
        )


def _add_handover(
    scene,
    table_x: float,
    table_y: float,
    table_z: float,
    _material_style: str,
    _cabinet_object: str,
) -> None:
    # RoboTwin handover_block uses two runtime-generated ProceduralBlock assets;
    # there is no external GLB/USD mesh for this task.
    _set(
        scene,
        "benchmark_handover_block",
        _robotwin_procedural_block(
            "benchmark_handover_block",
            pos=(table_x - 0.18, table_y + 0.08, table_z + 0.10),
            size=(0.05, 0.05, 0.20),
            color=(1.0, 0.0, 0.0),
        ),
    )
    _set(
        scene,
        "benchmark_handover_target",
        _robotwin_procedural_block(
            "benchmark_handover_target",
            pos=(table_x - 0.20, table_y + 0.04, table_z + 0.005),
            size=(0.10, 0.10, 0.01),
            color=(0.0, 0.0, 1.0),
        ),
    )


def _add_scan_object(
    scene,
    table_x: float,
    table_y: float,
    table_z: float,
    material_style: str,
    _cabinet_object: str,
) -> None:
    """Add RoboTwin's base0 scanner and tea-box assets."""

    _set(
        scene,
        "benchmark_scanner",
        _rigid_robotwin_asset(
            "benchmark_scanner",
            "024_scanner",
            pos=(table_x + 0.18, table_y + 0.08, table_z + 0.003),
            rot=(1.0, 0.0, 0.0, 0.0),
            mass=0.12,
            material_style=material_style,
        ),
    )
    _set(
        scene,
        "benchmark_scan_object",
        _rigid_robotwin_asset(
            "benchmark_scan_object",
            "112_tea-box",
            pos=(table_x - 0.18, table_y + 0.08, table_z + 0.003),
            rot=(1.0, 0.0, 0.0, 0.0),
            mass=0.06,
            material_style=material_style,
        ),
    )


def _add_scan_rock(
    scene,
    table_x: float,
    table_y: float,
    table_z: float,
    material_style: str,
    _cabinet_object: str,
) -> None:
    """Add the annotated RoboTwin scanner and a scaled OmniLRS ground rock."""

    _set(
        scene,
        "benchmark_scanner",
        _rigid_robotwin_asset(
            "benchmark_scanner",
            "024_scanner",
            pos=(table_x + 0.18, table_y + 0.08, table_z + 0.003),
            rot=(1.0, 0.0, 0.0, 0.0),
            mass=0.12,
            material_style=material_style,
        ),
    )
    _set(
        scene,
        "benchmark_scan_rock",
        _static_omnilrs_ground_rock(
            "benchmark_scan_rock",
            pos=(
                table_x - 0.03,
                table_y,
                table_z + OMNILRS_SCAN_ROCK_ROOT_Z_OFFSET,
            ),
        ),
    )
    _set(
        scene,
        "benchmark_scan_rock_visual",
        _omnilrs_ground_rock_visual(
            "visual",
            parent_name="benchmark_scan_rock",
        ),
    )


def _add_dump(
    scene,
    table_x: float,
    table_y: float,
    table_z: float,
    material_style: str,
    _cabinet_object: str,
) -> None:
    _set(
        scene,
        "benchmark_big_bin",
        _static_robotwin_asset(
            "benchmark_big_bin",
            "011_dustbin",
            pos=(table_x - 0.62, table_y + 0.02, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            material_style=material_style,
        ),
    )
    _set(
        scene,
        "benchmark_small_bin",
        _rigid_robotwin_asset(
            "benchmark_small_bin",
            "063_tabletrashbin",
            pos=(table_x + 0.18, table_y - 0.08, table_z + 0.003),
            rot=(1.0, 0.0, 0.0, 0.0),
            mass=0.10,
            material_style=material_style,
        ),
    )
    for index in range(5):
        name = f"benchmark_garbage_{index}"
        _set(
            scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name}",
                spawn=sim_utils.SphereCfg(
                    radius=0.008,
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=0.001
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=0.5,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.0001),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=COLORS["red"],
                        roughness=0.5,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(
                        table_x + 0.18 + 0.010 * ((index % 2) - 0.5),
                        table_y - 0.08 + 0.010 * ((index // 2) - 1.0),
                        table_z + 0.09 + 0.010 * index,
                    )
                ),
            ),
        )


def _add_cabinet(
    scene,
    table_x: float,
    table_y: float,
    table_z: float,
    material_style: str,
    cabinet_object: str,
) -> None:
    if cabinet_object not in CABINET_OBJECT_CHOICES:
        choices = ", ".join(CABINET_OBJECT_CHOICES)
        raise ValueError(
            f"Unsupported cabinet object '{cabinet_object}'. Available: {choices}"
        )
    cabinet_source = ROBOTTWIN_URDF_ROOT / "036_cabinet" / "46653"
    _set(
        scene,
        "benchmark_cabinet",
        _articulated_robotwin_asset(
            "benchmark_cabinet",
            urdf_path=cabinet_source / "mobility.urdf",
            usd_dir=ROBOTTWIN_USD_ROOT / "036_cabinet" / "46653",
            scale=0.27,
            pos=(table_x, table_y + 0.30, table_z + 0.21688),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
            joint_names="joint_[1-3]",
            joint_pos={"joint_[1-3]": 0.0},
            material_style=material_style,
        ),
    )
    _set(
        scene,
        "benchmark_cabinet_object",
        _rigid_robotwin_asset(
            "benchmark_cabinet_object",
            cabinet_object,
            pos=(table_x + 0.22, table_y + 0.20, table_z + 0.003),
            rot=(1.0, 0.0, 0.0, 0.0),
            mass=0.01,
            material_style=material_style,
        ),
    )


def _add_switch(
    scene,
    table_x: float,
    table_y: float,
    table_z: float,
    material_style: str,
    _cabinet_object: str,
    switch_variant: str,
) -> None:
    variant = load_switch_variant(switch_variant)
    switch_source = ROBOTTWIN_URDF_ROOT / "056_switch" / variant.object_id
    _set(
        scene,
        "benchmark_switch",
        _articulated_robotwin_asset(
            "benchmark_switch",
            urdf_path=switch_source / "mobility.urdf",
            usd_dir=ROBOTTWIN_USD_ROOT / "056_switch" / variant.object_id,
            scale=variant.scale,
            pos=(table_x + 0.18, table_y + 0.35, table_z + 0.075),
            # RoboTwin faces the switch toward -Y. Our rover approaches the
            # table along world -Y, so rotate the original root pose by 180
            # degrees to keep the switch front facing the robot.
            rot=(0.704141, 0.0, 0.0, -0.71006),
            joint_names="joint_0",
            joint_pos={"joint_0": 0.0},
            material_style=material_style,
            actuator_effort=5.0,
            actuator_stiffness=0.0,
            actuator_damping=0.05,
        ),
    )


def add_benchmark_task_assets(
    env_cfg,
    task_name: str,
    *,
    table_x: float,
    table_y: float,
    table_height: float,
    material_style: str = "lunar",
    cabinet_object: str = "073_rubikscube",
    switch_variant: str = "100880",
) -> None:
    builders = {
        "stack_blocks_two": _add_stack_two,
        "handover_block": _add_handover,
        "scan_object": _add_scan_object,
        "scan_rock": _add_scan_rock,
        "dump_bin_bigbin": _add_dump,
        "put_object_cabinet": _add_cabinet,
        "turn_switch": _add_switch,
    }
    if task_name not in builders:
        return
    print(
        f"[INFO]: Configuring RoboTwin task assets: task={task_name}, "
        f"material={material_style}",
        flush=True,
    )
    builder_args = (
        env_cfg.scene,
        table_x,
        table_y,
        table_height,
        material_style,
        cabinet_object,
    )
    if task_name == "turn_switch":
        builders[task_name](*builder_args, switch_variant)
    else:
        builders[task_name](*builder_args)
    print(f"[INFO]: RoboTwin task assets configured: {task_name}", flush=True)


RECORD_OBJECT_NAMES: Mapping[str, tuple[str, ...]] = {
    "stack_blocks_two": (
        "benchmark_block_red",
        "benchmark_block_black",
    ),
    "handover_block": ("benchmark_handover_block",),
    "scan_object": (
        "benchmark_scanner",
        "benchmark_scan_object",
    ),
    "scan_rock": (
        "benchmark_scanner",
        "benchmark_scan_rock",
    ),
    "dump_bin_bigbin": (
        "benchmark_small_bin",
        "benchmark_garbage_0",
        "benchmark_garbage_1",
        "benchmark_garbage_2",
        "benchmark_garbage_3",
        "benchmark_garbage_4",
    ),
    "put_object_cabinet": (
        "benchmark_cabinet",
        "benchmark_cabinet_object",
    ),
    "turn_switch": ("benchmark_switch",),
}


def benchmark_record_objects(scene, task_name: str) -> dict[str, object]:
    return {name: scene[name] for name in RECORD_OBJECT_NAMES.get(task_name, ())}
