"""Native Isaac Lab assets for ExRoMa-authored benchmark tasks."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from exroma_bench.paths import asset_path


def _static_part(
    scene_cfg,
    name: str,
    *,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
) -> None:
    setattr(
        scene_cfg,
        name,
        AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.1,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.22, 0.27, 0.31),
                    metallic=0.75,
                    roughness=0.32,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        ),
    )


def add_test_tube_rack(
    scene_cfg,
    *,
    table_x: float,
    table_y: float,
    table_height: float,
    target_x: float,
    target_y: float,
) -> None:
    tube_radius = 0.018
    tube_height = 0.14
    scene_cfg.test_tube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/test_tube",
        spawn=sim_utils.CylinderCfg(
            radius=tube_radius,
            height=tube_height,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0015,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.04,
                angular_damping=0.04,
                max_depenetration_velocity=0.8,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=3.0,
                dynamic_friction=2.5,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.58, 0.82, 0.94),
                roughness=0.22,
                opacity=0.72,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(table_x, table_y, table_height + 0.075),
        ),
    )

    rack_height = 0.040
    wall = 0.006
    inner_half_width = 0.027
    slot_pitch = 0.065
    center_z = table_height + 0.5 * rack_height
    side_x = inner_half_width + 0.5 * wall
    rack_length = 3.0 * slot_pitch + wall
    for side_name, x_offset in (("left", -side_x), ("right", side_x)):
        _static_part(
            scene_cfg,
            f"test_tube_rack_side_{side_name}",
            pos=(target_x + x_offset, target_y, center_z),
            size=(wall, rack_length, rack_height),
        )
    for index, y_offset in enumerate(
        (-1.5 * slot_pitch, -0.5 * slot_pitch, 0.5 * slot_pitch, 1.5 * slot_pitch)
    ):
        _static_part(
            scene_cfg,
            f"test_tube_rack_cross_{index}",
            pos=(target_x, target_y + y_offset, center_z),
            size=(2.0 * side_x + wall, wall, rack_height),
        )
    scene_cfg.test_tube_target_slot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/test_tube_target_slot",
        spawn=sim_utils.CylinderCfg(
            radius=0.022,
            height=0.002,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.82, 0.30),
                roughness=0.48,
                opacity=0.72,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(target_x, target_y, table_height + 0.001),
        ),
    )


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _task_xy_to_world(
    task_x: float,
    task_y: float,
    *,
    table_x: float,
    table_y: float,
    mount_yaw_deg: float = -90.0,
) -> tuple[float, float]:
    task_yaw = math.radians(mount_yaw_deg - 90.0)
    return (
        table_x + math.cos(task_yaw) * task_x - math.sin(task_yaw) * task_y,
        table_y + math.sin(task_yaw) * task_x + math.cos(task_yaw) * task_y,
    )


def add_beat_block_hammer(
    scene_cfg,
    *,
    table_x: float,
    table_y: float,
    table_height: float,
) -> None:
    hammer_x, hammer_y = _task_xy_to_world(
        0.0, -0.06, table_x=table_x, table_y=table_y
    )
    half = math.radians(-180.0) * 0.5
    task_quat = (math.cos(half), 0.0, 0.0, math.sin(half))
    hammer_quat = _quat_multiply(task_quat, (0.0, 0.0, 0.994505, 0.104947))
    scene_cfg.robotwin_hammer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/robotwin_hammer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset_path("usd/robotwin/020_hammer/hammer.usd").as_posix(),
            scale=(0.079, 0.079, 0.079),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.04),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(hammer_x, hammer_y, table_height + 0.043),
            rot=hammer_quat,
        ),
    )
    block_x, block_y = _task_xy_to_world(
        -0.16, 0.05, table_x=table_x, table_y=table_y
    )
    block_half = 0.025
    scene_cfg.hammer_block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/hammer_block",
        spawn=sim_utils.CuboidCfg(
            size=(2.0 * block_half,) * 3,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.04, 0.03),
                roughness=0.55,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(block_x, block_y, table_height + block_half),
        ),
    )
