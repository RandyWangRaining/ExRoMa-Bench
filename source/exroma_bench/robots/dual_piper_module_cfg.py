"""Isaac Lab articulation configuration for the dual PiPER rover module."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from exroma_bench.paths import asset_path


DUAL_PIPER_MODULE_USD = "usd/robots/dual_piper/dual_piper_module.usd"
DUAL_PIPER_MODULE_URDF = "urdf/robots/dual_piper/dual_piper_module.urdf"

DEFAULT_ARM_POSE = {
    "fl_joint1": 0.10,
    "fl_joint2": 0.929,
    "fl_joint3": -1.267,
    "fl_joint4": 0.0,
    "fl_joint5": 0.950,
    "fl_joint6": 0.050,
    "fl_joint7": 0.024,
    "fl_joint8": -0.024,
    "fr_joint1": 0.10,
    "fr_joint2": 0.929,
    "fr_joint3": -1.267,
    "fr_joint4": 0.0,
    "fr_joint5": 0.950,
    "fr_joint6": 0.050,
    "fr_joint7": 0.024,
    "fr_joint8": -0.024,
    "camera_stand_1_joint": 0.347,
    "camera_stand_2_joint": 0.494,
}


def _build_spawn_cfg(*, fix_root_link: bool = False):
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=2.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        fix_root_link=fix_root_link,
        enabled_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
    )
    if os.environ.get("EXROMA_DUAL_PIPER_SOURCE", "usd").lower() == "urdf":
        return sim_utils.UrdfFileCfg(
            asset_path=asset_path(DUAL_PIPER_MODULE_URDF).as_posix(),
            usd_dir=os.environ.get("EXROMA_CONVERTED_ASSET_ROOT"),
            make_instanceable=False,
            fix_base=fix_root_link,
            root_link_name="dual_piper_mount",
            merge_fixed_joints=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
            activate_contact_sensors=True,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    return sim_utils.UsdFileCfg(
        usd_path=asset_path(DUAL_PIPER_MODULE_USD).as_posix(),
        activate_contact_sensors=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
    )


def build_dual_piper_module_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/dual_piper",
    pos: tuple[float, float, float] = (0.0, 0.0, 0.8),
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    fix_root_link: bool = False,
) -> ArticulationCfg:
    """Return the independently controlled dual-arm payload configuration."""

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=_build_spawn_cfg(fix_root_link=fix_root_link),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos=DEFAULT_ARM_POSE,
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["[f][lr]_joint[1-6]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=3.0,
                stiffness=120.0,
                damping=12.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=["[f][lr]_joint[7-8]"],
                effort_limit_sim=15.0,
                velocity_limit_sim=1.0,
                stiffness=300.0,
                damping=30.0,
            ),
            "mast": ImplicitActuatorCfg(
                joint_names_expr=["camera_stand_.*_joint"],
                effort_limit_sim=20.0,
                velocity_limit_sim=1.0,
                stiffness=80.0,
                damping=8.0,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )
