"""Isaac Lab configuration for the Pragyan rover base."""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

from exroma_bench.paths import asset_path


PRAGYAN_USD = "usd/robots/pragyan/pragyan.usdz"
PRAGYAN_WHEEL_RADIUS = 0.1125
PRAGYAN_TRACK_WIDTH = 0.945
PRAGYAN_LINEAR_SCALE = 0.35
PRAGYAN_ANGULAR_SCALE = 1.3962634015954636


def build_pragyan_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/robot",
    pos: tuple[float, float, float] = (0.0, 0.55, 0.25),
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset_path(PRAGYAN_USD).as_posix(),
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,
                rest_offset=0.005,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=1.5,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "drive_joints": ImplicitActuatorCfg(
                joint_names_expr=["wheel_drive_joint_.*"],
                velocity_limit_sim=40.0,
                effort_limit_sim=150.0,
                damping=5000.0,
                stiffness=0.0,
            ),
            "rocker_joints": ImplicitActuatorCfg(
                joint_names_expr=["rocker_joint_.*"],
                velocity_limit_sim=5.0,
                effort_limit_sim=2500.0,
                damping=400.0,
                stiffness=1000.0,
            ),
            "bogie_joints": ImplicitActuatorCfg(
                joint_names_expr=["boogie_joint_.*"],
                velocity_limit_sim=4.0,
                effort_limit_sim=500.0,
                damping=200.0,
                stiffness=250.0,
            ),
            "solar_panel_lock": ImplicitActuatorCfg(
                joint_names_expr=["solar_panel_joint"],
                velocity_limit_sim=0.1,
                effort_limit_sim=2500.0,
                damping=400.0,
                stiffness=1800.0,
            ),
        },
    )
