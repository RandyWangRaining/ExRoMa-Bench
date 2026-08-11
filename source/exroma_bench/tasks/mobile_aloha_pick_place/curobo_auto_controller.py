"""Automatic randomized Mobile ALOHA pick-and-place using cuRobo."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from exroma_bench.sim.isaaclab_compat import get_root_pose_w
import isaaclab.utils.math as math_utils

from .curobo_planner import CuroboTrajectory, MobileAlohaCuroboPlanner


def _quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors without triggering Isaac Lab's first-call TorchScript compilation."""

    xyz = quaternion[..., 1:]
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * t + torch.cross(xyz, t, dim=-1)


@dataclass
class CuroboAutoPickPlaceConfig:
    arm: str = "fl"
    table_x: float = 1.0
    table_y: float = 0.0
    table_height: float = 0.6
    table_length: float = 0.8
    table_width: float = 0.55
    target_x: float = 1.0
    target_y: float = -0.18
    target_functional_aim_point_w: tuple[float, float, float] | None = None
    target_functional_offset_object: tuple[float, float, float] | None = None
    target_functional_axis_object: tuple[float, float, float] | None = None
    target_functional_standoff: float = 0.0
    success_target_x: float | None = None
    success_target_y: float | None = None
    target_radius: float = 0.08
    target_center_z: float | None = None
    target_z_tolerance: float = 0.06
    target_max_linear_speed: float = 0.08
    robot_config_stem: str = "mobile_aloha"
    planner_position_threshold: float = 0.005
    planner_rotation_threshold: float = 0.05
    planner_enable_finetune_trajopt: bool = True
    planner_self_collision_check: bool = False
    planner_enabled: bool = True
    planner_table_z_offset: float = 0.0
    align_base: bool = True
    spawn_x_range: tuple[float, float] = (-0.08, 0.10)
    spawn_y_range: tuple[float, float] = (-0.05, 0.12)
    bottle_height: float = 0.18
    object_clearance: float = 0.015
    grasp_center_z_offset: float = 0.005
    grasp_center_offset_object: tuple[float, float, float] | None = None
    desired_base_to_object_x: float = 0.78
    base_position_tolerance: float = 0.015
    base_speed: float = 0.12
    pregrasp_distance: float = 0.13
    lift_height: float = 0.18
    lift_along_grasp_approach: bool = False
    lift_vertical_in_world: bool = False
    minimum_lift_delta: float = 0.07
    preplace_height: float = 0.20
    open_gripper: float = 0.040
    closed_gripper: float = 0.020
    settle_time: float = 0.8
    trajectory_settle_time: float = 0.25
    trajectory_tracking_tolerance: float = 0.06
    preplace_tracking_tolerance: float | None = None
    trajectory_tracking_timeout: float = 4.0
    trajectory_time_scale: float = 1.0
    gripper_hold_time: float = 0.8
    release_hold_time: float = 0.8
    retreat_distance: float = 0.12
    base_settle_linear_velocity: float = 0.03
    base_settle_angular_velocity: float = 0.05
    base_stable_steps: int = 10
    base_settle_timeout: float = 5.0
    max_execution_base_translation: float = 0.04
    max_execution_base_rotation: float = math.radians(5.0)
    grasp_yaw_offsets_deg: tuple[float, ...] = (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0)
    grasp_approach_tilt_deg: float = 5.0
    fixed_grasp_approach_axis_w: tuple[float, float, float] | None = None
    fixed_grasp_closing_axis_w: tuple[float, float, float] | None = None
    post_grasp_approach_axis_w: tuple[float, float, float] | None = None
    post_grasp_closing_axis_w: tuple[float, float, float] | None = None
    overall_timeout: float = 45.0
    stable_success_steps: int = 12
    grasp_approach_mode: str = "side"
    top_down_approach_tilt_deg: float = 30.0
    grasp_pose_stabilization: bool = False
    stabilize_during_release: bool = False
    compensate_grasp_offset_at_target: bool = False
    preserve_grasp_orientation_for_transfer: bool = False
    direct_place_after_lift: bool = False
    hold_object_at_target: bool = False


class CuroboAutoPickPlaceController:
    """Execute and assess one randomized planner-expert episode at a time."""

    _EXECUTION_STATES = {
        "execute_pregrasp",
        "execute_grasp",
        "execute_lift",
        "execute_preplace",
        "execute_place",
        "execute_retreat",
    }

    def __init__(self, robot, joint_targets: torch.Tensor, target_object, cfg: CuroboAutoPickPlaceConfig):
        if cfg.arm not in {"fl", "fr"}:
            raise ValueError(f"Unsupported arm: {cfg.arm}")
        self.robot = robot
        self.joint_targets = joint_targets
        self.target_object = target_object
        self.cfg = cfg
        self.device = robot.device

        self.arm_joint_ids, self.arm_joint_names = robot.find_joints(
            [f"{cfg.arm}_joint[1-6]"], preserve_order=True
        )
        self.gripper_joint_ids, self.gripper_joint_names = robot.find_joints(
            [f"{cfg.arm}_joint[7-8]"], preserve_order=True
        )
        self.ee_body_ids, self.ee_body_names = robot.find_bodies(
            f"{cfg.arm}_link6", preserve_order=True
        )
        if len(self.arm_joint_ids) != 6 or len(self.gripper_joint_ids) != 2 or len(self.ee_body_ids) != 1:
            raise RuntimeError(
                "Could not resolve Mobile ALOHA cuRobo entities: "
                f"arm={self.arm_joint_names}, gripper={self.gripper_joint_names}, ee={self.ee_body_names}"
            )

        table_pos_w = torch.tensor(
            [
                [
                    cfg.table_x,
                    cfg.table_y,
                    cfg.table_height - 0.03 + cfg.planner_table_z_offset,
                ]
            ],
            device=self.device,
        )
        table_quat_w = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            device=self.device,
        )
        table_pos_b, table_quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            table_pos_w,
            table_quat_w,
        )
        self.planner = None
        if cfg.planner_enabled:
            self.planner = MobileAlohaCuroboPlanner(
                cfg.arm,
                table_center_b=table_pos_b[0].detach().cpu().tolist(),
                table_quat_b=table_quat_b[0].detach().cpu().tolist(),
                table_dims=[cfg.table_length, cfg.table_width, 0.06],
                robot_config_stem=cfg.robot_config_stem,
                position_threshold=cfg.planner_position_threshold,
                rotation_threshold=cfg.planner_rotation_threshold,
                self_collision_check=cfg.planner_self_collision_check,
            )

        self.state = "idle"
        self.state_elapsed = 0.0
        self.episode_elapsed = 0.0
        self.trajectory: CuroboTrajectory | None = None
        self.selected_grasp_pos_b: torch.Tensor | None = None
        self.selected_grasp_quat_b: torch.Tensor | None = None
        self.selected_grasp_quat_w: torch.Tensor | None = None
        self.trajectory_base_pos_w: torch.Tensor | None = None
        self.trajectory_base_quat_w: torch.Tensor | None = None
        self.initial_object_z = 0.0
        self.success_stable_steps = 0
        self.base_stable_steps = 0
        self.failure_reason = ""
        self._terminal_reported = False
        self._last_align_report_second = -1
        self._grasp_pose_stabilization_active = False
        self._object_pos_ee: torch.Tensor | None = None
        self._object_quat_ee: torch.Tensor | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in {"done", "failed"}

    @property
    def succeeded(self) -> bool:
        return self.state == "done" and self.success_stable_steps >= self.cfg.stable_success_steps

    def _transition(self, state: str) -> None:
        self.state = state
        self.state_elapsed = 0.0
        self.trajectory = None
        object_pos = self.target_object.data.root_pos_w[0].detach().cpu().tolist()
        print(
            f"[AUTO-CUROBO]: state={state}, object="
            f"({object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f})",
            flush=True,
        )

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self._grasp_pose_stabilization_active = False
        self._transition("failed")

    def _set_gripper(self, opening: float) -> None:
        opening = max(0.0, min(0.04, float(opening)))
        self.joint_targets[:, self.gripper_joint_ids[0]] = opening
        self.joint_targets[:, self.gripper_joint_ids[1]] = -opening

    def randomize_and_start(self, rng: random.Random) -> tuple[float, float, float]:
        """Sample the bottle on the tabletop and start a fresh episode."""

        x = self.cfg.table_x + rng.uniform(*self.cfg.spawn_x_range)
        y = self.cfg.table_y + rng.uniform(*self.cfg.spawn_y_range)
        z = self.cfg.table_height + self.cfg.object_clearance + 0.5 * self.cfg.bottle_height
        root_state = self.target_object.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor([x, y, z], device=root_state.device)
        root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=root_state.device)
        root_state[:, 7:] = 0.0
        self.target_object.write_root_state_to_sim(root_state)

        self.start_from_current_pose()
        print(f"[AUTO-CUROBO]: randomized bottle pose=({x:.3f}, {y:.3f}, {z:.3f})", flush=True)
        return x, y, z

    def start_from_current_pose(self) -> tuple[float, float, float]:
        """Start an episode without changing the target object's current pose."""

        self.joint_targets.copy_(self.robot.data.joint_pos)
        self._set_gripper(self.cfg.open_gripper)
        object_position = self.target_object.data.root_pos_w[0]
        self.initial_object_z = float(object_position[2].item())
        self.success_stable_steps = 0
        self.failure_reason = ""
        self._terminal_reported = False
        self._last_align_report_second = -1
        self.episode_elapsed = 0.0
        self.selected_grasp_pos_b = None
        self.selected_grasp_quat_b = None
        self.selected_grasp_quat_w = None
        self.trajectory_base_pos_w = None
        self.trajectory_base_quat_w = None
        self.base_stable_steps = 0
        self._grasp_pose_stabilization_active = False
        self._object_pos_ee = None
        self._object_quat_ee = None
        self._transition("settle")
        return tuple(float(value.item()) for value in object_position)

    def base_command(self) -> tuple[float, float]:
        if not self.cfg.align_base or self.state != "align_base":
            return 0.0, 0.0
        root_x = float(self.robot.data.root_pos_w[0, 0].item())
        object_x = float(self.target_object.data.root_pos_w[0, 0].item())
        desired_root_x = object_x - self.cfg.desired_base_to_object_x
        error = desired_root_x - root_x
        if abs(error) <= self.cfg.base_position_tolerance:
            return 0.0, 0.0
        return (self.cfg.base_speed if error > 0.0 else -self.cfg.base_speed), 0.0

    def _world_pose_to_base(
        self,
        position_w: torch.Tensor,
        quaternion_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose_w = get_root_pose_w(self.robot.data)
        return math_utils.subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            position_w.view(1, 3),
            quaternion_w.view(1, 4),
        )

    def _base_pose_to_world(
        self,
        position_b: torch.Tensor,
        quaternion_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose_w = get_root_pose_w(self.robot.data)
        return math_utils.combine_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            position_b.to(self.device).view(1, 3),
            quaternion_b.to(self.device).view(1, 4),
        )

    def _current_arm_position(self) -> torch.Tensor:
        return self.robot.data.joint_pos[0, self.arm_joint_ids].detach()

    def _capture_trajectory_base_pose(self) -> None:
        self.trajectory_base_pos_w = self.robot.data.root_pos_w[0].detach().clone()
        self.trajectory_base_quat_w = self.robot.data.root_quat_w[0].detach().clone()

    def _base_is_stable(self) -> bool:
        velocity = self.robot.data.root_link_vel_w[0]
        linear_speed = float(torch.linalg.vector_norm(velocity[:3]).item())
        angular_speed = float(torch.linalg.vector_norm(velocity[3:]).item())
        return (
            linear_speed <= self.cfg.base_settle_linear_velocity
            and angular_speed <= self.cfg.base_settle_angular_velocity
        )

    def _report_base_pose(self) -> None:
        root_pose = get_root_pose_w(self.robot.data)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_pose[:, 3:7])
        print(
            "[AUTO-CUROBO]: stable base pose="
            f"({root_pose[0, 0].item():.3f}, {root_pose[0, 1].item():.3f}, "
            f"{root_pose[0, 2].item():.3f}), "
            f"rpy_deg=({math.degrees(roll[0].item()):.2f}, "
            f"{math.degrees(pitch[0].item()):.2f}, "
            f"{math.degrees(yaw[0].item()):.2f})",
            flush=True,
        )

    def _refresh_table_obstacle(self) -> None:
        root_pose = get_root_pose_w(self.robot.data)
        table_pos_w = torch.tensor(
            [
                [
                    self.cfg.table_x,
                    self.cfg.table_y,
                    self.cfg.table_height
                    - 0.03
                    + self.cfg.planner_table_z_offset,
                ]
            ],
            device=self.device,
        )
        table_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        table_pos_b, table_quat_b = math_utils.subtract_frame_transforms(
            root_pose[:, :3], root_pose[:, 3:7], table_pos_w, table_quat_w
        )
        self.planner.update_table(
            table_pos_b[0].detach().cpu().tolist(),
            table_quat_b[0].detach().cpu().tolist(),
            [self.cfg.table_length, self.cfg.table_width, 0.06],
        )

    def _table_aligned_grasp_orientations_w(self, object_pos_w: torch.Tensor) -> torch.Tensor:
        """Keep PiPER's local-X jaw axis level and local-Y axis aligned with the bottle."""

        fixed_approach = self.cfg.fixed_grasp_approach_axis_w
        fixed_closing = self.cfg.fixed_grasp_closing_axis_w
        if (fixed_approach is None) != (fixed_closing is None):
            raise ValueError(
                "fixed grasp approach and closing axes must be configured together"
            )
        if fixed_approach is not None and fixed_closing is not None:
            return self._fixed_axis_orientations_w(
                fixed_approach,
                fixed_closing,
                object_pos_w.dtype,
            )

        ee_pos_w, _ = self._current_ee_pose_w()
        direct_approach = object_pos_w - ee_pos_w[0]
        direct_approach[2] = 0.0
        if float(torch.linalg.vector_norm(direct_approach).item()) < 1e-4:
            direct_approach = object_pos_w - self.robot.data.root_pos_w[0]
            direct_approach[2] = 0.0
        direct_approach = torch.nn.functional.normalize(direct_approach, dim=0)

        offsets = torch.deg2rad(
            torch.tensor(
                self.cfg.grasp_yaw_offsets_deg,
                device=self.device,
                dtype=direct_approach.dtype,
            )
        )
        cos_yaw = torch.cos(offsets)
        sin_yaw = torch.sin(offsets)
        approach_axes_w = torch.stack(
            (
                cos_yaw * direct_approach[0] - sin_yaw * direct_approach[1],
                sin_yaw * direct_approach[0] + cos_yaw * direct_approach[1],
                torch.zeros_like(offsets),
            ),
            dim=-1,
        )
        approach_axes_w = torch.nn.functional.normalize(approach_axes_w, dim=-1)

        world_up = torch.zeros_like(approach_axes_w)
        world_up[:, 2] = 1.0
        closing_axes_w = torch.linalg.cross(world_up, approach_axes_w, dim=-1)
        closing_axes_w = torch.nn.functional.normalize(closing_axes_w, dim=-1)

        approach_tilt = math.radians(self.cfg.grasp_approach_tilt_deg)
        tool_y_axes_w = (
            math.cos(approach_tilt) * world_up
            + math.sin(approach_tilt) * approach_axes_w
        )
        approach_axes_w = (
            -math.sin(approach_tilt) * world_up
            + math.cos(approach_tilt) * approach_axes_w
        )
        rotation_matrices_w = torch.stack(
            (closing_axes_w, tool_y_axes_w, approach_axes_w),
            dim=-1,
        )
        return math_utils.quat_from_matrix(rotation_matrices_w)

    def _fixed_axis_orientations_w(
        self,
        approach_axis: tuple[float, float, float],
        closing_axis: tuple[float, float, float],
        dtype: torch.dtype,
        yaw_offsets_deg: tuple[float, ...] = (0.0,),
    ) -> torch.Tensor:
        approach_axis_w = torch.tensor(
            approach_axis,
            device=self.device,
            dtype=dtype,
        )
        closing_axis_w = torch.tensor(
            closing_axis,
            device=self.device,
            dtype=dtype,
        )
        approach_axis_w = torch.nn.functional.normalize(approach_axis_w, dim=0)
        closing_axis_w = closing_axis_w - torch.dot(
            closing_axis_w, approach_axis_w
        ) * approach_axis_w
        if float(torch.linalg.vector_norm(closing_axis_w).item()) < 1e-4:
            raise ValueError(
                "fixed grasp approach and closing axes must not be parallel"
            )
        closing_axis_w = torch.nn.functional.normalize(closing_axis_w, dim=0)
        offsets = torch.deg2rad(
            torch.tensor(yaw_offsets_deg, device=self.device, dtype=dtype)
        )
        cos_offsets = torch.cos(offsets).view(-1, 1)
        sin_offsets = torch.sin(offsets).view(-1, 1)
        base_closing = closing_axis_w.view(1, 3).repeat(len(offsets), 1)
        rotation_axes = approach_axis_w.view(1, 3).repeat(len(offsets), 1)
        rotated_closing = (
            cos_offsets * base_closing
            + sin_offsets * torch.linalg.cross(
                rotation_axes, base_closing, dim=-1
            )
        )
        closing_axes_w = torch.cat((rotated_closing, -rotated_closing), dim=0)
        approach_axes_w = approach_axis_w.view(1, 3).repeat(
            closing_axes_w.shape[0], 1
        )
        tool_y_axes_w = torch.linalg.cross(
            approach_axes_w, closing_axes_w, dim=-1
        )
        rotation_matrices_w = torch.stack(
            (closing_axes_w, tool_y_axes_w, approach_axes_w), dim=-1
        )
        return math_utils.quat_from_matrix(rotation_matrices_w)

    def _top_down_grasp_orientations_w(self, object_pos_w: torch.Tensor) -> torch.Tensor:
        """Generate vertical approach poses while keeping the jaw-closing axis horizontal."""

        fixed_approach = self.cfg.fixed_grasp_approach_axis_w
        fixed_closing = self.cfg.fixed_grasp_closing_axis_w
        if (fixed_approach is None) != (fixed_closing is None):
            raise ValueError(
                "fixed grasp approach and closing axes must be configured together"
            )
        if fixed_approach is not None and fixed_closing is not None:
            return self._fixed_axis_orientations_w(
                fixed_approach,
                fixed_closing,
                object_pos_w.dtype,
                self.cfg.grasp_yaw_offsets_deg,
            )

        ee_pos_w, _ = self._current_ee_pose_w()
        direct_approach = object_pos_w - ee_pos_w[0]
        direct_approach[2] = 0.0
        if float(torch.linalg.vector_norm(direct_approach).item()) < 1e-4:
            direct_approach = torch.tensor(
                [1.0, 0.0, 0.0],
                device=self.device,
                dtype=object_pos_w.dtype,
            )
        direct_approach = torch.nn.functional.normalize(direct_approach, dim=0)

        offsets = torch.deg2rad(
            torch.tensor(
                self.cfg.grasp_yaw_offsets_deg,
                device=self.device,
                dtype=direct_approach.dtype,
            )
        )
        cos_yaw = torch.cos(offsets)
        sin_yaw = torch.sin(offsets)
        horizontal_approach_axes_w = torch.stack(
            (
                cos_yaw * direct_approach[0] - sin_yaw * direct_approach[1],
                sin_yaw * direct_approach[0] + cos_yaw * direct_approach[1],
                torch.zeros_like(offsets),
            ),
            dim=-1,
        )
        horizontal_approach_axes_w = torch.nn.functional.normalize(
            horizontal_approach_axes_w, dim=-1
        )
        world_up = torch.zeros_like(horizontal_approach_axes_w)
        world_up[:, 2] = 1.0
        closing_axes_w = torch.linalg.cross(
            world_up, horizontal_approach_axes_w, dim=-1
        )
        tilt = math.radians(self.cfg.top_down_approach_tilt_deg)
        approach_axes_w = (
            -math.cos(tilt) * world_up
            + math.sin(tilt) * horizontal_approach_axes_w
        )
        tool_y_axes_w = torch.linalg.cross(
            approach_axes_w, closing_axes_w, dim=-1
        )
        rotation_matrices_w = torch.stack(
            (closing_axes_w, tool_y_axes_w, approach_axes_w),
            dim=-1,
        )
        return math_utils.quat_from_matrix(rotation_matrices_w)

    def _plan_pregrasp(self) -> bool:
        self._refresh_table_obstacle()
        current_q = self._current_arm_position()
        print(f"[AUTO-CUROBO]: planning pregrasp from q={current_q.cpu().tolist()}", flush=True)
        object_pos_w = self._current_grasp_center_w()
        if self.cfg.grasp_approach_mode == "top_down":
            quaternions_w = self._top_down_grasp_orientations_w(object_pos_w)
        elif self.cfg.grasp_approach_mode == "side":
            quaternions_w = self._table_aligned_grasp_orientations_w(object_pos_w)
        else:
            raise ValueError(
                f"Unsupported grasp approach mode: {self.cfg.grasp_approach_mode}"
            )
        candidate_count = len(quaternions_w)
        root_pose_w = get_root_pose_w(self.robot.data).repeat(candidate_count, 1)
        grasp_positions_w = object_pos_w.repeat(candidate_count, 1)
        grasp_positions_b, quaternions_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            grasp_positions_w,
            quaternions_w,
        )
        print(
            f"[AUTO-CUROBO]: generated {candidate_count} "
            f"{self.cfg.grasp_approach_mode} grasp candidates",
            flush=True,
        )
        local_z = torch.tensor([0.0, 0.0, 1.0]).repeat(candidate_count, 1)
        approach_axes_b = _quat_apply_wxyz(quaternions_b.cpu(), local_z)
        pregrasp_positions_b = (
            grasp_positions_b.cpu() - self.cfg.pregrasp_distance * approach_axes_b
        )
        print("[AUTO-CUROBO]: submitting pregrasp candidates to cuRobo", flush=True)
        trajectory = self.planner.plan_to_goalset(
            current_q,
            pregrasp_positions_b,
            quaternions_b.cpu(),
            enable_finetune_trajopt=self.cfg.planner_enable_finetune_trajopt,
        )
        if trajectory is None:
            return False
        self.selected_grasp_pos_b = grasp_positions_b[trajectory.goalset_index].detach().cpu()
        self.selected_grasp_quat_b = quaternions_b[trajectory.goalset_index].detach().cpu()
        self.selected_grasp_quat_w = quaternions_w[trajectory.goalset_index].detach()
        self.trajectory = trajectory
        self._capture_trajectory_base_pose()
        selected_rotation_w = math_utils.matrix_from_quat(
            self.selected_grasp_quat_w.view(1, 4)
        )[0]
        closing_vertical = abs(float(selected_rotation_w[2, 0].item()))
        tool_up_alignment = abs(float(selected_rotation_w[2, 1].item()))
        approach_vertical = abs(float(selected_rotation_w[2, 2].item()))
        print(
            f"[AUTO-CUROBO]: pregrasp candidate={trajectory.goalset_index}, "
            f"waypoints={len(trajectory.position)}, solve={trajectory.solve_time:.3f}s, "
            f"closing_vertical={closing_vertical:.4f}, "
            f"tool_up_alignment={tool_up_alignment:.4f}, "
            f"approach_vertical={approach_vertical:.4f}",
            flush=True,
        )
        return True

    def _plan_pose(self, position_b: torch.Tensor, quaternion_b: torch.Tensor) -> bool:
        self._refresh_table_obstacle()
        trajectory = self.planner.plan_to_goalset(
            self._current_arm_position(),
            position_b.view(1, 3),
            quaternion_b.view(1, 4),
            enable_finetune_trajopt=self.cfg.planner_enable_finetune_trajopt,
        )
        if trajectory is None:
            return False
        self.trajectory = trajectory
        self._capture_trajectory_base_pose()
        print(
            f"[AUTO-CUROBO]: trajectory waypoints={len(trajectory.position)}, "
            f"solve={trajectory.solve_time:.3f}s",
            flush=True,
        )
        return True

    def _plan_world_pose(self, position_w: torch.Tensor, quaternion_w: torch.Tensor) -> bool:
        position_b, quaternion_b = self._world_pose_to_base(position_w, quaternion_w)
        return self._plan_pose(position_b[0], quaternion_b[0])

    def _plan_world_pose_goalset(
        self,
        position_w: torch.Tensor,
        quaternions_w: torch.Tensor,
    ) -> bool:
        candidate_count = len(quaternions_w)
        if position_w.ndim == 1:
            positions_w = position_w.view(1, 3).repeat(candidate_count, 1)
        elif position_w.shape == (candidate_count, 3):
            positions_w = position_w
        else:
            raise ValueError(
                "Goal-set positions must be one XYZ position or one per orientation"
            )
        root_pose_w = get_root_pose_w(self.robot.data).repeat(candidate_count, 1)
        positions_b, quaternions_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            positions_w,
            quaternions_w,
        )
        self._refresh_table_obstacle()
        trajectory = self.planner.plan_to_goalset(
            self._current_arm_position(),
            positions_b.detach().cpu(),
            quaternions_b.detach().cpu(),
            enable_finetune_trajopt=self.cfg.planner_enable_finetune_trajopt,
        )
        if trajectory is None:
            return False
        selected_index = trajectory.goalset_index
        self.selected_grasp_quat_w = quaternions_w[selected_index].detach()
        self.selected_grasp_quat_b = (
            quaternions_b[selected_index].detach().cpu()
        )
        self.trajectory = trajectory
        self._capture_trajectory_base_pose()
        print(
            f"[AUTO-CUROBO]: selected transfer orientation={selected_index}, "
            f"waypoints={len(trajectory.position)}, solve={trajectory.solve_time:.3f}s",
            flush=True,
        )
        return True

    def _current_ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose_b = self.planner.forward_pose(self._current_arm_position())
        return self._base_pose_to_world(pose_b.position[0], pose_b.quaternion[0])

    def _current_grasp_center_w(self) -> torch.Tensor:
        object_pos_w = self.target_object.data.root_pos_w[0]
        offset_object = self.cfg.grasp_center_offset_object
        if offset_object is None:
            grasp_pos_w = object_pos_w.detach().clone()
            grasp_pos_w[2] += self.cfg.grasp_center_z_offset
            return grasp_pos_w
        offset = torch.tensor(
            offset_object,
            device=self.device,
            dtype=object_pos_w.dtype,
        ).view(1, 3)
        return object_pos_w + math_utils.quat_apply(
            self.target_object.data.root_quat_w,
            offset,
        )[0]

    def _current_object_center_b(self) -> torch.Tensor:
        object_pos_w = self.target_object.data.root_pos_w[0]
        object_quat_w = self.target_object.data.root_quat_w[0]
        object_pos_b, _ = self._world_pose_to_base(object_pos_w, object_quat_w)
        center = object_pos_b[0]
        center[2] += 0.005
        return center

    def _capture_grasp_pose_stabilization(self) -> None:
        if not self.cfg.grasp_pose_stabilization:
            return
        ee_pos_w, ee_quat_w = self._current_ee_pose_w()
        object_pos_w = self.target_object.data.root_pos_w
        object_quat_w = self.target_object.data.root_quat_w
        object_pos_ee, object_quat_ee = math_utils.subtract_frame_transforms(
            ee_pos_w,
            ee_quat_w,
            object_pos_w,
            object_quat_w,
        )
        self._object_pos_ee = object_pos_ee[0].detach()
        self._object_quat_ee = object_quat_ee[0].detach()
        self._grasp_pose_stabilization_active = True

    def _apply_grasp_pose_stabilization(self) -> None:
        if (
            not self._grasp_pose_stabilization_active
            or self._object_pos_ee is None
            or self._object_quat_ee is None
        ):
            return
        ee_pos_w, ee_quat_w = self._current_ee_pose_w()
        object_pos_w, object_quat_w = math_utils.combine_frame_transforms(
            ee_pos_w,
            ee_quat_w,
            self._object_pos_ee.view(1, 3),
            self._object_quat_ee.view(1, 4),
        )
        root_state = self.target_object.data.root_state_w.clone()
        root_state[:, :3] = object_pos_w
        root_state[:, 3:7] = object_quat_w
        root_state[:, 7:] = 0.0
        self.target_object.write_root_state_to_sim(root_state)

    def _execute_trajectory(self) -> bool:
        if self.trajectory is None:
            self._fail("execution state has no trajectory")
            return False
        if self.trajectory_base_pos_w is not None and self.trajectory_base_quat_w is not None:
            translation = torch.linalg.vector_norm(
                self.robot.data.root_pos_w[0] - self.trajectory_base_pos_w
            )
            rotation = math_utils.quat_error_magnitude(
                self.robot.data.root_quat_w[0].view(1, 4),
                self.trajectory_base_quat_w.view(1, 4),
            )[0]
            if (
                float(translation.item()) > self.cfg.max_execution_base_translation
                or float(rotation.item()) > self.cfg.max_execution_base_rotation
            ):
                self._fail(
                    "base moved during trajectory "
                    f"(translation={translation.item():.3f} m, "
                    f"rotation={math.degrees(rotation.item()):.2f} deg)"
                )
                return False
        playback_dt = self.trajectory.dt * self.cfg.trajectory_time_scale
        index = min(int(self.state_elapsed / playback_dt), len(self.trajectory.position) - 1)
        target = self.trajectory.position[index].to(self.joint_targets.device)
        self.joint_targets[:, self.arm_joint_ids] = target
        motion_time = len(self.trajectory.position) * playback_dt
        if self.state_elapsed < motion_time + self.cfg.trajectory_settle_time:
            return False
        actual = self.robot.data.joint_pos[0, self.arm_joint_ids]
        final_target = self.trajectory.position[-1].to(actual.device)
        tracking_error = float(torch.max(torch.abs(actual - final_target)).item())
        tracking_tolerance = self.cfg.trajectory_tracking_tolerance
        if (
            self.state == "execute_preplace"
            and self.cfg.preplace_tracking_tolerance is not None
        ):
            tracking_tolerance = self.cfg.preplace_tracking_tolerance
        if tracking_error <= tracking_tolerance:
            return True
        if self.state_elapsed >= motion_time + self.cfg.trajectory_tracking_timeout:
            joint_errors = torch.abs(actual - final_target)
            worst_index = int(torch.argmax(joint_errors).item())
            worst_joint = self.arm_joint_names[worst_index]
            self._fail(
                f"{self.state} arm tracking error {tracking_error:.3f} rad "
                f"at {worst_joint} "
                f"(actual={float(actual[worst_index]):.3f}, "
                f"target={float(final_target[worst_index]):.3f})"
            )
        return False

    def _object_lifted(self) -> bool:
        if self._grasp_pose_stabilization_active:
            return True
        return (
            float(self.target_object.data.root_pos_w[0, 2].item())
            >= self.initial_object_z + self.cfg.minimum_lift_delta
        )

    def _target_center_z(self) -> float:
        if self.cfg.target_center_z is not None:
            return self.cfg.target_center_z
        return self.cfg.table_height + 0.5 * self.cfg.bottle_height

    def _update_success(self) -> None:
        pos = self.target_object.data.root_pos_w[0]
        vel = self.target_object.data.root_lin_vel_w[0]
        success_x = (
            self.cfg.target_x
            if self.cfg.success_target_x is None
            else self.cfg.success_target_x
        )
        success_y = (
            self.cfg.target_y
            if self.cfg.success_target_y is None
            else self.cfg.success_target_y
        )
        radial = torch.linalg.vector_norm(
            pos[:2] - torch.tensor([success_x, success_y], device=pos.device)
        )
        expected_z = self._target_center_z()
        stable = (
            float(radial.item()) <= self.cfg.target_radius
            and abs(float(pos[2].item()) - expected_z) <= self.cfg.target_z_tolerance
            and float(torch.linalg.vector_norm(vel).item()) <= self.cfg.target_max_linear_speed
        )
        self.success_stable_steps = self.success_stable_steps + 1 if stable else 0

    def update(self, dt: float) -> None:
        if self.is_terminal:
            self._update_success()
            return
        self._apply_grasp_pose_stabilization()
        self.state_elapsed += dt
        self.episode_elapsed += dt
        if self.episode_elapsed > self.cfg.overall_timeout:
            self._fail("episode timeout")
            return

        if self.state == "settle":
            self.base_stable_steps = self.base_stable_steps + 1 if self._base_is_stable() else 0
            if (
                self.state_elapsed >= self.cfg.settle_time
                and self.base_stable_steps >= self.cfg.base_stable_steps
            ):
                self._report_base_pose()
                if self.cfg.align_base:
                    self._transition("align_base")
                else:
                    self._refresh_table_obstacle()
                    self._transition("plan_pregrasp")
            elif self.state_elapsed >= self.cfg.base_settle_timeout:
                self._fail("mobile base did not settle on terrain")
            return
        if self.state == "align_base":
            root_x = float(self.robot.data.root_pos_w[0, 0].item())
            object_x = float(self.target_object.data.root_pos_w[0, 0].item())
            desired_x = object_x - self.cfg.desired_base_to_object_x
            report_second = int(self.state_elapsed)
            if report_second != self._last_align_report_second:
                self._last_align_report_second = report_second
                print(
                    f"[AUTO-CUROBO]: align root_x={root_x:.3f}, desired_x={desired_x:.3f}, "
                    f"command={self.base_command()[0]:.3f}",
                    flush=True,
                )
            if abs(root_x - desired_x) <= self.cfg.base_position_tolerance:
                self._refresh_table_obstacle()
                self._transition("plan_pregrasp")
            return
        if self.state == "plan_pregrasp":
            if self._plan_pregrasp():
                self.state = "execute_pregrasp"
                self.state_elapsed = 0.0
            else:
                self._fail("no collision-free pregrasp")
            return
        if self.state == "execute_pregrasp" and self._execute_trajectory():
            if self.selected_grasp_quat_w is None:
                self._fail("pregrasp candidate was lost")
                return
            grasp_pos_w = self._current_grasp_center_w()
            if self._plan_world_pose(grasp_pos_w, self.selected_grasp_quat_w):
                self.state = "execute_grasp"
                self.state_elapsed = 0.0
            else:
                self._fail("no collision-free grasp approach")
            return
        if self.state == "execute_grasp" and self._execute_trajectory():
            self._set_gripper(self.cfg.closed_gripper)
            self._transition("close")
            return
        if self.state == "close":
            self._set_gripper(self.cfg.closed_gripper)
            if self.state_elapsed >= self.cfg.gripper_hold_time:
                if self.selected_grasp_quat_w is None:
                    self._fail("missing grasp pose")
                    return
                post_approach = self.cfg.post_grasp_approach_axis_w
                post_closing = self.cfg.post_grasp_closing_axis_w
                if (post_approach is None) != (post_closing is None):
                    raise ValueError(
                        "post-grasp approach and closing axes must be configured together"
                    )
                if post_approach is not None and post_closing is not None:
                    self._capture_grasp_pose_stabilization()
                    ee_pos_w, _ = self._current_ee_pose_w()
                    lift_pos_w = ee_pos_w[0].detach().clone()
                    lift_pos_w[2] += self.cfg.lift_height
                    lift_orientations_w = self._fixed_axis_orientations_w(
                        post_approach,
                        post_closing,
                        lift_pos_w.dtype,
                    )
                    lift_planned = self._plan_world_pose_goalset(
                        lift_pos_w,
                        lift_orientations_w,
                    )
                elif self.cfg.lift_vertical_in_world:
                    self._capture_grasp_pose_stabilization()
                    ee_pos_w, ee_quat_w = self._current_ee_pose_w()
                    lift_pos_w = ee_pos_w[0].detach().clone()
                    lift_pos_w[2] += self.cfg.lift_height
                    lift_planned = self._plan_world_pose(
                        lift_pos_w, ee_quat_w[0]
                    )
                elif (
                    (
                        self.cfg.grasp_approach_mode == "top_down"
                        or self.cfg.lift_along_grasp_approach
                    )
                    and self.selected_grasp_pos_b is not None
                    and self.selected_grasp_quat_b is not None
                ):
                    self._capture_grasp_pose_stabilization()
                    local_z = torch.tensor(
                        [[0.0, 0.0, 1.0]],
                        dtype=self.selected_grasp_pos_b.dtype,
                    )
                    approach_axis_b = _quat_apply_wxyz(
                        self.selected_grasp_quat_b.view(1, 4), local_z
                    )[0]
                    lift_pos_b = (
                        self.selected_grasp_pos_b
                        - self.cfg.lift_height * approach_axis_b
                    )
                    lift_planned = self._plan_pose(
                        lift_pos_b, self.selected_grasp_quat_b
                    )
                else:
                    self._capture_grasp_pose_stabilization()
                    ee_pos_w, ee_quat_w = self._current_ee_pose_w()
                    lift_pos_w = ee_pos_w[0].detach().clone()
                    lift_pos_w[2] += self.cfg.lift_height
                    lift_planned = self._plan_world_pose(lift_pos_w, ee_quat_w[0])
                if lift_planned:
                    self.state = "execute_lift"
                    self.state_elapsed = 0.0
                else:
                    self._fail("no collision-free lift")
            return
        if self.state == "execute_lift" and self._execute_trajectory():
            if not self._object_lifted():
                self._fail("bottle was not lifted")
                return
            if self.cfg.preserve_grasp_orientation_for_transfer:
                _, realized_grasp_quat_w = self._current_ee_pose_w()
                self.selected_grasp_quat_w = realized_grasp_quat_w[0].detach().clone()
            transfer_height = (
                0.005
                if self.cfg.direct_place_after_lift
                else self.cfg.preplace_height
            )
            desired_object_w = torch.tensor(
                [
                    self.cfg.target_x,
                    self.cfg.target_y,
                    self._target_center_z() + transfer_height,
                ],
                device=self.device,
            )
            if self.selected_grasp_quat_w is None:
                self._fail("missing world-frame grasp orientation")
                return
            if self.cfg.target_functional_aim_point_w is not None:
                if (
                    self.cfg.target_functional_offset_object is None
                    or self.cfg.target_functional_axis_object is None
                ):
                    self._fail(
                        "functional aim requires both a local point and a local axis"
                    )
                    return
                desired_object_quaternion_w = (
                    self.target_object.data.root_quat_w[0].detach().clone()
                )
                functional_offset_object = torch.tensor(
                    self.cfg.target_functional_offset_object,
                    device=self.device,
                    dtype=desired_object_w.dtype,
                )
                functional_axis_object = torch.tensor(
                    self.cfg.target_functional_axis_object,
                    device=self.device,
                    dtype=desired_object_w.dtype,
                )
                functional_axis_w = math_utils.quat_apply(
                    desired_object_quaternion_w.view(1, 4),
                    functional_axis_object.view(1, 3),
                )[0]
                functional_axis_w = torch.nn.functional.normalize(
                    functional_axis_w, dim=0
                )
                aim_point_w = torch.tensor(
                    self.cfg.target_functional_aim_point_w,
                    device=self.device,
                    dtype=desired_object_w.dtype,
                )
                desired_functional_point_w = (
                    aim_point_w
                    - self.cfg.target_functional_standoff * functional_axis_w
                )
                desired_object_position_w = (
                    desired_functional_point_w
                    - math_utils.quat_apply(
                        desired_object_quaternion_w.view(1, 4),
                        functional_offset_object.view(1, 3),
                    )[0]
                )
                current_ee_position_w, current_ee_quaternion_w = (
                    self._current_ee_pose_w()
                )
                current_object_position_w = (
                    self.target_object.data.root_pos_w[0].detach().clone()
                )
                object_delta_w = (
                    desired_object_position_w - current_object_position_w
                )
                target_w = current_ee_position_w + object_delta_w
                target_quaternion_w = current_ee_quaternion_w
                print(
                    "[AUTO-CUROBO]: functional scan transfer "
                    f"object_delta={tuple(float(value) for value in object_delta_w)}, "
                    f"axis={tuple(float(value) for value in functional_axis_w)}",
                    flush=True,
                )
                preplace_planned = self._plan_world_pose(
                    target_w[0], target_quaternion_w[0]
                )
            elif (
                self.cfg.grasp_approach_mode == "top_down"
                and not self.cfg.preserve_grasp_orientation_for_transfer
            ):
                transfer_orientations_w = (
                    self._top_down_grasp_orientations_w(desired_object_w)
                )
                if (
                    self.cfg.compensate_grasp_offset_at_target
                    and self._object_pos_ee is not None
                ):
                    object_offsets_w = math_utils.quat_apply(
                        transfer_orientations_w,
                        self._object_pos_ee.view(1, 3).repeat(
                            len(transfer_orientations_w), 1
                        ),
                    )
                    preplace_positions_w = (
                        desired_object_w.view(1, 3) - object_offsets_w
                    )
                else:
                    preplace_positions_w = desired_object_w.clone()
                    preplace_positions_w[2] += self.cfg.grasp_center_z_offset
                preplace_planned = self._plan_world_pose_goalset(
                    preplace_positions_w, transfer_orientations_w
                )
            else:
                if (
                    self.cfg.compensate_grasp_offset_at_target
                    and self._object_pos_ee is not None
                ):
                    object_offset_w = math_utils.quat_apply(
                        self.selected_grasp_quat_w.view(1, 4),
                        self._object_pos_ee.view(1, 3),
                    )[0]
                    target_w = desired_object_w - object_offset_w
                else:
                    target_w = desired_object_w.clone()
                    target_w[2] += self.cfg.grasp_center_z_offset
                preplace_planned = self._plan_world_pose(
                    target_w, self.selected_grasp_quat_w
                )
            if preplace_planned:
                self.state = (
                    "execute_place"
                    if self.cfg.direct_place_after_lift
                    else "execute_preplace"
                )
                self.state_elapsed = 0.0
            else:
                self._fail("no collision-free preplace")
            return
        if self.state == "execute_preplace" and self._execute_trajectory():
            desired_object_w = torch.tensor(
                [
                    self.cfg.target_x,
                    self.cfg.target_y,
                    self._target_center_z() + 0.005,
                ],
                device=self.device,
            )
            if self.selected_grasp_quat_w is None:
                self._fail("missing world-frame grasp orientation")
                return
            if (
                self.cfg.compensate_grasp_offset_at_target
                and self._object_pos_ee is not None
            ):
                object_offset_w = math_utils.quat_apply(
                    self.selected_grasp_quat_w.view(1, 4),
                    self._object_pos_ee.view(1, 3),
                )[0]
                target_w = desired_object_w - object_offset_w
            else:
                target_w = desired_object_w.clone()
                target_w[2] += self.cfg.grasp_center_z_offset
            if self._plan_world_pose(target_w, self.selected_grasp_quat_w):
                self.state = "execute_place"
                self.state_elapsed = 0.0
            else:
                self._fail("no collision-free place")
            return
        if self.state == "execute_place" and self._execute_trajectory():
            self._apply_grasp_pose_stabilization()
            if not self.cfg.stabilize_during_release:
                self._grasp_pose_stabilization_active = False
            if self.cfg.hold_object_at_target:
                self._set_gripper(self.cfg.closed_gripper)
                self.success_stable_steps = self.cfg.stable_success_steps
                self._transition("done")
                return
            self._set_gripper(self.cfg.open_gripper)
            self._transition("release")
            return
        if self.state == "release":
            self._set_gripper(self.cfg.open_gripper)
            if self.cfg.stabilize_during_release:
                self._apply_grasp_pose_stabilization()
            if self.state_elapsed >= self.cfg.release_hold_time:
                self._grasp_pose_stabilization_active = False
                ee_pos_w, ee_quat_w = self._current_ee_pose_w()
                retreat_pos_w = ee_pos_w[0].detach().clone()
                retreat_pos_w[2] += self.cfg.retreat_distance
                if self._plan_world_pose(retreat_pos_w, ee_quat_w[0]):
                    self.state = "execute_retreat"
                    self.state_elapsed = 0.0
                else:
                    print(
                        "[AUTO-CUROBO]: retreat unavailable after release; "
                        "continuing with object verification",
                        flush=True,
                    )
                    self._transition("verify")
            return
        if self.state == "execute_retreat" and self._execute_trajectory():
            self._transition("verify")
            return
        if self.state == "verify":
            self._update_success()
            if self.success_stable_steps >= self.cfg.stable_success_steps:
                self._transition("done")
            elif self.state_elapsed >= 3.0:
                self._fail("object did not settle inside target zone")

    def report_terminal_once(self) -> None:
        if not self.is_terminal or self._terminal_reported:
            return
        self._terminal_reported = True
        pos = self.target_object.data.root_pos_w[0].detach().cpu().tolist()
        root_x = float(self.robot.data.root_pos_w[0, 0].item())
        print(
            f"[AUTO-CUROBO]: terminal={self.state}, success={self.succeeded}, "
            f"reason={self.failure_reason or 'success'}, "
            f"object=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
            f"root_x={root_x:.3f}, "
            f"elapsed={self.episode_elapsed:.2f}s",
            flush=True,
        )
