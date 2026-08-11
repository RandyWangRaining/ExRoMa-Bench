"""RoboTwin-style automatic hammer task using cuRobo motion generation."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import random
from typing import Literal

import torch

from exroma_bench.sim.isaaclab_compat import get_root_pose_w
import isaaclab.utils.math as math_utils

from exroma_bench.tasks.mobile_aloha_pick_place.curobo_planner import (
    CuroboTrajectory,
    MobileAlohaCuroboPlanner,
)


ArmName = Literal["fl", "fr"]


@dataclass
class CuroboBeatBlockHammerConfig:
    """Task geometry and execution parameters adapted from RoboTwin."""

    arm: Literal["auto", "fl", "fr"] = "auto"
    table_x: float = 0.0
    table_y: float = -0.85
    table_height: float = 0.6
    table_length: float = 0.8
    table_width: float = 0.55
    planner_table_clearance: float = 0.04
    robot_config_stem: str = "dual_piper"
    planner_position_threshold: float = 0.01
    planner_rotation_threshold: float = 0.08
    planner_self_collision_check: bool = False
    planner_enabled: bool = True
    block_x_range: tuple[float, float] = (-0.25, 0.25)
    block_y_range: tuple[float, float] = (-0.05, 0.15)
    block_half_size: float = 0.025
    block_yaw_limit: float = 0.5
    hammer_table_x: float = 0.0
    hammer_table_y: float = -0.06
    hammer_root_height: float = 0.043
    hammer_functional_offset: tuple[float, float, float] = (0.0, 0.0632, 0.05135)
    hammer_object_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.994505, 0.104947)
    open_gripper: float = 0.035
    closed_gripper: float = 0.008
    pregrasp_distance: float = 0.12
    grasp_distance: float = 0.005
    grasp_approach_tilt_deg: float = 22.0
    lift_height: float = 0.09
    prestrike_height: float = 0.06
    settle_time: float = 0.8
    base_stable_steps: int = 10
    base_settle_linear_velocity: float = 0.03
    base_settle_angular_velocity: float = 0.05
    base_settle_timeout: float = 5.0
    gripper_hold_time: float = 0.8
    strike_hold_time: float = 0.5
    trajectory_settle_time: float = 0.25
    trajectory_tracking_tolerance: float = 0.10
    trajectory_tracking_timeout: float = 4.0
    max_execution_base_translation: float = 0.04
    max_execution_base_rotation: float = math.radians(5.0)
    grasp_yaw_offsets_deg: tuple[float, ...] = (
        0.0,
        180.0,
        -15.0,
        15.0,
        165.0,
        -165.0,
    )
    strike_yaw_offsets_deg: tuple[float, ...] = (0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0)
    contact_force_threshold: float = 1e-3
    success_xy_tolerance: float = 0.02
    stable_success_steps: int = 1
    overall_timeout: float = 55.0


class CuroboBeatBlockHammerController:
    """Run the RoboTwin hammer expert against Isaac Lab rigid-body physics."""

    def __init__(
        self,
        robot,
        joint_targets: torch.Tensor,
        hammer,
        block,
        contact_sensor,
        cfg: CuroboBeatBlockHammerConfig,
    ) -> None:
        self.robot = robot
        self.joint_targets = joint_targets
        self.hammer = hammer
        self.block = block
        self.contact_sensor = contact_sensor
        self.cfg = cfg
        self.device = robot.device

        self.planners: dict[ArmName, MobileAlohaCuroboPlanner] = {}
        self.planner: MobileAlohaCuroboPlanner | None = None
        self.arm: ArmName | None = None
        self.arm_joint_ids: list[int] = []
        self.arm_joint_names: list[str] = []
        self.gripper_joint_ids: list[int] = []
        self.gripper_joint_names: list[str] = []

        self.state = "idle"
        self.state_elapsed = 0.0
        self.episode_elapsed = 0.0
        self.trajectory: CuroboTrajectory | None = None
        self.trajectory_base_pos_w: torch.Tensor | None = None
        self.trajectory_base_quat_w: torch.Tensor | None = None
        self.selected_grasp_quat_w: torch.Tensor | None = None
        self.ee_to_hammer_pos: torch.Tensor | None = None
        self.ee_to_hammer_quat: torch.Tensor | None = None
        self.strike_hammer_quat_w: torch.Tensor | None = None
        self.initial_hammer_z = 0.0
        self.contact_seen = False
        self.current_contact_force = 0.0
        self.success_stable_steps = 0
        self.base_stable_steps = 0
        self.failure_reason = ""
        self._terminal_reported = False

    @property
    def is_terminal(self) -> bool:
        return self.state in {"done", "failed"}

    @property
    def succeeded(self) -> bool:
        return self.state == "done" and self.success_stable_steps >= self.cfg.stable_success_steps

    def base_command(self) -> tuple[float, float]:
        return 0.0, 0.0

    def _transition(self, state: str) -> None:
        self.state = state
        self.state_elapsed = 0.0
        self.trajectory = None
        hammer_pos = self.hammer.data.root_pos_w[0].detach().cpu().tolist()
        print(
            f"[HAMMER-CUROBO]: state={state}, arm={self.arm}, "
            f"hammer=({hammer_pos[0]:.3f}, {hammer_pos[1]:.3f}, {hammer_pos[2]:.3f})",
            flush=True,
        )

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        self._transition("failed")

    def _task_frame_yaw(self) -> float:
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        return float(yaw[0].item()) - 0.5 * math.pi

    def _task_xy_to_world(self, task_x: float, task_y: float, task_yaw: float) -> tuple[float, float]:
        cos_yaw = math.cos(task_yaw)
        sin_yaw = math.sin(task_yaw)
        return (
            self.cfg.table_x + cos_yaw * task_x - sin_yaw * task_y,
            self.cfg.table_y + sin_yaw * task_x + cos_yaw * task_y,
        )

    def _task_quat(self, task_yaw: float) -> torch.Tensor:
        zero = torch.zeros(1, device=self.device)
        yaw = torch.tensor([task_yaw], device=self.device)
        return math_utils.quat_from_euler_xyz(zero, zero, yaw)[0]

    def _make_planner(self, arm: ArmName) -> MobileAlohaCuroboPlanner:
        table_pos_w = torch.tensor(
            [[
                self.cfg.table_x,
                self.cfg.table_y,
                self.cfg.table_height - 0.03 - self.cfg.planner_table_clearance,
            ]],
            device=self.device,
        )
        table_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        table_pos_b, table_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            table_pos_w,
            table_quat_w,
        )
        return MobileAlohaCuroboPlanner(
            arm,
            table_center_b=table_pos_b[0].detach().cpu().tolist(),
            table_quat_b=table_quat_b[0].detach().cpu().tolist(),
            table_dims=[self.cfg.table_length, self.cfg.table_width, 0.06],
            robot_config_stem=self.cfg.robot_config_stem,
            position_threshold=self.cfg.planner_position_threshold,
            rotation_threshold=self.cfg.planner_rotation_threshold,
            self_collision_check=self.cfg.planner_self_collision_check,
        )

    def _activate_arm(self, arm: ArmName) -> None:
        arm_joint_ids, arm_joint_names = self.robot.find_joints(
            [f"{arm}_joint[1-6]"], preserve_order=True
        )
        gripper_joint_ids, gripper_joint_names = self.robot.find_joints(
            [f"{arm}_joint[7-8]"], preserve_order=True
        )
        if len(arm_joint_ids) != 6 or len(gripper_joint_ids) != 2:
            raise RuntimeError(
                f"Could not resolve {arm} PiPER joints: "
                f"arm={arm_joint_names}, gripper={gripper_joint_names}"
            )

        if self.cfg.planner_enabled and arm not in self.planners:
            try:
                self.planners[arm] = self._make_planner(arm)
            except torch.OutOfMemoryError:
                print(
                    "[HAMMER-CUROBO]: planner cache exceeded GPU memory; "
                    "releasing the inactive arm and retrying.",
                    flush=True,
                )
                self.planners.clear()
                self.planner = None
                gc.collect()
                torch.cuda.empty_cache()
                self.planners[arm] = self._make_planner(arm)

        self.arm = arm
        self.arm_joint_ids = list(arm_joint_ids)
        self.arm_joint_names = list(arm_joint_names)
        self.gripper_joint_ids = list(gripper_joint_ids)
        self.gripper_joint_names = list(gripper_joint_names)
        self.planner = self.planners.get(arm)

    def _set_gripper(self, opening: float) -> None:
        if len(self.gripper_joint_ids) != 2:
            return
        opening = max(0.0, min(0.04, float(opening)))
        self.joint_targets[:, self.gripper_joint_ids[0]] = opening
        self.joint_targets[:, self.gripper_joint_ids[1]] = -opening

    def randomize_and_start(self, rng: random.Random) -> dict[str, float | str]:
        """Randomize the block exactly in RoboTwin's tabletop coordinate ranges."""

        if self.cfg.arm == "fl":
            block_x = -rng.uniform(0.05, max(0.05, abs(self.cfg.block_x_range[0])))
        elif self.cfg.arm == "fr":
            block_x = rng.uniform(0.05, max(0.05, self.cfg.block_x_range[1]))
        else:
            while True:
                block_x = rng.uniform(*self.cfg.block_x_range)
                block_y = rng.uniform(*self.cfg.block_y_range)
                if abs(block_x) >= 0.05 and block_x * block_x + block_y * block_y >= 0.001:
                    break
        if self.cfg.arm != "auto":
            block_y = rng.uniform(*self.cfg.block_y_range)

        arm: ArmName = "fl" if block_x < 0.0 else "fr"
        self._activate_arm(arm)
        task_yaw = self._task_frame_yaw()

        hammer_x, hammer_y = self._task_xy_to_world(
            self.cfg.hammer_table_x,
            self.cfg.hammer_table_y,
            task_yaw,
        )
        task_quat = self._task_quat(task_yaw)
        hammer_local_quat = torch.tensor(self.cfg.hammer_object_quat, device=self.device)
        hammer_quat = math_utils.quat_mul(task_quat.view(1, 4), hammer_local_quat.view(1, 4))[0]
        hammer_state = self.hammer.data.default_root_state.clone()
        hammer_state[:, :3] = torch.tensor(
            [hammer_x, hammer_y, self.cfg.table_height + self.cfg.hammer_root_height],
            device=hammer_state.device,
        )
        hammer_state[:, 3:7] = hammer_quat
        hammer_state[:, 7:] = 0.0
        self.hammer.write_root_state_to_sim(hammer_state)

        block_world_x, block_world_y = self._task_xy_to_world(block_x, block_y, task_yaw)
        block_yaw = task_yaw + rng.uniform(-self.cfg.block_yaw_limit, self.cfg.block_yaw_limit)
        block_quat = self._task_quat(block_yaw)
        block_state = self.block.data.default_root_state.clone()
        block_state[:, :3] = torch.tensor(
            [
                block_world_x,
                block_world_y,
                self.cfg.table_height + self.cfg.block_half_size,
            ],
            device=block_state.device,
        )
        block_state[:, 3:7] = block_quat
        block_state[:, 7:] = 0.0
        self.block.write_root_state_to_sim(block_state)

        self.joint_targets.copy_(self.robot.data.joint_pos)
        self._set_gripper(self.cfg.open_gripper)
        self.initial_hammer_z = float(hammer_state[0, 2].item())
        self.selected_grasp_quat_w = None
        self.ee_to_hammer_pos = None
        self.ee_to_hammer_quat = None
        self.strike_hammer_quat_w = None
        self.trajectory_base_pos_w = None
        self.trajectory_base_quat_w = None
        self.contact_seen = False
        self.current_contact_force = 0.0
        self.success_stable_steps = 0
        self.base_stable_steps = 0
        self.failure_reason = ""
        self._terminal_reported = False
        self.episode_elapsed = 0.0
        self._transition("settle")
        sample = {
            "arm": arm,
            "block_task_x": block_x,
            "block_task_y": block_y,
            "block_world_x": block_world_x,
            "block_world_y": block_world_y,
            "block_yaw": block_yaw,
        }
        print(f"[HAMMER-CUROBO]: randomized sample={sample}", flush=True)
        return sample

    def _current_arm_position(self) -> torch.Tensor:
        return self.robot.data.joint_pos[0, self.arm_joint_ids].detach()

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

    def _current_ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.planner is None:
            raise RuntimeError("The cuRobo planner has not been initialized.")
        pose_b = self.planner.forward_pose(self._current_arm_position())
        return self._base_pose_to_world(pose_b.position[0], pose_b.quaternion[0])

    def _refresh_table_obstacle(self) -> None:
        if self.planner is None:
            raise RuntimeError("The cuRobo planner has not been initialized.")
        table_pos_w = torch.tensor(
            [[
                self.cfg.table_x,
                self.cfg.table_y,
                self.cfg.table_height - 0.03 - self.cfg.planner_table_clearance,
            ]],
            device=self.device,
        )
        table_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        table_pos_b, table_quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            table_pos_w,
            table_quat_w,
        )
        self.planner.update_table(
            table_pos_b[0].detach().cpu().tolist(),
            table_quat_b[0].detach().cpu().tolist(),
            [self.cfg.table_length, self.cfg.table_width, 0.06],
        )

    def _capture_trajectory_base_pose(self) -> None:
        self.trajectory_base_pos_w = self.robot.data.root_pos_w[0].detach().clone()
        self.trajectory_base_quat_w = self.robot.data.root_quat_w[0].detach().clone()

    def _plan_goalset(
        self,
        positions_w: torch.Tensor,
        quaternions_w: torch.Tensor,
    ) -> bool:
        if self.planner is None:
            return False
        self._refresh_table_obstacle()
        count = positions_w.shape[0]
        root_pose = get_root_pose_w(self.robot.data).repeat(count, 1)
        positions_b, quaternions_b = math_utils.subtract_frame_transforms(
            root_pose[:, :3],
            root_pose[:, 3:7],
            positions_w,
            quaternions_w,
        )
        trajectory = self.planner.plan_to_goalset(
            self._current_arm_position(),
            positions_b.detach().cpu(),
            quaternions_b.detach().cpu(),
        )
        if trajectory is None:
            return False
        self.trajectory = trajectory
        self._capture_trajectory_base_pose()
        print(
            f"[HAMMER-CUROBO]: arm={self.arm}, candidate={trajectory.goalset_index}, "
            f"waypoints={len(trajectory.position)}, solve={trajectory.solve_time:.3f}s",
            flush=True,
        )
        return True

    def _plan_world_pose(self, position_w: torch.Tensor, quaternion_w: torch.Tensor) -> bool:
        return self._plan_goalset(position_w.view(1, 3), quaternion_w.view(1, 4))

    def _grasp_orientations_w(self) -> torch.Tensor:
        """Generate strictly top-down grasps with jaws crossing the hammer handle."""

        candidate_count = len(self.cfg.grasp_yaw_offsets_deg)
        hammer_quat = self.hammer.data.root_quat_w[0]
        handle_axis_w = math_utils.quat_apply(
            hammer_quat.view(1, 4),
            torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
        )[0]
        handle_axis_w[2] = 0.0
        if float(torch.linalg.vector_norm(handle_axis_w).item()) < 1e-4:
            task_yaw = self._task_frame_yaw()
            handle_axis_w = torch.tensor(
                [math.cos(task_yaw), math.sin(task_yaw), 0.0],
                device=self.device,
            )
        handle_axis_w = torch.nn.functional.normalize(handle_axis_w, dim=0)

        offsets = torch.deg2rad(
            torch.tensor(self.cfg.grasp_yaw_offsets_deg, device=self.device)
        )
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            candidate_count, 1
        )
        yaw_offsets = math_utils.quat_from_angle_axis(offsets, world_up)
        tool_y_axes_w = math_utils.quat_apply(
            yaw_offsets,
            handle_axis_w.repeat(candidate_count, 1),
        )
        tool_y_axes_w[:, 2] = 0.0
        tool_y_axes_w = torch.nn.functional.normalize(tool_y_axes_w, dim=-1)

        # PiPER's grasp-frame local Z is the approach direction. Exact world -Z
        # is outside this mounted arm's wrist workspace at the hammer pose, so
        # keep a small radial tilt while remaining a top-down grasp.
        radial_axis_w = self.hammer.data.root_pos_w[0] - self.robot.data.root_pos_w[0]
        radial_axis_w[2] = 0.0
        radial_axis_w = torch.nn.functional.normalize(radial_axis_w, dim=0)
        tilt = math.radians(self.cfg.grasp_approach_tilt_deg)
        approach_axes_w = (
            -math.cos(tilt) * world_up
            + math.sin(tilt) * radial_axis_w.repeat(candidate_count, 1)
        )
        approach_axes_w = torch.nn.functional.normalize(approach_axes_w, dim=-1)
        tool_y_axes_w = (
            tool_y_axes_w
            - torch.sum(tool_y_axes_w * approach_axes_w, dim=-1, keepdim=True)
            * approach_axes_w
        )
        tool_y_axes_w = torch.nn.functional.normalize(tool_y_axes_w, dim=-1)
        closing_axes_w = torch.linalg.cross(
            tool_y_axes_w,
            approach_axes_w,
            dim=-1,
        )
        closing_axes_w = torch.nn.functional.normalize(closing_axes_w, dim=-1)
        tool_y_axes_w = torch.linalg.cross(
            approach_axes_w,
            closing_axes_w,
            dim=-1,
        )
        rotation_matrices_w = torch.stack(
            (closing_axes_w, tool_y_axes_w, approach_axes_w),
            dim=-1,
        )
        return math_utils.quat_from_matrix(rotation_matrices_w)

    def _plan_pregrasp(self) -> bool:
        quaternions_w = self._grasp_orientations_w()
        approach_axes_w = math_utils.quat_apply(
            quaternions_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(quaternions_w), 1),
        )
        contact_positions_w = self.hammer.data.root_pos_w[0].repeat(len(quaternions_w), 1)
        pregrasp_positions_w = contact_positions_w - self.cfg.pregrasp_distance * approach_axes_w
        if not self._plan_goalset(pregrasp_positions_w, quaternions_w):
            return False
        if self.trajectory is None:
            return False
        self.selected_grasp_quat_w = quaternions_w[self.trajectory.goalset_index].detach().clone()
        selected_rotation = math_utils.matrix_from_quat(
            self.selected_grasp_quat_w.view(1, 4)
        )[0]
        approach_down_alignment = -float(selected_rotation[2, 2].item())
        selected_yaw = self.cfg.grasp_yaw_offsets_deg[self.trajectory.goalset_index]
        hammer_handle_axis = math_utils.quat_apply(
            self.hammer.data.root_quat_w[0].view(1, 4),
            torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
        )[0]
        hammer_handle_axis[2] = 0.0
        hammer_handle_axis = torch.nn.functional.normalize(
            hammer_handle_axis,
            dim=0,
        )
        jaw_handle_alignment = abs(
            float(torch.dot(selected_rotation[:, 0], hammer_handle_axis).item())
        )
        print(
            f"[HAMMER-CUROBO]: selected top-down grasp; "
            f"yaw_offset={selected_yaw:.1f} deg, "
            f"commanded_tilt={self.cfg.grasp_approach_tilt_deg:.1f} deg, "
            f"approach_down_alignment={approach_down_alignment:.4f}, "
            f"jaw_handle_alignment={jaw_handle_alignment:.4f}",
            flush=True,
        )
        return True

    def _plan_grasp(self) -> bool:
        if self.selected_grasp_quat_w is None:
            return False
        approach_axis = math_utils.quat_apply(
            self.selected_grasp_quat_w.view(1, 4),
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device),
        )[0]
        grasp_position = (
            self.hammer.data.root_pos_w[0].detach().clone()
            - self.cfg.grasp_distance * approach_axis
        )
        return self._plan_world_pose(grasp_position, self.selected_grasp_quat_w)

    def _plan_lift(self) -> bool:
        ee_pos_w, ee_quat_w = self._current_ee_pose_w()
        target_pos = ee_pos_w[0].detach().clone()
        target_pos[2] += self.cfg.lift_height
        return self._plan_world_pose(target_pos, ee_quat_w[0])

    def _capture_grasp_relation(self) -> None:
        ee_pos_w, ee_quat_w = self._current_ee_pose_w()
        hammer_pos_w = self.hammer.data.root_pos_w
        hammer_quat_w = self.hammer.data.root_quat_w
        self.ee_to_hammer_pos, self.ee_to_hammer_quat = math_utils.subtract_frame_transforms(
            ee_pos_w,
            ee_quat_w,
            hammer_pos_w,
            hammer_quat_w,
        )
        self.strike_hammer_quat_w = hammer_quat_w[0].detach().clone()

    def _block_top_point_w(self) -> torch.Tensor:
        local_top = torch.tensor(
            [[0.0, 0.0, self.cfg.block_half_size]],
            device=self.device,
        )
        return (
            self.block.data.root_pos_w
            + math_utils.quat_apply(self.block.data.root_quat_w, local_top)
        )[0]

    def _hammer_functional_point_w(self) -> torch.Tensor:
        local_point = torch.tensor(
            [self.cfg.hammer_functional_offset],
            device=self.device,
        )
        return (
            self.hammer.data.root_pos_w
            + math_utils.quat_apply(self.hammer.data.root_quat_w, local_point)
        )[0]

    def _desired_ee_for_functional_target(
        self,
        target_functional_point_w: torch.Tensor,
        hammer_quaternion_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.ee_to_hammer_pos is None
            or self.ee_to_hammer_quat is None
            or self.strike_hammer_quat_w is None
        ):
            raise RuntimeError("Hammer-to-gripper transform was not captured.")

        functional_local = torch.tensor(
            [self.cfg.hammer_functional_offset],
            device=self.device,
        )
        desired_hammer_quat = (
            self.strike_hammer_quat_w
            if hammer_quaternion_w is None
            else hammer_quaternion_w
        ).view(1, 4)
        desired_hammer_pos = (
            target_functional_point_w.view(1, 3)
            - math_utils.quat_apply(desired_hammer_quat, functional_local)
        )

        hammer_to_ee_quat = math_utils.quat_inv(self.ee_to_hammer_quat)
        hammer_to_ee_pos = math_utils.quat_apply(
            hammer_to_ee_quat,
            -self.ee_to_hammer_pos,
        )
        return math_utils.combine_frame_transforms(
            desired_hammer_pos,
            desired_hammer_quat,
            hammer_to_ee_pos,
            hammer_to_ee_quat,
        )

    def _plan_strike_pose(self, height: float) -> bool:
        target_point = self._block_top_point_w().detach().clone()
        target_point[2] += height
        if height > 0.0:
            if self.strike_hammer_quat_w is None:
                return False
            offsets = torch.deg2rad(
                torch.tensor(self.cfg.strike_yaw_offsets_deg, device=self.device)
            )
            world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
                len(offsets), 1
            )
            yaw_quaternions = math_utils.quat_from_angle_axis(offsets, world_z)
            hammer_quaternions = math_utils.quat_mul(
                yaw_quaternions,
                self.strike_hammer_quat_w.repeat(len(offsets), 1),
            )
            ee_positions = []
            ee_quaternions = []
            for hammer_quaternion in hammer_quaternions:
                ee_pos_w, ee_quat_w = self._desired_ee_for_functional_target(
                    target_point,
                    hammer_quaternion,
                )
                ee_positions.append(ee_pos_w[0])
                ee_quaternions.append(ee_quat_w[0])
            if not self._plan_goalset(
                torch.stack(ee_positions),
                torch.stack(ee_quaternions),
            ):
                return False
            if self.trajectory is None:
                return False
            self.strike_hammer_quat_w = hammer_quaternions[
                self.trajectory.goalset_index
            ].detach().clone()
            return True

        ee_pos_w, ee_quat_w = self._desired_ee_for_functional_target(target_point)
        return self._plan_world_pose(ee_pos_w[0], ee_quat_w[0])

    def _base_is_stable(self) -> bool:
        velocity = self.robot.data.root_link_vel_w[0]
        return (
            float(torch.linalg.vector_norm(velocity[:3]).item())
            <= self.cfg.base_settle_linear_velocity
            and float(torch.linalg.vector_norm(velocity[3:]).item())
            <= self.cfg.base_settle_angular_velocity
        )

    def _update_contact(self) -> None:
        if self.contact_sensor is None:
            self.current_contact_force = 0.0
            return
        force_matrix = getattr(self.contact_sensor.data, "force_matrix_w", None)
        if force_matrix is None or force_matrix.numel() == 0:
            self.current_contact_force = 0.0
            return
        self.current_contact_force = float(
            torch.linalg.vector_norm(force_matrix, dim=-1).max().item()
        )
        if self.current_contact_force >= self.cfg.contact_force_threshold:
            self.contact_seen = True

    def _execute_trajectory(self, *, allow_contact_completion: bool = False) -> bool:
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

        index = min(int(self.state_elapsed / self.trajectory.dt), len(self.trajectory.position) - 1)
        target = self.trajectory.position[index].to(self.joint_targets.device)
        self.joint_targets[:, self.arm_joint_ids] = target
        motion_time = len(self.trajectory.position) * self.trajectory.dt
        if self.state_elapsed < motion_time + self.cfg.trajectory_settle_time:
            return False

        actual = self.robot.data.joint_pos[0, self.arm_joint_ids]
        final_target = self.trajectory.position[-1].to(actual.device)
        tracking_error = float(torch.max(torch.abs(actual - final_target)).item())
        if tracking_error <= self.cfg.trajectory_tracking_tolerance:
            return True
        if allow_contact_completion and self.contact_seen:
            return True
        if self.state_elapsed >= motion_time + self.cfg.trajectory_tracking_timeout:
            self._fail(f"arm tracking error {tracking_error:.3f} rad")
        return False

    def _hammer_lifted(self) -> bool:
        return float(self.hammer.data.root_pos_w[0, 2].item()) >= self.initial_hammer_z + 0.045

    def _update_success(self) -> None:
        functional_point = self._hammer_functional_point_w()
        block_point = self._block_top_point_w()
        xy_error = torch.abs(functional_point[:2] - block_point[:2])
        functional_point_aligned = bool(
            torch.all(xy_error < self.cfg.success_xy_tolerance).item()
        )
        contact_ok = self.contact_seen if self.contact_sensor is not None else False
        success = functional_point_aligned and contact_ok
        self.success_stable_steps = self.success_stable_steps + 1 if success else 0

    def update(self, dt: float) -> None:
        self._update_contact()
        if self.is_terminal:
            self._update_success()
            return

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
                self._transition("plan_pregrasp")
            elif self.state_elapsed >= self.cfg.base_settle_timeout:
                self._fail("mobile base did not settle on terrain")
            return
        if self.state == "plan_pregrasp":
            if self._plan_pregrasp():
                self.state = "execute_pregrasp"
                self.state_elapsed = 0.0
            else:
                self._fail("no collision-free hammer pregrasp")
            return
        if self.state == "execute_pregrasp" and self._execute_trajectory():
            if self._plan_grasp():
                self.state = "execute_grasp"
                self.state_elapsed = 0.0
            else:
                self._fail("no collision-free hammer grasp")
            return
        if self.state == "execute_grasp" and self._execute_trajectory():
            self._set_gripper(self.cfg.closed_gripper)
            self._transition("close")
            return
        if self.state == "close":
            self._set_gripper(self.cfg.closed_gripper)
            if self.state_elapsed >= self.cfg.gripper_hold_time:
                if self._plan_lift():
                    self.state = "execute_lift"
                    self.state_elapsed = 0.0
                else:
                    self._fail("no collision-free hammer lift")
            return
        if self.state == "execute_lift" and self._execute_trajectory():
            if not self._hammer_lifted():
                self._fail("hammer was not lifted")
                return
            self._capture_grasp_relation()
            if self._plan_strike_pose(self.cfg.prestrike_height):
                self.state = "execute_prestrike"
                self.state_elapsed = 0.0
            else:
                self._fail("no collision-free prestrike")
            return
        if self.state == "execute_prestrike" and self._execute_trajectory():
            if self._plan_strike_pose(0.0):
                self.state = "execute_strike"
                self.state_elapsed = 0.0
            else:
                self._fail("no collision-free strike")
            return
        if self.state == "execute_strike" and self._execute_trajectory(
            allow_contact_completion=True
        ):
            self._transition("verify")
            return
        if self.state == "verify":
            self._set_gripper(self.cfg.closed_gripper)
            self._update_success()
            if self.success_stable_steps >= self.cfg.stable_success_steps:
                self._transition("done")
            elif self.state_elapsed >= self.cfg.strike_hold_time + 2.0:
                self._fail("hammer head did not contact the block target")

    def report_terminal_once(self) -> None:
        if not self.is_terminal or self._terminal_reported:
            return
        self._terminal_reported = True
        functional_point = self._hammer_functional_point_w().detach().cpu()
        block_point = self._block_top_point_w().detach().cpu()
        error = functional_point - block_point
        print(
            f"[HAMMER-CUROBO]: terminal={self.state}, success={self.succeeded}, "
            f"reason={self.failure_reason or 'success'}, arm={self.arm}, "
            f"functional_error=({error[0]:.4f}, {error[1]:.4f}, {error[2]:.4f}) m, "
            f"contact_force={self.current_contact_force:.3f} N, "
            f"elapsed={self.episode_elapsed:.2f}s",
            flush=True,
        )
