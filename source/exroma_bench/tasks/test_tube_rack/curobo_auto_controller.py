"""RoboTwin-style randomized test-tube placement with a cuRobo expert."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

import isaaclab.utils.math as math_utils

from exroma_bench.tasks.mobile_aloha_pick_place.curobo_auto_controller import (
    CuroboAutoPickPlaceConfig,
    CuroboAutoPickPlaceController,
)


@dataclass
class CuroboTestTubeRackConfig(CuroboAutoPickPlaceConfig):
    """Task geometry, planner motion, and strict insertion thresholds."""

    bottle_height: float = 0.14
    tube_radius: float = 0.018
    slot_xy_tolerance: float = 0.018
    upright_tolerance_rad: float = math.radians(12.0)
    bottom_height_tolerance: float = 0.018
    max_linear_velocity: float = 0.05
    max_angular_velocity: float = 0.20
    open_gripper_threshold: float = 0.025
    object_clearance: float = 0.005
    pregrasp_distance: float = 0.11
    grasp_approach_tilt_deg: float = 5.0
    lift_height: float = 0.14
    preplace_height: float = 0.16
    open_gripper: float = 0.040
    closed_gripper: float = 0.008
    gripper_hold_time: float = 1.0
    release_hold_time: float = 0.8
    retreat_distance: float = 0.10
    overall_timeout: float = 50.0
    stable_success_steps: int = 15


class CuroboTestTubeRackController(CuroboAutoPickPlaceController):
    """Pick an upright tube and insert it into the highlighted rack slot."""

    cfg: CuroboTestTubeRackConfig

    def _ee_pose_for_vertical_object(
        self,
        desired_object_position_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preserve the measured grasp transform while centering the tube."""

        ee_position_w, ee_quaternion_w = self._current_ee_pose_w()
        object_position_w = self.target_object.data.root_pos_w[0].view(1, 3)
        object_quaternion_w = self.target_object.data.root_quat_w[0].view(1, 4)
        object_position_ee, object_quaternion_ee = math_utils.subtract_frame_transforms(
            ee_position_w,
            ee_quaternion_w,
            object_position_w,
            object_quaternion_w,
        )
        ee_quaternion_object = math_utils.quat_inv(object_quaternion_ee)
        ee_position_object = math_utils.quat_apply(
            ee_quaternion_object,
            -object_position_ee,
        )
        desired_object_quaternion_w = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            device=self.device,
            dtype=desired_object_position_w.dtype,
        )
        desired_ee_position_w, desired_ee_quaternion_w = (
            math_utils.combine_frame_transforms(
                desired_object_position_w.view(1, 3),
                desired_object_quaternion_w,
                ee_position_object,
                ee_quaternion_object,
            )
        )
        return desired_ee_position_w[0], desired_ee_quaternion_w[0]

    def _plan_world_pose(
        self,
        position_w: torch.Tensor,
        quaternion_w: torch.Tensor,
    ) -> bool:
        """Straighten the tube for all rack-side transfer and insertion poses."""

        target_error = torch.linalg.vector_norm(
            position_w[:2]
            - torch.tensor(
                [self.cfg.target_x, self.cfg.target_y],
                device=position_w.device,
                dtype=position_w.dtype,
            )
        )
        if float(target_error.item()) <= 0.03:
            position_w, quaternion_w = self._ee_pose_for_vertical_object(position_w)
        return super()._plan_world_pose(position_w, quaternion_w)

    def randomize_and_start(self, rng: random.Random) -> tuple[float, float, float]:
        """Randomize an upright tube and reset the expert state machine."""

        x = self.cfg.table_x + rng.uniform(*self.cfg.spawn_x_range)
        y = self.cfg.table_y + rng.uniform(*self.cfg.spawn_y_range)
        z = self.cfg.table_height + self.cfg.object_clearance + 0.5 * self.cfg.bottle_height
        root_state = self.target_object.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor([x, y, z], device=root_state.device)
        root_state[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=root_state.device,
        )
        root_state[:, 7:] = 0.0
        self.target_object.write_root_state_to_sim(root_state)

        self.joint_targets.copy_(self.robot.data.joint_pos)
        self._set_gripper(self.cfg.open_gripper)
        self.initial_object_z = z
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
        self._transition("settle")
        print(
            f"[AUTO-CUROBO][TEST-TUBE]: randomized tube="
            f"({x:.3f}, {y:.3f}, {z:.3f})",
            flush=True,
        )
        return x, y, z

    def _update_success(self) -> None:
        """Require centered, upright, inserted, stationary, and released."""

        pos = self.target_object.data.root_pos_w[0]
        quat = self.target_object.data.root_quat_w[0].view(1, 4)
        lin_vel = self.target_object.data.root_lin_vel_w[0]
        ang_vel = self.target_object.data.root_ang_vel_w[0]
        radial = torch.linalg.vector_norm(
            pos[:2]
            - torch.tensor(
                [self.cfg.target_x, self.cfg.target_y],
                device=pos.device,
                dtype=pos.dtype,
            )
        )
        local_z = torch.tensor(
            [[0.0, 0.0, 1.0]],
            device=pos.device,
            dtype=pos.dtype,
        )
        tube_axis_w = math_utils.quat_apply(quat, local_z)[0]
        upright = abs(float(tube_axis_w[2].item())) >= math.cos(
            self.cfg.upright_tolerance_rad
        )
        tube_bottom_z = float(pos[2].item()) - 0.5 * self.cfg.bottle_height
        inserted = (
            abs(tube_bottom_z - self.cfg.table_height)
            <= self.cfg.bottom_height_tolerance
        )
        gripper_position = self.robot.data.joint_pos[0, self.gripper_joint_ids]
        released = (
            float(torch.min(torch.abs(gripper_position)).item())
            >= self.cfg.open_gripper_threshold
        )
        stable = (
            float(radial.item()) <= self.cfg.slot_xy_tolerance
            and upright
            and inserted
            and released
            and float(torch.linalg.vector_norm(lin_vel).item())
            <= self.cfg.max_linear_velocity
            and float(torch.linalg.vector_norm(ang_vel).item())
            <= self.cfg.max_angular_velocity
        )
        self.success_stable_steps = self.success_stable_steps + 1 if stable else 0

    def report_terminal_once(self) -> None:
        if not self.is_terminal or self._terminal_reported:
            return
        self._terminal_reported = True
        pos = self.target_object.data.root_pos_w[0].detach().cpu()
        quat = self.target_object.data.root_quat_w[0].view(1, 4)
        tube_axis_w = math_utils.quat_apply(
            quat,
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device),
        )[0]
        radial = math.hypot(
            float(pos[0]) - self.cfg.target_x,
            float(pos[1]) - self.cfg.target_y,
        )
        tilt_deg = math.degrees(
            math.acos(max(-1.0, min(1.0, abs(float(tube_axis_w[2].item())))))
        )
        print(
            f"[AUTO-CUROBO][TEST-TUBE]: terminal={self.state}, "
            f"success={self.succeeded}, reason={self.failure_reason or 'success'}, "
            f"tube=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
            f"slot_error={radial:.4f} m, tilt={tilt_deg:.2f} deg, "
            f"elapsed={self.episode_elapsed:.2f}s",
            flush=True,
        )
