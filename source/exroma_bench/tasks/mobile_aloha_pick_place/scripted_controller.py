"""Ground-truth scripted pick-and-place controller for Mobile ALOHA."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg


@dataclass
class ScriptedPickPlaceConfig:
    """Geometry and timing for the first scripted bottle pick-and-place expert."""

    arm: str = "fl"
    table_height: float = 0.8
    target_x: float = 1.0
    target_y: float = -0.18
    target_radius: float = 0.08
    wrist_standoff: float = 0.10
    pregrasp_standoff: float = 0.18
    wrist_z_offset: float = 0.025
    lift_height: float = 0.20
    open_gripper: float = 0.035
    closed_gripper: float = 0.002
    position_tolerance: float = 0.025
    move_timeout: float = 4.0
    gripper_hold_time: float = 0.8
    stable_success_steps: int = 10
    desired_base_to_object_x: float = 0.62
    base_position_tolerance: float = 0.02
    base_speed: float = 0.12


class ScriptedPickPlaceController:
    """Execute a pose-reactive DLS-IK sequence using the simulated object pose."""

    _MOVE_STATES = {"pregrasp", "approach", "lift", "transfer", "lower", "retreat"}

    def __init__(self, robot, joint_targets: torch.Tensor, target_object, cfg: ScriptedPickPlaceConfig) -> None:
        if cfg.arm not in {"fl", "fr"}:
            raise ValueError(f"Unsupported arm: {cfg.arm}")

        self.robot = robot
        self.joint_targets = joint_targets
        self.target_object = target_object
        self.cfg = cfg
        self.device = robot.device
        self.num_envs = robot.data.joint_pos.shape[0]

        self.arm_joint_ids, self.arm_joint_names = robot.find_joints(
            [f"{cfg.arm}_joint[1-6]"],
            preserve_order=True,
        )
        self.gripper_joint_ids, self.gripper_joint_names = robot.find_joints(
            [f"{cfg.arm}_joint[7-8]"],
            preserve_order=True,
        )
        self.ee_body_ids, self.ee_body_names = robot.find_bodies(f"{cfg.arm}_link6", preserve_order=True)
        self.finger_body_ids, self.finger_body_names = robot.find_bodies(
            [f"{cfg.arm}_link7", f"{cfg.arm}_link8"],
            preserve_order=True,
        )
        if len(self.arm_joint_ids) != 6 or len(self.gripper_joint_ids) != 2 or len(self.ee_body_ids) != 1:
            raise RuntimeError(
                "Could not resolve Mobile ALOHA scripted-control entities: "
                f"arm={self.arm_joint_names}, gripper={self.gripper_joint_names}, ee={self.ee_body_names}"
            )

        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)
        self.ee_body_id = self.ee_body_ids[0]
        self.ee_jacobian_id = self.ee_body_id - 1 if robot.is_fixed_base else self.ee_body_id

        self.state = "idle"
        self.state_elapsed = 0.0
        self.total_elapsed = 0.0
        self.success_stable_steps = 0
        self.initial_object_pos = target_object.data.root_pos_w.clone()
        self.grasp_object_pos = target_object.data.root_pos_w.clone()
        self.fixed_ee_quat_b: torch.Tensor | None = None
        self.last_position_error = float("inf")
        self._reported_terminal = False
        self._last_align_report_second = -1

    @property
    def is_terminal(self) -> bool:
        return self.state in {"done", "failed"}

    @property
    def succeeded(self) -> bool:
        return self.state == "done" and self.success_stable_steps >= self.cfg.stable_success_steps

    def start(self) -> None:
        self.ik.reset()
        self.initial_object_pos = self.target_object.data.root_pos_w.clone()
        self.grasp_object_pos = self.initial_object_pos.clone()
        self.success_stable_steps = 0
        self.total_elapsed = 0.0
        self._reported_terminal = False

        root_pose_w = self.robot.data.root_pose_w
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]
        _, self.fixed_ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )
        self._set_gripper(self.cfg.open_gripper)
        self._transition("align_base")

    def base_command(self) -> tuple[float, float]:
        """Return the automatic linear/yaw command for the mobile base."""

        if self.state != "align_base":
            return 0.0, 0.0
        root_x = float(self.robot.data.root_pos_w[0, 0].item())
        object_x = float(self.target_object.data.root_pos_w[0, 0].item())
        desired_root_x = object_x - self.cfg.desired_base_to_object_x
        error = desired_root_x - root_x
        linear = max(-self.cfg.base_speed, min(self.cfg.base_speed, error))
        return linear, 0.0

    def _transition(self, state: str) -> None:
        self.state = state
        self.state_elapsed = 0.0
        if state == "close":
            self.grasp_object_pos = self.target_object.data.root_pos_w.clone()
        object_pos = self.target_object.data.root_pos_w[0].detach().cpu().tolist()
        ee_pos = self.robot.data.body_pos_w[0, self.ee_body_id].detach().cpu().tolist()
        finger_text = ""
        if len(self.finger_body_ids) == 2:
            finger_positions = self.robot.data.body_pos_w[0, self.finger_body_ids].detach().cpu().tolist()
            finger_text = (
                f", fingers=({finger_positions[0][0]:.3f},{finger_positions[0][1]:.3f},{finger_positions[0][2]:.3f})"
                f"/({finger_positions[1][0]:.3f},{finger_positions[1][1]:.3f},{finger_positions[1][2]:.3f})"
            )
        print(
            f"[AUTO]: state={state}, object=({object_pos[0]:.3f},{object_pos[1]:.3f},{object_pos[2]:.3f}), "
            f"ee=({ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}){finger_text}",
            flush=True,
        )

    def _set_gripper(self, opening: float) -> None:
        opening = max(0.0, min(0.04, opening))
        self.joint_targets[:, self.gripper_joint_ids[0]] = opening
        self.joint_targets[:, self.gripper_joint_ids[1]] = -opening

    def _target_world_position(self) -> torch.Tensor:
        object_pos = self.target_object.data.root_pos_w
        target = object_pos.clone()

        if self.state == "pregrasp":
            target[:, 0] -= self.cfg.pregrasp_standoff
            target[:, 2] += self.cfg.wrist_z_offset
        elif self.state in {"approach", "close"}:
            target[:, 0] -= self.cfg.wrist_standoff
            target[:, 2] += self.cfg.wrist_z_offset
        elif self.state == "lift":
            target = self.grasp_object_pos.clone()
            target[:, 0] -= self.cfg.wrist_standoff
            target[:, 2] += self.cfg.lift_height
        elif self.state == "transfer":
            target[:, 0] = self.cfg.target_x - self.cfg.wrist_standoff
            target[:, 1] = self.cfg.target_y
            target[:, 2] = self.cfg.table_height + 0.30
        elif self.state in {"lower", "open"}:
            target[:, 0] = self.cfg.target_x - self.cfg.wrist_standoff
            target[:, 1] = self.cfg.target_y
            target[:, 2] = self.cfg.table_height + 0.12
        elif self.state == "retreat":
            target[:, 0] = self.cfg.target_x - self.cfg.wrist_standoff
            target[:, 1] = self.cfg.target_y
            target[:, 2] = self.cfg.table_height + 0.30
        return target

    def _apply_ik_target(self, target_pos_w: torch.Tensor) -> None:
        if self.fixed_ee_quat_b is None:
            return

        root_pose_w = self.robot.data.root_pose_w
        target_quat_w = math_utils.quat_mul(root_pose_w[:, 3:7], self.fixed_ee_quat_b)
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            target_pos_w,
            target_quat_w,
        )
        command = torch.cat((target_pos_b, target_quat_b), dim=-1)
        self.ik.set_command(command)

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobian_id, :, self.arm_joint_ids
        ]
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        joint_pos_des = self.ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :]
        joint_pos_des = torch.clamp(joint_pos_des, limits[:, :, 0], limits[:, :, 1])
        self.joint_targets[:, self.arm_joint_ids] = joint_pos_des
        self.last_position_error = float(torch.linalg.vector_norm(target_pos_w - ee_pose_w[:, :3], dim=-1).max().item())

    def _move_finished(self) -> bool:
        return self.last_position_error <= self.cfg.position_tolerance or self.state_elapsed >= self.cfg.move_timeout

    def _update_success(self) -> None:
        object_pos = self.target_object.data.root_pos_w
        object_vel = self.target_object.data.root_lin_vel_w
        dx = object_pos[:, 0] - self.cfg.target_x
        dy = object_pos[:, 1] - self.cfg.target_y
        in_zone = torch.sqrt(dx.square() + dy.square()) <= self.cfg.target_radius
        on_table = torch.abs(object_pos[:, 2] - (self.cfg.table_height + 0.09)) <= 0.08
        stable = torch.linalg.vector_norm(object_vel, dim=-1) <= 0.05
        if bool(torch.all(in_zone & on_table & stable).item()):
            self.success_stable_steps += 1
        else:
            self.success_stable_steps = 0

    def update(self, dt: float) -> None:
        if self.state == "idle":
            self.start()
        if self.is_terminal:
            self.state_elapsed += dt
            self._update_success()
            return

        self.state_elapsed += dt
        self.total_elapsed += dt
        if self.state == "align_base":
            root_x = float(self.robot.data.root_pos_w[0, 0].item())
            object_x = float(self.target_object.data.root_pos_w[0, 0].item())
            desired_root_x = object_x - self.cfg.desired_base_to_object_x
            report_second = int(self.state_elapsed)
            if report_second != self._last_align_report_second:
                self._last_align_report_second = report_second
                print(
                    f"[AUTO]: align root_x={root_x:.3f}, desired_x={desired_root_x:.3f}, "
                    f"linear={self.base_command()[0]:.3f}",
                    flush=True,
                )
            if abs(desired_root_x - root_x) <= self.cfg.base_position_tolerance:
                self._transition("pregrasp")
            return

        target_pos_w = self._target_world_position()
        self._apply_ik_target(target_pos_w)

        if self.state == "pregrasp" and self._move_finished():
            self._transition("approach")
        elif self.state == "approach" and self._move_finished():
            self._transition("close")
        elif self.state == "close":
            self._set_gripper(self.cfg.closed_gripper)
            if self.state_elapsed >= self.cfg.gripper_hold_time:
                self._transition("lift")
        elif self.state == "lift" and self._move_finished():
            lifted = self.target_object.data.root_pos_w[:, 2] > self.initial_object_pos[:, 2] + 0.06
            if not bool(torch.all(lifted).item()):
                self._transition("failed")
            else:
                self._transition("transfer")
        elif self.state == "transfer" and self._move_finished():
            self._transition("lower")
        elif self.state == "lower" and self._move_finished():
            self._transition("open")
        elif self.state == "open":
            self._set_gripper(self.cfg.open_gripper)
            if self.state_elapsed >= self.cfg.gripper_hold_time:
                self._transition("retreat")
        elif self.state == "retreat" and self._move_finished():
            self._transition("done")

        self._update_success()

    def report_terminal_once(self) -> None:
        if not self.is_terminal or self._reported_terminal:
            return
        if self.state == "done" and not self.succeeded and self.state_elapsed < 2.0:
            return
        self._reported_terminal = True
        object_pos = self.target_object.data.root_pos_w[0].detach().cpu().tolist()
        print(
            f"[AUTO]: terminal={self.state}, success={self.succeeded}, "
            f"object_pos=({object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}), "
            f"elapsed={self.total_elapsed:.2f}s",
            flush=True,
        )
