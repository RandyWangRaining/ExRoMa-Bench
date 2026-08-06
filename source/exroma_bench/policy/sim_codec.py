"""Map Isaac Lab dual-PiPER state and controls to the compact policy contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import ACTION_DIM, CAMERA_KEYS, STATE_DIM, validate_actions


def body_planar_velocity_wxyz(base_pose: Any, root_velocity_world: Any) -> np.ndarray:
    """Project world linear/angular velocity onto rover forward and yaw axes."""

    pose = np.asarray(base_pose, dtype=np.float64)
    velocity = np.asarray(root_velocity_world, dtype=np.float64)
    if pose.shape != (7,) or velocity.shape != (6,):
        raise ValueError(
            f"Expected base pose[7] and velocity[6], got {pose.shape}, {velocity.shape}."
        )
    quaternion = pose[3:7]
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError("Base pose contains a zero-length quaternion.")
    w, x, y, z = quaternion / norm
    linear = velocity[:3]
    angular = velocity[3:]
    forward = (
        (1.0 - 2.0 * (y * y + z * z)) * linear[0]
        + 2.0 * (x * y + z * w) * linear[1]
        + 2.0 * (x * z - y * w) * linear[2]
    )
    yaw = (
        2.0 * (x * z + y * w) * angular[0]
        + 2.0 * (y * z - x * w) * angular[1]
        + (1.0 - 2.0 * (x * x + y * y)) * angular[2]
    )
    return np.asarray([forward, yaw], dtype=np.float32)


class DualPiperPolicyCodec:
    """Encode 16-D observations and apply validated 16-D policy actions."""

    def __init__(self, robot, rover) -> None:
        self.robot = robot
        self.rover = rover
        self.left_arm_ids = self._joint_ids("fl_joint[1-6]", expected=6)
        self.left_gripper_ids = self._joint_ids("fl_joint[7-8]", expected=2)
        self.right_arm_ids = self._joint_ids("fr_joint[1-6]", expected=6)
        self.right_gripper_ids = self._joint_ids("fr_joint[7-8]", expected=2)
        mast_ids, _ = robot.find_joints(["camera_stand_.*_joint"], preserve_order=True)
        self.mast_ids = list(mast_ids)

    def _joint_ids(self, expression: str, *, expected: int) -> list[int]:
        ids, names = self.robot.find_joints([expression], preserve_order=True)
        if len(ids) != expected:
            raise RuntimeError(f"Expected {expected} joints for {expression}, got {names}.")
        return list(ids)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def state(self) -> np.ndarray:
        qpos = self._numpy(self.robot.data.joint_pos[0]).astype(np.float32, copy=False)
        compact = np.concatenate(
            (
                qpos[self.left_arm_ids],
                [0.5 * (qpos[self.left_gripper_ids[0]] - qpos[self.left_gripper_ids[1]])],
                qpos[self.right_arm_ids],
                [0.5 * (qpos[self.right_gripper_ids[0]] - qpos[self.right_gripper_ids[1]])],
            )
        ).astype(np.float32)
        root_pos = self._numpy(self.rover.data.root_pos_w[0])
        root_quat = self._numpy(self.rover.data.root_quat_w[0])
        root_velocity = self._numpy(self.rover.data.root_link_vel_w[0])
        planar = body_planar_velocity_wxyz(
            np.concatenate((root_pos, root_quat)),
            root_velocity,
        )
        state = np.concatenate((compact, planar)).astype(np.float32)
        if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"Invalid compact policy state: shape={state.shape}.")
        return state

    def images(self, cameras: dict[str, Any]) -> dict[str, np.ndarray]:
        if set(cameras) != set(CAMERA_KEYS):
            raise ValueError(f"Expected cameras {list(CAMERA_KEYS)}, got {sorted(cameras)}.")
        images = {}
        for key in CAMERA_KEYS:
            value = cameras[key].data.output.get("rgb")
            if value is None:
                raise RuntimeError(f"Camera {key} has no RGB output.")
            image = self._numpy(value)
            if image.ndim == 4:
                image = image[0]
            images[key] = np.ascontiguousarray(image[..., :3])
        return images

    def hold_action(self) -> np.ndarray:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[:14] = self.state()[:14]
        return action

    def apply_action(self, action: Any, joint_targets) -> tuple[float, float]:
        import torch

        compact = validate_actions(action)
        if len(compact) != 1:
            raise ValueError("apply_action expects one action, not an action chunk.")
        values = torch.as_tensor(
            compact[0],
            dtype=joint_targets.dtype,
            device=joint_targets.device,
        )
        joint_targets[0, self.left_arm_ids] = values[:6]
        joint_targets[0, self.left_gripper_ids[0]] = values[6]
        joint_targets[0, self.left_gripper_ids[1]] = -values[6]
        joint_targets[0, self.right_arm_ids] = values[7:13]
        joint_targets[0, self.right_gripper_ids[0]] = values[13]
        joint_targets[0, self.right_gripper_ids[1]] = -values[13]

        controlled_ids = [
            *self.left_arm_ids,
            *self.left_gripper_ids,
            *self.right_arm_ids,
            *self.right_gripper_ids,
        ]
        limits = self.robot.data.soft_joint_pos_limits[0, controlled_ids]
        joint_targets[0, controlled_ids] = torch.clamp(
            joint_targets[0, controlled_ids],
            min=limits[:, 0],
            max=limits[:, 1],
        )
        if self.mast_ids:
            joint_targets[:, self.mast_ids] = self.robot.data.default_joint_pos[:, self.mast_ids]
        return (
            float(np.clip(compact[0, 14], -1.0, 1.0)),
            float(np.clip(compact[0, 15], -1.0, 1.0)),
        )


class PolicyTaskMonitor:
    """Evaluate task geometry without running the cuRobo expert state machine."""

    def __init__(self, controller, task_name: str, *, max_steps: int) -> None:
        self.controller = controller
        self.task_name = task_name
        self.max_steps = int(max_steps)
        self.steps = 0
        self.elapsed = 0.0
        self.stable_steps = 0
        self.succeeded = False
        self.failure_reason = ""

    @property
    def is_terminal(self) -> bool:
        return self.succeeded or bool(self.failure_reason)

    def reset(self) -> None:
        self.steps = 0
        self.elapsed = 0.0
        self.stable_steps = 0
        self.succeeded = False
        self.failure_reason = ""
        if hasattr(self.controller, "success_stable_steps"):
            self.controller.success_stable_steps = 0

    def update(self, dt: float) -> None:
        if self.is_terminal:
            return
        self.steps += 1
        self.elapsed += float(dt)
        if callable(getattr(self.controller, "final_success", None)):
            success = bool(self.controller.final_success())
            self.stable_steps = self.stable_steps + 1 if success else 0
            self.succeeded = self.stable_steps >= 12
        elif self.task_name == "test_tube_rack":
            self.controller._update_success()
            self.succeeded = (
                self.controller.success_stable_steps >= self.controller.cfg.stable_success_steps
            )
        elif self.task_name == "beat_block_hammer":
            self.controller._update_contact()
            self.controller._update_success()
            self.succeeded = (
                self.controller.success_stable_steps >= self.controller.cfg.stable_success_steps
            )
        else:
            raise RuntimeError(f"No learned-policy success monitor for task {self.task_name!r}.")
        if not self.succeeded and self.steps >= self.max_steps:
            self.failure_reason = f"policy episode timeout after {self.max_steps} simulation steps"
