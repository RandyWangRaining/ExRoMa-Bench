"""Normalized skid-steer control for the six-wheel Pragyan model."""

from __future__ import annotations

import torch

from exroma_bench.robots.pragyan_cfg import (
    PRAGYAN_ANGULAR_SCALE,
    PRAGYAN_LINEAR_SCALE,
    PRAGYAN_TRACK_WIDTH,
    PRAGYAN_WHEEL_RADIUS,
)


class PragyanDrive:
    def __init__(self, rover) -> None:
        self.rover = rover
        joint_ids, joint_names = rover.find_joints(
            ["wheel_drive_joint_.*"], preserve_order=True
        )
        if len(joint_ids) != 6:
            raise RuntimeError(
                f"Expected six Pragyan drive joints, got {joint_names}"
            )
        self.joint_ids = list(joint_ids)
        self.left_indices = [i for i, name in enumerate(joint_names) if "_l" in name]
        self.right_indices = [i for i, name in enumerate(joint_names) if "_r" in name]
        if len(self.left_indices) != 3 or len(self.right_indices) != 3:
            raise RuntimeError(f"Cannot split Pragyan wheels by side: {joint_names}")

    def apply(self, linear_command: float, angular_command: float) -> None:
        linear = max(-1.0, min(1.0, float(linear_command))) * PRAGYAN_LINEAR_SCALE
        angular = max(-1.0, min(1.0, float(angular_command))) * PRAGYAN_ANGULAR_SCALE
        # Pragyan's wheel joint axes are mirrored relative to the common
        # differential-drive sign convention.
        left = (linear + 0.5 * PRAGYAN_TRACK_WIDTH * angular) / PRAGYAN_WHEEL_RADIUS
        right = (linear - 0.5 * PRAGYAN_TRACK_WIDTH * angular) / PRAGYAN_WHEEL_RADIUS
        target = torch.zeros(
            (self.rover.num_instances, len(self.joint_ids)),
            device=self.rover.device,
        )
        target[:, self.left_indices] = left
        target[:, self.right_indices] = right
        self.rover.set_joint_velocity_target(target, joint_ids=self.joint_ids)

    def stop(self) -> None:
        self.apply(0.0, 0.0)
