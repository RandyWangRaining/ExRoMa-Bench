"""Runtime safety envelope applied before real-robot commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interfaces import RobotAction


@dataclass(frozen=True)
class SafetyLimits:
    joint_lower: np.ndarray
    joint_upper: np.ndarray
    max_joint_step: float
    max_base_linear: float
    max_base_angular: float


def clamp_action(
    action: RobotAction,
    current_qpos: np.ndarray,
    limits: SafetyLimits,
) -> RobotAction:
    target = np.asarray(action.joint_position_target, dtype=np.float32)
    if target.shape != (14,):
        raise ValueError(f"Expected 14 joint targets, got {target.shape}")
    delta = np.clip(
        target - current_qpos,
        -limits.max_joint_step,
        limits.max_joint_step,
    )
    target = np.clip(current_qpos + delta, limits.joint_lower, limits.joint_upper)
    base = np.asarray(action.base_velocity, dtype=np.float32)
    if base.shape != (2,):
        raise ValueError(f"Expected two base commands, got {base.shape}")
    base = np.array(
        [
            np.clip(base[0], -limits.max_base_linear, limits.max_base_linear),
            np.clip(base[1], -limits.max_base_angular, limits.max_base_angular),
        ],
        dtype=np.float32,
    )
    return RobotAction(target, base)
