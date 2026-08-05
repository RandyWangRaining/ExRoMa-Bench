"""Shared policy and hardware contracts for simulation-to-real deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RobotObservation:
    qpos: np.ndarray
    qvel: np.ndarray
    base_pose: np.ndarray
    base_velocity: np.ndarray
    images: dict[str, np.ndarray]
    timestamp: float


@dataclass(frozen=True)
class RobotAction:
    joint_position_target: np.ndarray
    base_velocity: np.ndarray


class HardwareAdapter(Protocol):
    def connect(self) -> None: ...

    def read(self) -> RobotObservation: ...

    def write(self, action: RobotAction) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class Policy(Protocol):
    def reset(self) -> None: ...

    def predict(self, observation: RobotObservation) -> RobotAction: ...
