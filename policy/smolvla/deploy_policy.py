"""Translate ExRoMa observations into LeRobot/SmolVLA feature keys."""

from __future__ import annotations

from typing import Any

import numpy as np
from exroma_bench.policy.backend import (
    create_backend,
    invoke_backend,
    normalize_backend_result,
    reset_backend,
)
from exroma_bench.policy.protocol import ACTION_DIM, STATE_DIM

from .lerobot_backend import LeRobotSmolVLABackend


def encode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Build the flat LeRobot observation used by an ExRoMa-trained SmolVLA."""

    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"SmolVLA expected state[{STATE_DIM}], got {state.shape}.")
    images = observation["images"]
    return {
        "observation.state": np.ascontiguousarray(state),
        "observation.images.cam_high": np.ascontiguousarray(images["cam_high"], dtype=np.uint8),
        "observation.images.cam_left_wrist": np.ascontiguousarray(
            images["cam_left_wrist"], dtype=np.uint8
        ),
        "observation.images.cam_right_wrist": np.ascontiguousarray(
            images["cam_right_wrist"], dtype=np.uint8
        ),
        "task": str(observation.get("prompt", "")),
        "robot_type": "dual_piper_rover",
        "timestamp": float(observation.get("timestamp", 0.0)),
    }


class SmolVLAPolicyAdapter:
    """Serve action chunks from a native LeRobot or custom SmolVLA backend."""

    name = "smolvla"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        if int(self.config.get("state_dim", STATE_DIM)) != STATE_DIM:
            raise ValueError(f"SmolVLA state_dim must be {STATE_DIM}.")
        if int(self.config.get("action_dim", ACTION_DIM)) != ACTION_DIM:
            raise ValueError(f"SmolVLA action_dim must be {ACTION_DIM}.")
        self.action_steps = int(self.config.get("action_steps", 50))
        if self.action_steps <= 0:
            raise ValueError("SmolVLA action_steps must be positive.")
        if self.config.get("backend_factory"):
            self.backend = create_backend(self.config)
        else:
            self.backend = LeRobotSmolVLABackend(self.config)

    def infer(self, observation: dict[str, Any]) -> Any:
        encoded = encode_observation(observation)
        result = invoke_backend(self.backend, encoded)
        return normalize_backend_result(result, max_actions=self.action_steps)

    def reset(self) -> None:
        reset_backend(self.backend)


def create_policy(config: dict[str, Any]) -> SmolVLAPolicyAdapter:
    return SmolVLAPolicyAdapter(config)
