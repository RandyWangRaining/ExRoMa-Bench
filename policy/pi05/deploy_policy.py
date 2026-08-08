"""Translate the ExRoMa wire observation into an OpenPI-style input."""

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


def encode_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Produce image-list and state fields used by RoboTwin-style pi0.5 adapters."""

    images = observation["images"]
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"pi05 expected state[{STATE_DIM}], got {state.shape}.")
    image_list = [
        np.ascontiguousarray(images["cam_high"], dtype=np.uint8),
        np.ascontiguousarray(images["cam_right_wrist"], dtype=np.uint8),
        np.ascontiguousarray(images["cam_left_wrist"], dtype=np.uint8),
    ]
    return {
        "input_rgb_arr": image_list,
        "input_state": state,
        "prompt": observation.get("prompt", ""),
        "timestamp": observation.get("timestamp", 0.0),
    }


class Pi05PolicyAdapter:
    """Lifecycle wrapper around an OpenPI/pi0.5 backend."""

    name = "pi05"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        if int(self.config.get("state_dim", STATE_DIM)) != STATE_DIM:
            raise ValueError(f"pi05 state_dim must be {STATE_DIM}.")
        if int(self.config.get("action_dim", ACTION_DIM)) != ACTION_DIM:
            raise ValueError(f"pi05 action_dim must be {ACTION_DIM}.")
        self.action_steps = int(self.config.get("action_steps", 50))
        if self.action_steps <= 0:
            raise ValueError("pi05 action_steps must be positive.")
        self.backend = create_backend(self.config)

    def infer(self, observation: dict[str, Any]) -> Any:
        encoded = encode_observation(observation)
        update_window = getattr(self.backend, "update_observation_window", None)
        get_action = getattr(self.backend, "get_action", None)
        if callable(update_window) and callable(get_action):
            set_language = getattr(self.backend, "set_language", None)
            if callable(set_language) and getattr(self.backend, "observation_window", None) is None:
                set_language(encoded["prompt"])
            update_window(encoded["input_rgb_arr"], encoded["input_state"])
            result = get_action()
        else:
            result = invoke_backend(self.backend, encoded)
        return normalize_backend_result(result, max_actions=self.action_steps)

    def reset(self) -> None:
        reset_backend(self.backend)


def create_policy(config: dict[str, Any]) -> Pi05PolicyAdapter:
    return Pi05PolicyAdapter(config)
