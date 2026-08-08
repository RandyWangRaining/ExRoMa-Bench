"""Translate the ExRoMa wire observation into the conventional ACT input."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from exroma_bench.policy.backend import (
    create_backend,
    invoke_backend,
    normalize_backend_result,
    reset_backend,
)
from exroma_bench.policy.protocol import ACTION_DIM, STATE_DIM


def _chw_float(image: Any, *, width: int | None, height: int | None) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    if width is not None and height is not None and rgb.shape[:2] != (height, width):
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(np.moveaxis(rgb, -1, 0), dtype=np.float32) / 255.0


def encode_observation(
    observation: dict[str, Any],
    *,
    image_width: int | None = 640,
    image_height: int | None = 480,
) -> dict[str, Any]:
    """Produce the image and qpos keys used by common ACT implementations."""

    images = observation["images"]
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"ACT expected state[{STATE_DIM}], got {state.shape}.")
    return {
        "head_cam": _chw_float(images["cam_high"], width=image_width, height=image_height),
        "left_cam": _chw_float(images["cam_left_wrist"], width=image_width, height=image_height),
        "right_cam": _chw_float(images["cam_right_wrist"], width=image_width, height=image_height),
        "qpos": state,
        "prompt": observation.get("prompt", ""),
        "timestamp": observation.get("timestamp", 0.0),
    }


class ACTPolicyAdapter:
    """Lifecycle wrapper around an ACT backend supplied by the policy environment."""

    name = "ACT"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        if int(self.config.get("state_dim", STATE_DIM)) != STATE_DIM:
            raise ValueError(f"ACT state_dim must be {STATE_DIM}.")
        if int(self.config.get("action_dim", ACTION_DIM)) != ACTION_DIM:
            raise ValueError(f"ACT action_dim must be {ACTION_DIM}.")
        self.image_width = self.config.get("image_width", 640)
        self.image_height = self.config.get("image_height", 480)
        self.backend = create_backend(self.config)

    def infer(self, observation: dict[str, Any]) -> Any:
        encoded = encode_observation(
            observation,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        return normalize_backend_result(invoke_backend(self.backend, encoded))

    def reset(self) -> None:
        reset_backend(self.backend)


def create_policy(config: dict[str, Any]) -> ACTPolicyAdapter:
    return ACTPolicyAdapter(config)
