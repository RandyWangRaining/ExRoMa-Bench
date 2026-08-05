"""Deterministic adapters for RoboTwin 1.0 task randomization rules."""

from __future__ import annotations

import math
import random
from typing import NamedTuple


class PlanarPose(NamedTuple):
    x: float
    y: float
    yaw: float


def _sample_stack_block(
    rng: random.Random,
    *,
    other: PlanarPose | None = None,
) -> PlanarPose:
    """Sample one block using RoboTwin 1.0 ``blocks_stack_easy`` limits."""

    while True:
        x = rng.uniform(-0.25, 0.25)
        y = rng.uniform(-0.15, 0.05)
        if abs(x) < 0.15 and y > 0.0:
            y = rng.uniform(-0.15, 0.0)
        yaw = rng.uniform(-1.57, 1.57)

        if abs(x) < 0.05:
            continue
        if math.hypot(x, y + 0.10) < 0.15:
            continue
        if other is not None and math.hypot(x - other.x, y - other.y) < 0.10:
            continue
        return PlanarPose(x, y, yaw)


def sample_blocks_stack_easy(rng: random.Random) -> dict[str, PlanarPose]:
    """Return red/black local poses with RoboTwin's rejection constraints."""

    red = _sample_stack_block(rng)
    black = _sample_stack_block(rng, other=red)
    return {"red": red, "black": black}
