"""Named policy adapter for replaying one recorded ExRoMa episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exroma_bench.policy.server import Hdf5ReplayPolicy


class ReplayPolicy(Hdf5ReplayPolicy):
    """Return recorded 16-D actions in fixed-size prediction chunks."""

    name = "replay"

    def __init__(self, episode_path: str | Path, *, action_horizon: int = 8) -> None:
        super().__init__(episode_path, action_horizon=action_horizon)
        self.name = "replay"


def create_policy(config: dict[str, Any]) -> ReplayPolicy:
    episode_path = config.get("episode_path")
    if not episode_path:
        raise ValueError(
            "replay requires episode_path. Pass --replay-episode or set it in deploy_policy.yml."
        )
    return ReplayPolicy(
        episode_path,
        action_horizon=int(config.get("action_horizon", 8)),
    )
