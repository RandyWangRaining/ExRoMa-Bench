"""Native OpenPI checkpoint backend for ExRoMa policy serving."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .openpi_exroma import DEFAULT_DATASETS, make_train_config

LOGGER = logging.getLogger(__name__)


def _checkpoint_repo_ids(checkpoint: Path) -> tuple[str, ...]:
    assets_dir = checkpoint / "assets"
    asset_ids = sorted(path.name for path in assets_dir.iterdir() if path.is_dir())
    if "exroma_v21_all" in asset_ids:
        return DEFAULT_DATASETS
    prefix = "exroma_v21_"
    for asset_id in asset_ids:
        if not asset_id.startswith(prefix):
            continue
        suffix = asset_id.removeprefix(prefix)
        repo_ids = tuple(suffix.split("__"))
        if repo_ids and all(repo_id in DEFAULT_DATASETS for repo_id in repo_ids):
            return repo_ids
    raise ValueError(
        f"Could not identify ExRoMa dataset assets in {assets_dir}. Found: {asset_ids}"
    )


class OpenPIExRoMaBackend:
    """Load a JAX pi0.5 checkpoint and expose ExRoMa's backend contract."""

    def __init__(self, config: dict[str, Any]) -> None:
        checkpoint_value = config.get("checkpoint_dir")
        if not checkpoint_value:
            raise ValueError(
                "pi0.5 requires an ExRoMa-finetuned checkpoint. Pass --checkpoint or set "
                "checkpoint_dir in deploy_policy.yml."
            )
        checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
        if not checkpoint.joinpath("params").is_dir():
            raise FileNotFoundError(f"OpenPI checkpoint params not found: {checkpoint / 'params'}")

        repo_ids = _checkpoint_repo_ids(checkpoint)
        action_horizon = int(config.get("action_horizon", 16))
        datasets_root = config.get("datasets_root") or (
            "/file_system/vepfs/algorithm/ruilin.wang/code/ExRoMa/"
            "ExRoMa-Datasets/lerobot_v21_release"
        )
        train_config = make_train_config(
            datasets_root=str(datasets_root),
            repo_ids=repo_ids,
            exp_name="inference",
            action_horizon=action_horizon,
        )

        from openpi.policies import policy_config

        LOGGER.info(
            "Loading OpenPI checkpoint %s (datasets=%s, horizon=%d)",
            checkpoint,
            ",".join(repo_ids),
            action_horizon,
        )
        self.policy = policy_config.create_trained_policy(train_config, checkpoint)
        self.checkpoint = checkpoint
        self.repo_ids = repo_ids

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        images = observation["input_rgb_arr"]
        if len(images) != 3:
            raise ValueError(f"OpenPI expected three camera images, got {len(images)}.")
        state = np.asarray(observation["input_state"], dtype=np.float32)
        result = self.policy.infer(
            {
                "images": {
                    "cam_high": np.ascontiguousarray(images[0], dtype=np.uint8),
                    "cam_right_wrist": np.ascontiguousarray(images[1], dtype=np.uint8),
                    "cam_left_wrist": np.ascontiguousarray(images[2], dtype=np.uint8),
                },
                "state": np.ascontiguousarray(state),
                "prompt": str(observation.get("prompt", "")),
            }
        )
        return {
            "actions": np.ascontiguousarray(result["actions"], dtype=np.float32),
            "openpi_timing": result.get("policy_timing", {}),
        }

    def reset(self) -> None:
        return None


def create_model(config: dict[str, Any]) -> OpenPIExRoMaBackend:
    return OpenPIExRoMaBackend(config)
