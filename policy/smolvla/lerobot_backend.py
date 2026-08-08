"""Lazy native LeRobot backend for SmolVLA inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from exroma_bench.policy.protocol import ACTION_DIM


class LeRobotSmolVLABackend:
    """Load a fine-tuned SmolVLA checkpoint and its saved processors once."""

    def __init__(self, config: dict[str, Any]) -> None:
        checkpoint = config.get("checkpoint_dir")
        if not checkpoint:
            raise ValueError(
                "SmolVLA requires an ExRoMa-finetuned checkpoint. Pass --checkpoint or set "
                "checkpoint_dir in deploy_policy.yml."
            )
        try:
            import torch
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.smolvla import SmolVLAPolicy
            from lerobot.policies.utils import prepare_observation_for_inference
        except ImportError as exc:
            raise ImportError(
                "SmolVLA dependencies are missing. Create policy/smolvla/conda_env.yaml "
                "and run the policy server from the exroma-smolvla environment."
            ) from exc

        self.torch = torch
        self.prepare_observation = prepare_observation_for_inference
        self.device = torch.device(str(config.get("device", "cuda:0")))
        self.checkpoint = str(checkpoint)
        self.policy = SmolVLAPolicy.from_pretrained(self.checkpoint)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            self.checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        raw = {
            key: np.array(value, copy=True)
            for key, value in observation.items()
            if key.startswith("observation.")
        }
        batch = self.prepare_observation(
            raw,
            self.device,
            task=str(observation.get("task", "")),
            robot_type=str(observation.get("robot_type", "dual_piper_rover")),
        )
        batch = self.preprocessor(batch)
        with self.torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
            if actions.ndim != 3 or actions.shape[0] != 1:
                raise ValueError(
                    f"SmolVLA predict_action_chunk must return [1,T,D], got {tuple(actions.shape)}."
                )
            processed = [
                self.postprocessor(actions[:, index, :]) for index in range(actions.shape[1])
            ]
            actions = self.torch.stack(processed, dim=1)
        array = actions.squeeze(0).detach().cpu().float().numpy()
        if array.ndim != 2 or array.shape[1] != ACTION_DIM:
            raise ValueError(
                f"SmolVLA checkpoint returned {array.shape}; ExRoMa requires [T,{ACTION_DIM}]. "
                "Fine-tune the checkpoint on the ExRoMa 16-D dataset contract."
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    def reset(self) -> None:
        self.policy.reset()
