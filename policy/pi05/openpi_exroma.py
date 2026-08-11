"""ExRoMa data transforms and training config for the official OpenPI package."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import einops
import numpy as np
import openpi.training.config as openpi_config
import openpi.training.data_loader as openpi_data_loader
import openpi.training.optimizer as openpi_optimizer
from openpi import transforms
from openpi.models import pi0_config
from openpi.training import weight_loaders

CONFIG_NAME = "pi05_base_exroma_lora"
ACTION_DIM = 16
MODEL_ACTION_DIM = 32
DEFAULT_ACTION_HORIZON = 16
DEFAULT_DATASETS = (
    "beat_block_hammer_procedural_moon",
    "handover_block_procedural_moon",
    "stack_blocks_two_procedural_moon",
    "test_tube_rack_procedural_moon",
)


def _parse_image(image: Any) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {image.shape}.")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}.")
    return image


@dataclasses.dataclass(frozen=True)
class ExRoMaInputs(transforms.DataTransformFn):
    """Map ExRoMa observations to the three-camera pi0.5 input contract."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        images = data["images"]
        result = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(images["cam_high"]),
                "left_wrist_0_rgb": _parse_image(images["cam_left_wrist"]),
                "right_wrist_0_rgb": _parse_image(images["cam_right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            result["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return result


@dataclasses.dataclass(frozen=True)
class ExRoMaOutputs(transforms.DataTransformFn):
    """Remove the model's padding dimensions from predicted actions."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.asarray(data["actions"])[..., :ACTION_DIM]}


@dataclasses.dataclass(frozen=True)
class PromptFromExRoMaTask(transforms.DataTransformFn):
    tasks: tuple[dict[int, str], ...]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        dataset_index = int(data["dataset_index"])
        task_index = int(data["task_index"])
        try:
            prompt = self.tasks[dataset_index][task_index]
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"No prompt for dataset_index={dataset_index}, task_index={task_index}."
            ) from exc
        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class ExRoMaDataConfig(openpi_config.DataConfig):
    repo_ids: tuple[str, ...] = ()
    repo_root: str = ""


@dataclasses.dataclass(frozen=True)
class LeRobotExRoMaDataConfig(openpi_config.DataConfigFactory):
    repo_ids: tuple[str, ...] = DEFAULT_DATASETS
    repo_root: str = ""
    use_delta_joint_actions: bool = True

    def create(
        self,
        assets_dirs: Path,
        model_config: pi0_config.Pi0Config,
    ) -> ExRoMaDataConfig:
        if not self.repo_ids:
            raise ValueError("At least one ExRoMa dataset is required.")
        if not self.repo_root:
            raise ValueError("The ExRoMa dataset root is required.")

        asset_id = self.assets.asset_id or self.repo_id
        norm_stats = self._load_norm_stats(Path(self.assets.assets_dir or assets_dirs), asset_id)
        data_transforms = transforms.Group(
            inputs=[ExRoMaInputs()],
            outputs=[ExRoMaOutputs()],
        )
        if self.use_delta_joint_actions:
            # Arm joints are targets relative to their current positions. Grippers
            # and the two rover velocity commands remain absolute.
            delta_mask = transforms.make_bool_mask(6, -1, 6, -3)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_mask)],
                outputs=[transforms.AbsoluteActions(delta_mask)],
            )

        return ExRoMaDataConfig(
            repo_id=self.repo_id,
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            data_transforms=data_transforms,
            model_transforms=openpi_config.ModelTransformFactory()(model_config),
            use_quantile_norm=True,
            action_sequence_keys=("action",),
            prompt_from_task=False,
            repo_ids=self.repo_ids,
            repo_root=self.repo_root,
        )


def _dataset_asset_id(repo_ids: Sequence[str]) -> str:
    if tuple(repo_ids) == DEFAULT_DATASETS:
        return "exroma_v21_all"
    return "exroma_v21_" + "__".join(repo_ids)


def make_train_config(
    *,
    datasets_root: str | Path,
    repo_ids: Sequence[str],
    exp_name: str,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    batch_size: int = 32,
    num_workers: int = 8,
    num_train_steps: int = 20_000,
    save_interval: int = 1_000,
    keep_period: int | None = 5_000,
    fsdp_devices: int = 1,
    overwrite: bool = False,
    resume: bool = False,
    wandb_enabled: bool = False,
) -> openpi_config.TrainConfig:
    repo_ids = tuple(repo_ids)
    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=MODEL_ACTION_DIM,
        action_horizon=action_horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    workspace = Path(__file__).resolve().parent / "openpi"
    return openpi_config.TrainConfig(
        name=CONFIG_NAME,
        project_name="exroma-pi05",
        exp_name=exp_name,
        model=model,
        data=LeRobotExRoMaDataConfig(
            repo_id=_dataset_asset_id(repo_ids),
            repo_ids=repo_ids,
            repo_root=str(Path(datasets_root).resolve()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=openpi_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=num_train_steps,
            decay_lr=2.5e-6,
        ),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        assets_base_dir=str(workspace / "assets"),
        checkpoint_base_dir=str(workspace / "checkpoints"),
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        save_interval=save_interval,
        keep_period=keep_period,
        fsdp_devices=fsdp_devices,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=wandb_enabled,
    )


def install_data_loader_patch() -> None:
    """Teach OpenPI's loader to concatenate local ExRoMa task datasets."""

    if getattr(openpi_data_loader.create_torch_dataset, "_supports_exroma", False):
        return

    original_create_torch_dataset = openpi_data_loader.create_torch_dataset

    def create_torch_dataset(
        data_config: openpi_config.DataConfig,
        action_horizon: int,
        model_config: Any,
    ) -> openpi_data_loader.Dataset:
        if not isinstance(data_config, ExRoMaDataConfig):
            return original_create_torch_dataset(data_config, action_horizon, model_config)

        dataset = _create_lerobot_dataset(data_config, action_horizon, decode_images=True)
        task_maps = tuple(subdataset.meta.tasks for subdataset in dataset._datasets)
        return openpi_data_loader.TransformedDataset(
            dataset,
            [PromptFromExRoMaTask(task_maps)],
        )

    create_torch_dataset._supports_exroma = True  # type: ignore[attr-defined]
    openpi_data_loader.create_torch_dataset = create_torch_dataset


def register_config(config: openpi_config.TrainConfig) -> None:
    """Register a runtime config for OpenPI helper scripts."""

    openpi_config._CONFIGS_DICT[config.name] = config


def _create_lerobot_dataset(
    data_config: ExRoMaDataConfig,
    action_horizon: int,
    *,
    decode_images: bool,
) -> Any:
    from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset

    dataset = MultiLeRobotDataset(
        list(data_config.repo_ids),
        root=data_config.repo_root,
        delta_timestamps={
            key: [step / 10 for step in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        video_backend="pyav",
    )
    if not decode_images:
        for subdataset in dataset._datasets:
            subdataset.meta.info["features"] = {
                key: feature
                for key, feature in subdataset.meta.info["features"].items()
                if feature["dtype"] not in {"image", "video"}
            }
    return dataset


def compute_norm_stats(
    config: openpi_config.TrainConfig,
    *,
    max_frames: int | None = None,
) -> Path:
    """Compute state/action stats without decoding video observations."""

    import torch
    import tqdm
    from openpi.shared import normalize

    data_config = config.data.create(config.assets_dirs, config.model)
    if not isinstance(data_config, ExRoMaDataConfig):
        raise TypeError(f"Expected ExRoMaDataConfig, got {type(data_config).__name__}.")
    dataset = _create_lerobot_dataset(
        data_config,
        config.model.action_horizon,
        decode_images=False,
    )
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("--max-frames must be at least 2.")
        dataset = torch.utils.data.Subset(dataset, range(min(max_frames, len(dataset))))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    running_stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    delta_mask = np.asarray(transforms.make_bool_mask(6, -1, 6, -3))
    for batch in tqdm.tqdm(data_loader, desc="Computing stats"):
        state = np.asarray(batch["observation.state"], dtype=np.float32)
        actions = np.asarray(batch["action"], dtype=np.float32)
        actions[..., delta_mask] -= state[:, np.newaxis, delta_mask]
        running_stats["state"].update(state)
        running_stats["actions"].update(actions)

    stats = {key: value.get_statistics() for key, value in running_stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    normalize.save(output_path, stats)
    print(f"Writing stats to: {output_path}")
    return output_path
