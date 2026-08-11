"""Compute statistics and fine-tune pi0.5 on local ExRoMa datasets."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from types import ModuleType

from openpi_exroma import (
    DEFAULT_DATASETS,
    compute_norm_stats,
    install_data_loader_patch,
    make_train_config,
    register_config,
)

DEFAULT_DATASETS_ROOT = Path(
    "/file_system/vepfs/algorithm/ruilin.wang/code/ExRoMa/ExRoMa-Datasets/lerobot_v21_release"
)
OPENPI_ROOT = Path(__file__).resolve().parent / "openpi"


def _load_openpi_script(name: str) -> ModuleType:
    path = OPENPI_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exroma_openpi_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load OpenPI script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_repo_ids(raw_tasks: list[str]) -> tuple[str, ...]:
    if raw_tasks == ["all"]:
        return DEFAULT_DATASETS
    if "all" in raw_tasks:
        raise ValueError("Use either '--tasks all' or explicit task names, not both.")
    unknown = sorted(set(raw_tasks) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"Unknown ExRoMa task dataset(s): {', '.join(unknown)}")
    return tuple(raw_tasks)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
        command.add_argument(
            "--tasks",
            nargs="+",
            default=["all"],
            help="Use 'all' or one or more dataset directory names.",
        )
        command.add_argument("--action-horizon", type=int, default=16)
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--num-workers", type=int, default=8)

    norm_stats = subparsers.add_parser(
        "norm-stats", help="Compute quantile normalization statistics."
    )
    add_data_args(norm_stats)
    norm_stats.add_argument("--max-frames", type=int)

    train = subparsers.add_parser(
        "train", help="LoRA fine-tune the official pi0.5 base checkpoint."
    )
    add_data_args(train)
    train.add_argument("--exp-name", required=True)
    train.add_argument("--num-train-steps", type=int, default=20_000)
    train.add_argument("--save-interval", type=int, default=1_000)
    train.add_argument("--keep-period", type=int, default=5_000)
    train.add_argument("--fsdp-devices", type=int, default=1)
    train.add_argument("--wandb", action="store_true")
    mode = train.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_ids = _parse_repo_ids(args.tasks)
    missing = [
        repo_id
        for repo_id in repo_ids
        if not (args.datasets_root / repo_id / "meta/info.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing dataset roots: {', '.join(missing)}")

    train_kwargs = {}
    if args.command == "train":
        train_kwargs = {
            "num_train_steps": args.num_train_steps,
            "save_interval": args.save_interval,
            "keep_period": args.keep_period,
            "fsdp_devices": args.fsdp_devices,
            "overwrite": args.overwrite,
            "resume": args.resume,
            "wandb_enabled": args.wandb,
        }
    config = make_train_config(
        datasets_root=args.datasets_root,
        repo_ids=repo_ids,
        exp_name=getattr(args, "exp_name", "norm_stats"),
        action_horizon=args.action_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **train_kwargs,
    )
    install_data_loader_patch()
    register_config(config)

    os.chdir(OPENPI_ROOT)
    if args.command == "norm-stats":
        compute_norm_stats(config, max_frames=args.max_frames)
    else:
        _load_openpi_script("train").main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
