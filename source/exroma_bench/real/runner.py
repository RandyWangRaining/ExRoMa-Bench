"""Guarded entry point for deploying a trained ExRoMa policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/real/dual_piper_rover.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--confirm-hardware", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    print(f"robot            : {config['robot']['name']}")
    print(f"policy state dim : {config['policy_interface']['robot_state_dim']}")
    print(f"policy action dim: {config['policy_interface']['action_dim']}")
    print(f"camera streams   : {', '.join(config['cameras'])}")
    if args.dry_run:
        print("DRY RUN: configuration validated; no hardware command was sent.")
        return 0
    if os.environ.get("EXROMA_REAL_ROBOT_ENABLED") != "1":
        raise RuntimeError(
            "Real motion is disabled. Set EXROMA_REAL_ROBOT_ENABLED=1 only after "
            "calibration, limit, communication, and emergency-stop checks."
        )
    if not args.confirm_hardware:
        raise RuntimeError("Real motion also requires --confirm-hardware.")
    if args.checkpoint is None or not args.checkpoint.expanduser().is_file():
        raise FileNotFoundError("A valid --checkpoint is required for real execution.")
    raise NotImplementedError(
        "No site-specific PiPER/base hardware adapter is configured. Implement "
        "exroma_bench.real.interfaces.HardwareAdapter and register it in the real config."
    )
