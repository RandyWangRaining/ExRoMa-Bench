#!/usr/bin/env python3
"""Serve a trained or custom ExRoMa policy over WebSocket."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if SOURCE_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SOURCE_ROOT.as_posix())

from exroma_bench.policy.plugins import discover_policy_plugins, load_named_policy
from exroma_bench.policy.server import (
    Hdf5ReplayPolicy,
    HoldPositionPolicy,
    RemotePolicyServer,
    load_policy_factory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--policy-name",
        help="Named algorithm workspace under --policy-root, for example ACT or pi05.",
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=PROJECT_ROOT / "policy",
        help="Directory containing algorithm-specific policy workspaces.",
    )
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="List discovered named policy workspaces and exit.",
    )
    parser.add_argument(
        "--policy-factory",
        help="Optional module:callable factory. It receives the policy config dictionary.",
    )
    parser.add_argument(
        "--policy-config",
        type=Path,
        help="YAML/JSON policy config; defaults to the named workspace config.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Override checkpoint_dir for a named policy."
    )
    parser.add_argument("--device", help="Override device for a named policy, for example cuda:0.")
    parser.add_argument(
        "--replay-episode",
        type=Path,
        help="Serve the recorded 16-D actions from an ExRoMa HDF5 episode.",
    )
    parser.add_argument("--replay-action-horizon", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_policies:
        plugins = discover_policy_plugins(args.policy_root)
        if not plugins:
            print(f"No policy workspaces found under {args.policy_root.expanduser().resolve()}")
            return 0
        for plugin in plugins.values():
            print(f"{plugin.name}\t{plugin.config_path}")
        return 0
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535.")
    if args.policy_name and args.policy_factory:
        raise ValueError("--policy-name and --policy-factory are mutually exclusive.")
    if args.replay_episode and args.policy_factory:
        raise ValueError("--replay-episode and --policy-factory are mutually exclusive.")
    if args.replay_episode and args.policy_name not in (None, "replay"):
        raise ValueError("--replay-episode can only be combined with --policy-name replay.")
    if args.policy_config is not None and not (args.policy_name or args.policy_factory):
        raise ValueError("--policy-config requires --policy-name or --policy-factory.")
    if (args.checkpoint is not None or args.device is not None) and not args.policy_name:
        raise ValueError("--checkpoint and --device require --policy-name.")
    if args.policy_name:
        policy = load_named_policy(
            args.policy_name,
            args.policy_root,
            config_path=args.policy_config,
            overrides={
                "checkpoint_dir": (
                    str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None
                ),
                "device": args.device,
                "episode_path": (
                    str(args.replay_episode.expanduser().resolve()) if args.replay_episode else None
                ),
                "action_horizon": (
                    args.replay_action_horizon if args.policy_name == "replay" else None
                ),
            },
        )
    elif args.replay_episode:
        policy = Hdf5ReplayPolicy(
            args.replay_episode,
            action_horizon=(
                8 if args.replay_action_horizon is None else args.replay_action_horizon
            ),
        )
        logging.getLogger(__name__).info(
            "Loaded replay episode %s (%d frames)",
            policy.episode_path,
            policy.frame_count,
        )
    elif args.policy_factory:
        policy = load_policy_factory(args.policy_factory, args.policy_config)
    else:
        policy = HoldPositionPolicy()
    RemotePolicyServer(policy, host=args.host, port=args.port).serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
