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
        "--policy-factory",
        help="Optional module:callable factory. It receives the policy config dictionary.",
    )
    parser.add_argument("--policy-config", type=Path, help="JSON passed to --policy-factory.")
    parser.add_argument(
        "--replay-episode",
        type=Path,
        help="Serve the recorded 16-D actions from an ExRoMa HDF5 episode.",
    )
    parser.add_argument("--replay-action-horizon", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535.")
    if args.policy_config is not None and args.policy_factory is None:
        raise ValueError("--policy-config requires --policy-factory.")
    if args.policy_factory and args.replay_episode:
        raise ValueError("--policy-factory and --replay-episode are mutually exclusive.")
    if args.replay_episode:
        policy = Hdf5ReplayPolicy(
            args.replay_episode,
            action_horizon=args.replay_action_horizon,
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
