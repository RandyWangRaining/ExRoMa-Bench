"""Unified ExRoMa command line."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .assets import link_assets, pull_assets, verify_assets
from .paths import PROJECT_ROOT, asset_root, isaaclab_root

SIM_COMMANDS = {"preview", "collect", "evaluate"}


def _asset_parser(subparsers) -> None:
    assets = subparsers.add_parser("assets", help="Manage the external asset bundle.")
    commands = assets.add_subparsers(dest="asset_command", required=True)
    pull = commands.add_parser("pull", help="Download the external asset bundle.")
    pull.add_argument("--repo-id")
    pull.add_argument("--root", type=Path)
    link = commands.add_parser("link", help="Use an existing local asset directory.")
    link.add_argument("source", type=Path)
    commands.add_parser("verify", help="Check required runtime assets.")
    commands.add_parser("path", help="Print the selected asset root.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exroma", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _asset_parser(subparsers)
    subparsers.add_parser("doctor", help="Check the local runtime installation.")
    for command in sorted(SIM_COMMANDS):
        subparsers.add_parser(command, add_help=False)
    real = subparsers.add_parser("real", add_help=False)
    real.set_defaults(command="real")
    return parser


def _run_sim(command: str, passthrough: list[str]) -> int:
    root = isaaclab_root(require=True)
    script = PROJECT_ROOT / "scripts" / "run_sim.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(PROJECT_ROOT / "source"), env.get("PYTHONPATH", "")])
    )
    return subprocess.call(
        [str(root / "isaaclab.sh"), "-p", str(script), command, *passthrough],
        cwd=PROJECT_ROOT,
        env=env,
    )


def _doctor() -> int:
    missing = verify_assets()
    lab_root = isaaclab_root()
    checks = {
        "project": PROJECT_ROOT,
        "assets": asset_root(),
        "isaaclab": lab_root,
    }
    for name, value in checks.items():
        print(f"{name:10s}: {value}")
    print(f"assets_ok : {not missing}")
    for path in missing:
        print(f"missing   : {path}")
    required_checks = {
        "python_3_11": sys.version_info[:2] == (3, 11),
        "isaacsim": importlib.util.find_spec("isaacsim") is not None,
        "isaaclab_py": importlib.util.find_spec("isaaclab") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "curobo": importlib.util.find_spec("curobo") is not None,
        "simforge": importlib.util.find_spec("simforge") is not None,
        "foundry": importlib.util.find_spec("simforge_foundry") is not None,
    }
    optional_checks = {
        "blender": shutil.which("blender") is not None,
    }
    for name, available in required_checks.items():
        print(f"{name:10s}: {available}")
    for name, available in optional_checks.items():
        print(f"{name:10s}: {available} (optional for baked terrain)")
    isaaclab_ok = (lab_root / "isaaclab.sh").is_file()
    return 1 if missing or not isaaclab_ok or not all(required_checks.values()) else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.command == "assets":
        if passthrough:
            parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
        if args.asset_command == "pull":
            destination = pull_assets(
                repo_id=args.repo_id,
                root=args.root,
            )
            print(destination)
            return 0
        if args.asset_command == "link":
            marker = link_assets(args.source)
            print(f"Linked assets through {marker}: {asset_root()}")
            return 0
        if args.asset_command == "path":
            print(asset_root())
            return 0
        missing = verify_assets()
        if missing:
            for path in missing:
                print(f"MISSING {path}")
            return 1
        print(f"Asset bundle verified: {asset_root()}")
        return 0
    if args.command == "doctor":
        return _doctor()
    if args.command == "real":
        from .real.runner import main as real_main

        return real_main(passthrough)
    return _run_sim(args.command, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
