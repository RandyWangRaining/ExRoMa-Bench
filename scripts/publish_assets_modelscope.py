#!/usr/bin/env python3
"""Publish the external ExRoMa asset bundle to a ModelScope dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO_ID = "ruilin.wang/ExRoMa-Assets"
MODELSCOPE_ENDPOINT = "https://modelscope.cn"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Local ExRoMa-Assets directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--allow-pattern",
        action="append",
        help="Upload only matching paths; repeat this option for multiple paths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the local resumable-upload cache.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Asset folder does not exist: {folder}")
    from modelscope_hub import HubApi

    api = HubApi(endpoint=MODELSCOPE_ENDPOINT)
    try:
        user = api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "ModelScope authentication is unavailable. Run "
            "'modelscope-hub --endpoint https://modelscope.cn login' or set "
            "MODELSCOPE_API_TOKEN before publishing."
        ) from exc
    print(f"Authenticated to ModelScope as {user.username}", flush=True)
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=folder,
        revision=args.revision,
        commit_message="Publish ExRoMa-Assets v0.1.0",
        allow_patterns=args.allow_pattern,
        ignore_patterns=[".git/**", "*.tmp", "*.part"],
        max_workers=args.max_workers,
        use_cache=not args.force,
    )
    print(f"Upload complete: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
