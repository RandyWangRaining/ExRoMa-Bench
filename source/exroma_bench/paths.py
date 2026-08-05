"""Relocatable project and asset paths."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT_ENV = "EXROMA_ASSET_ROOT"
ISAACLAB_ROOT_ENV = "EXROMA_ISAACLAB_ROOT"


def _expanded(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def asset_root(*, require: bool = False) -> Path:
    """Return the external asset root without embedding workstation paths."""

    configured = os.environ.get(ASSET_ROOT_ENV)
    if configured:
        root = _expanded(configured)
    else:
        marker = PROJECT_ROOT / ".asset-root"
        if marker.is_file():
            root = _expanded(marker.read_text(encoding="utf-8").strip())
        elif (PROJECT_ROOT / "assets" / "runtime").is_dir():
            root = (PROJECT_ROOT / "assets" / "runtime").resolve()
        else:
            root = (Path.home() / ".cache" / "exroma" / "assets").resolve()
    if require and not root.is_dir():
        raise FileNotFoundError(
            f"ExRoMa asset root does not exist: {root}. "
            "Run `exroma assets pull` or `exroma assets link PATH`."
        )
    return root


def asset_path(relative_path: str | Path, *, require: bool = True) -> Path:
    path = asset_root(require=require) / Path(relative_path)
    if require and not path.is_file():
        raise FileNotFoundError(
            f"Required ExRoMa asset is missing: {path}. Run `exroma assets verify`."
        )
    return path


def isaaclab_root(*, require: bool = False) -> Path:
    candidates = []
    configured = os.environ.get(ISAACLAB_ROOT_ENV)
    if configured:
        candidates.append(_expanded(configured))
    candidates.extend(
        [
            PROJECT_ROOT.parent / "IsaacLab",
            Path.home() / "robotics" / "IsaacLab",
        ]
    )
    for candidate in candidates:
        if (candidate / "isaaclab.sh").is_file():
            return candidate.resolve()
    fallback = candidates[0] if candidates else Path.home() / "robotics" / "IsaacLab"
    if require:
        raise FileNotFoundError(
            "Isaac Lab was not found. Set EXROMA_ISAACLAB_ROOT to its checkout."
        )
    return fallback.resolve()
