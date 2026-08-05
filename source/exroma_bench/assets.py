"""Download, select, and validate the external ExRoMa asset bundle."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import PROJECT_ROOT, asset_root

DEFAULT_PROVIDER = "modelscope"
MODELSCOPE_ENDPOINT = "https://www.modelscope.ai"
DEFAULT_REPO_IDS = {
    "modelscope": "ruilin.wang/ExRoMa-Assets",
    "huggingface": "wrl2003/ExRoMa-Assets",
}
MANIFEST_PATH = PROJECT_ROOT / "assets" / "manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def required_paths() -> tuple[str, ...]:
    return tuple(load_manifest()["required_paths"])


def verify_assets(root: Path | None = None) -> list[Path]:
    root = (root or asset_root()).expanduser().resolve()
    return [root / path for path in required_paths() if not (root / path).is_file()]


def link_assets(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Asset directory does not exist: {source}")
    marker = PROJECT_ROOT / ".asset-root"
    marker.write_text(f"{source}\n", encoding="utf-8")
    return marker


def pull_assets(
    *,
    repo_id: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    root: Path | None = None,
) -> Path:
    if provider not in DEFAULT_REPO_IDS:
        raise ValueError(f"Unsupported asset provider: {provider}")

    repo_id = repo_id or DEFAULT_REPO_IDS[provider]
    destination = (root or asset_root()).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if provider == "modelscope":
        try:
            from modelscope_hub import HubApi
        except ImportError as exc:
            raise RuntimeError(
                "modelscope-hub is required. Install ExRoMa's compatible requirements."
            ) from exc
        downloaded = Path(
            HubApi(endpoint=MODELSCOPE_ENDPOINT).download_repo(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=destination,
            )
        ).resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required. Install ExRoMa's compatible requirements."
            ) from exc
        downloaded = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=destination,
            )
        ).resolve()

    missing = verify_assets(downloaded)
    if missing:
        preview = ", ".join(str(path.relative_to(downloaded)) for path in missing[:5])
        raise RuntimeError(
            f"Downloaded asset bundle is incomplete ({len(missing)} required files missing): "
            f"{preview}"
        )
    return downloaded
