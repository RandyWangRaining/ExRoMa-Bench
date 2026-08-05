from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from exroma_bench import assets


def test_modelscope_is_the_default_asset_provider(monkeypatch, tmp_path: Path) -> None:
    calls = {}
    module = ModuleType("modelscope_hub")

    class FakeHubApi:
        def __init__(self, **kwargs):
            calls["client"] = kwargs

        def download_repo(self, **kwargs):
            calls["download"] = kwargs
            return kwargs["local_dir"]

    module.HubApi = FakeHubApi
    monkeypatch.setitem(sys.modules, "modelscope_hub", module)
    monkeypatch.setattr(assets, "verify_assets", lambda _root: [])

    destination = assets.pull_assets(root=tmp_path)

    assert destination == tmp_path.resolve()
    assert calls == {
        "client": {"endpoint": "https://www.modelscope.ai"},
        "download": {
            "repo_id": "ruilin.wang/ExRoMa-Assets",
            "repo_type": "dataset",
            "local_dir": tmp_path.resolve(),
        },
    }


def test_huggingface_remains_available_as_a_fallback(monkeypatch, tmp_path: Path) -> None:
    calls = {}
    module = ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return kwargs["local_dir"]

    module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setattr(assets, "verify_assets", lambda _root: [])

    destination = assets.pull_assets(provider="huggingface", root=tmp_path)

    assert destination == tmp_path.resolve()
    assert calls == {
        "repo_id": "wrl2003/ExRoMa-Assets",
        "repo_type": "model",
        "local_dir": tmp_path.resolve(),
    }


def test_incomplete_download_is_rejected(monkeypatch, tmp_path: Path) -> None:
    module = ModuleType("modelscope_hub")

    class FakeHubApi:
        def __init__(self, **_kwargs):
            pass

        def download_repo(self, **kwargs):
            return kwargs["local_dir"]

        def download_file(self, **_kwargs):
            return missing

    module.HubApi = FakeHubApi
    monkeypatch.setitem(sys.modules, "modelscope_hub", module)
    missing = tmp_path / "usd" / "terrain" / "moon.usdz"
    monkeypatch.setattr(assets, "verify_assets", lambda _root: [missing])

    with pytest.raises(RuntimeError, match="incomplete .*1 required files missing"):
        assets.pull_assets(root=tmp_path)


def test_modelscope_recovers_files_missing_from_snapshot_tree(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    module = ModuleType("modelscope_hub")
    missing = tmp_path / "usd" / "scenery" / "lunalab.usdc"

    class FakeHubApi:
        def __init__(self, **_kwargs):
            pass

        def download_repo(self, **kwargs):
            return kwargs["local_dir"]

        def download_file(self, **kwargs):
            calls.append(kwargs)
            missing.parent.mkdir(parents=True)
            missing.touch()
            return missing

    module.HubApi = FakeHubApi
    monkeypatch.setitem(sys.modules, "modelscope_hub", module)
    monkeypatch.setattr(
        assets,
        "verify_assets",
        lambda _root: [] if missing.is_file() else [missing],
    )

    destination = assets.pull_assets(root=tmp_path)

    assert destination == tmp_path.resolve()
    assert calls == [
        {
            "repo_id": "ruilin.wang/ExRoMa-Assets",
            "repo_type": "dataset",
            "file_path": "usd/scenery/lunalab.usdc",
            "local_dir": tmp_path.resolve(),
            "force": True,
        }
    ]
