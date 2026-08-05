from __future__ import annotations

from pathlib import Path

from exroma_bench import cli


def _configure_healthy_runtime(monkeypatch, tmp_path: Path) -> None:
    isaaclab = tmp_path / "IsaacLab"
    isaaclab.mkdir()
    (isaaclab / "isaaclab.sh").touch()

    monkeypatch.setattr(cli, "verify_assets", list)
    monkeypatch.setattr(cli, "asset_root", lambda: tmp_path / "assets")
    monkeypatch.setattr(cli, "isaaclab_root", lambda: isaaclab)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/usr/bin/blender")


def test_doctor_accepts_complete_runtime(monkeypatch, tmp_path: Path) -> None:
    _configure_healthy_runtime(monkeypatch, tmp_path)

    assert cli._doctor() == 0


def test_doctor_rejects_missing_curobo(monkeypatch, tmp_path: Path) -> None:
    _configure_healthy_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli.importlib.util,
        "find_spec",
        lambda name: None if name == "curobo" else object(),
    )

    assert cli._doctor() == 1
