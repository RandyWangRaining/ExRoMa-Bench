from __future__ import annotations

import json
from pathlib import Path
import re

from exroma_bench.tasks.benchmark_suite.registry import BENCHMARK_TASKS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_has_no_reference_repository_imports_or_machine_paths() -> None:
    forbidden_import = re.compile(r"^\s*(?:from|import)\s+srb(?:\.|\s|$)", re.MULTILINE)
    forbidden_text = (
        "ref_space_robotics_bench",
        "/home/ruilin/robotics",
        "/home/ruilin/ExRoMa-assets",
    )
    candidates = [
        *PROJECT_ROOT.joinpath("source").rglob("*.py"),
        *PROJECT_ROOT.joinpath("scripts").rglob("*.py"),
        *PROJECT_ROOT.joinpath("configs").rglob("*.yml"),
    ]
    violations: list[str] = []
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        if forbidden_import.search(text) or any(value in text for value in forbidden_text):
            violations.append(str(path.relative_to(PROJECT_ROOT)))
    assert violations == []


def test_public_registry_contains_only_supported_tasks() -> None:
    assert tuple(BENCHMARK_TASKS) == (
        "stack_blocks_two",
        "handover_block",
        "scan_object",
        "scan_rock",
        "beat_block_hammer",
        "test_tube_rack",
    )


def test_asset_manifest_covers_active_external_task_assets() -> None:
    manifest = json.loads(
        PROJECT_ROOT.joinpath("assets", "manifest.json").read_text(encoding="utf-8")
    )
    required = set(manifest["required_paths"])
    assert {
        "usd/robotwin/020_hammer/hammer.usd",
        "usd/robotwin/024_scanner/base0.usdc",
        "usd/robotwin/112_tea-box/base0.usdc",
        "usd/rocks/omnilrs/rock4057.usdz",
    } <= required
