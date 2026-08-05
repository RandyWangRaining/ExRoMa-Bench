#!/usr/bin/env python3
"""Launch the ExRoMa Isaac Lab runtime."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if SOURCE_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SOURCE_ROOT.as_posix())

from exroma_bench.sim.entrypoint import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
