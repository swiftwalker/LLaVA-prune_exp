#!/usr/bin/env python3
"""Collect metadata from an official VisionZip run directory."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sparsevlm" / "scripts"))

from collect_results import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
