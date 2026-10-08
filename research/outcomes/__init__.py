"""Outcome table & trigger-based label coordination (feature 021).

Builds a per-bar outcome table from the shared exit policy (`utils.ExitPolicy`), so that
evaluating a candidate label's economic value or learnability is a query against a cached
table instead of a training run. Nothing in this package computes, reports or gates on
backtest return, P&L, drawdown or trade counts from a live strategy run (FR-017/FR-018) — the
table's economic value is a lookahead bound, never a realisable return.
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_VERSION = "0.1.0"

#: repo root = user_data/strategies
ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
CACHE_DIR = PKG / ".cache"


def assert_interpreter() -> None:
    """Refuse to run outside the repo's .venv.

    Guard on `sys.prefix`, never `Path(sys.executable).resolve()`: the venv's `bin/python` is a
    SYMLINK to the pyenv binary, so resolving it lands outside the venv (recorded 2026-09-15).
    """
    if sys.prefix == sys.base_prefix or "freqtrade/.venv" not in sys.prefix:
        raise SystemExit(
            f"wrong interpreter (prefix {sys.prefix})\n"
            f"use {ROOT.parent.parent}/.venv/bin/python"
        )


assert_interpreter()
