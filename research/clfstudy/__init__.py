"""NNC classifier architecture & input-data study (feature 018).

Measures CLASSIFIERS, not strategies: every number produced here is a classification metric
against the fixed gbb ternary labels (SELL=0, HOLD=1, BUY=2). Nothing in this package computes,
reports or gates on backtest return, P&L, drawdown or trade counts (spec FR-036). Its outputs
are a screen for a later, separate P&L evaluation (Constitution III).

Pinned constants live here so every module and every results row cites the same values.
"""
from __future__ import annotations

import sys
from pathlib import Path

HARNESS_VERSION = "0.1.0"

#: label look-ahead in bars (TrainingConfig.HORIZON); the embargo must be >= this
HORIZON = 48
#: bars between the end of any training span and the start of its evaluation window (2x HORIZON)
EMBARGO_BARS = 96
#: fraction of each training span (by time, from the end) used as the early-stopping slice
VAL_FRACTION = 0.10
#: a pair enters a window only with at least this much history before the window's val slice
MIN_HISTORY_DAYS = 365
TIMEFRAME = "15m"
EXCHANGE = "binanceus"
DATA_START = "2022-01-01"
#: the production static whitelist (config/config.json), in its order
PAIRS = (
    "ZEC/USDT", "XRP/USDT", "SOL/USDT", "LINK/USDT", "NEAR/USDT", "AAVE/USDT",
    "SUI/USDT", "AVAX/USDT", "LTC/USDT", "BCH/USDT", "DOT/USDT",
)
CLASS_NAMES = ("SELL", "HOLD", "BUY")
NUM_CLASSES = 3

#: repo root = user_data/strategies
ROOT = Path(__file__).resolve().parents[2]
PKG = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data" / EXCHANGE
CACHE_DIR = PKG / ".cache"
RESULTS_PATH = PKG / "results" / "results.jsonl"
SPLITS_DIR = PKG / "splits"
GATES_DIR = PKG / "gates"
REPORT_DIR = PKG / "report"
CONFIGS_DIR = PKG / "configs"


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
