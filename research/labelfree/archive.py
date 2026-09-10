"""Read the arbiter's realised trades from the backtest archive (T008).

FR-008: calibration replays the entry/exit timestamps the arbiter actually produced, as a
fixed series. This module is that reader; it never simulates anything.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ARCHIVE = Path(__file__).resolve().parents[3] / "backtest_results"

TRADE_COLS = ["pair", "open_date", "close_date", "open_rate", "close_rate",
              "amount", "stake_amount", "fee_open", "fee_close",
              "profit_abs", "profit_ratio", "trade_duration", "exit_reason"]


def latest_zip(strategy: str | None = None, archive: Path = ARCHIVE) -> Path:
    zips = sorted(archive.glob("backtest-result-*.zip"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    for z in zips:
        if strategy is None:
            return z
        try:
            if strategy in _strategies(z):
                return z
        except Exception:
            continue
    raise FileNotFoundError(f"no archive with strategy={strategy!r} under {archive}")


def _strategies(z: Path) -> list[str]:
    with zipfile.ZipFile(z) as f:
        main = [n for n in f.namelist()
                if n.endswith(".json") and "config" not in n and n.count("_") < 3]
        for n in sorted(main, key=len):
            d = json.loads(f.read(n))
            if "strategy" in d:
                return list(d["strategy"])
    return []


def load_trades(z: Path, strategy: str | None = None) -> pd.DataFrame:
    """Realised trades as a DataFrame with tz-aware dates."""
    with zipfile.ZipFile(z) as f:
        main = [n for n in f.namelist()
                if n.endswith(".json") and "config" not in n and n.count("_") < 3]
        for n in sorted(main, key=len):
            d = json.loads(f.read(n))
            if "strategy" not in d:
                continue
            key = strategy or list(d["strategy"])[0]
            if key not in d["strategy"]:
                continue
            tr = pd.DataFrame(d["strategy"][key]["trades"])
            if tr.empty:
                return tr
            for c in ("open_date", "close_date"):
                tr[c] = pd.to_datetime(tr[c], utc=True)
            keep = [c for c in TRADE_COLS if c in tr.columns]
            return tr[keep].sort_values(["open_date", "pair"]).reset_index(drop=True)
    raise KeyError(f"strategy {strategy!r} not found in {z}")


def summarise(tr: pd.DataFrame) -> dict:
    return {
        "n_trades": int(len(tr)),
        "profit_abs_total": float(tr["profit_abs"].sum()) if len(tr) else 0.0,
        "pairs": sorted(tr["pair"].unique().tolist()) if len(tr) else [],
        "first_open": str(tr["open_date"].min()) if len(tr) else None,
        "last_close": str(tr["close_date"].max()) if len(tr) else None,
    }
