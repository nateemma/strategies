"""035 T004: archive parsers and causal alignment (research R3, R4, R5).

Downloaded archives are untrusted data: they are read with zipfile into pandas, never extracted or
executed. Freqtrade stamps a candle at its OPEN; bar `date` closes at `date + 15m`.

  kline sources (global spot, Binance.US, premium index): the row with the same open stamp covers
      the same interval and closes at the same moment -> aligned to the same `date`;
  5-min metrics snapshots: usable SNAPSHOT_DELAY_MIN after their `create_time`; bar t takes the
      last snapshot with create_time + delay <= t + 15m, and none older than SNAPSHOT_STALE_MIN.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from research.new_inputs import inputs as I


BAR = pd.Timedelta(minutes=15)
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
METRIC_COLS = ["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
               "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
               "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def to_utc(stamps) -> pd.DatetimeIndex:
    """Epoch stamps in ms or us (spot archives from 2025 use us), detected per value."""
    v = np.asarray(stamps, dtype="int64")
    us = v > 10**14
    ms = np.where(us, v // 1000, v)
    return pd.DatetimeIndex(pd.to_datetime(ms, unit="ms", utc=True))


def _first_member_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected one csv member, got {names}")
        return z.read(names[0]).decode("utf-8")


def parse_klines_text(text: str) -> pd.DataFrame:
    """A kline csv with or without a header row (premium-index files have one)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and not lines[0].split(",")[0].strip().lstrip("-").isdigit():
        lines = lines[1:]
    if not lines:
        return pd.DataFrame(columns=KLINE_COLS[:-1])
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None)
    df = df.iloc[:, :len(KLINE_COLS)]
    df.columns = KLINE_COLS[:df.shape[1]]
    df["open_time"] = to_utc(df["open_time"])
    for c in KLINE_COLS[1:-1]:
        if c in df and c != "close_time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop(columns=[c for c in ("close_time", "ignore") if c in df])


def parse_metrics_text(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    missing = set(METRIC_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"metrics file lacks {sorted(missing)}")
    df = df[METRIC_COLS].copy()
    df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
    for c in METRIC_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop(columns=["symbol"])


def read_klines_zip(path: Path) -> pd.DataFrame:
    return parse_klines_text(_first_member_text(path))


def read_metrics_zip(path: Path) -> pd.DataFrame:
    return parse_metrics_text(_first_member_text(path))


def read_us_jsonl(path: Path) -> pd.DataFrame:
    """Binance.US REST pages, one JSON array of klines per line (written by fetch.py)."""
    rows = []
    for ln in Path(path).read_text().splitlines():
        if ln.strip():
            rows.extend(json.loads(ln))
    if not rows:
        return pd.DataFrame(columns=KLINE_COLS[:-1])
    df = pd.DataFrame([r[:len(KLINE_COLS)] for r in rows], columns=KLINE_COLS)
    df["open_time"] = to_utc(df["open_time"])
    for c in KLINE_COLS[1:-1]:
        if c != "close_time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop(columns=["close_time", "ignore"])


def dedupe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)


def align_kline(src: pd.DataFrame, dates: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    """Same open stamp -> same bar. Absent rows are NaN (never a computed zero)."""
    s = dedupe(src, "open_time").set_index("open_time")
    return s.reindex(pd.DatetimeIndex(dates))[cols]


def align_snapshots(src: pd.DataFrame, dates: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    """Backward as-of: the last snapshot usable by the bar close, at most SNAPSHOT_STALE_MIN old."""
    s = dedupe(src, "create_time")
    s = s.assign(avail=s["create_time"] + pd.Timedelta(minutes=I.SNAPSHOT_DELAY_MIN))
    left = pd.DataFrame({"bar_close": pd.DatetimeIndex(dates) + BAR})
    m = pd.merge_asof(left, s[["avail", *cols]].sort_values("avail"), left_on="bar_close",
                      right_on="avail", direction="backward",
                      tolerance=pd.Timedelta(minutes=I.SNAPSHOT_STALE_MIN))
    out = m[cols]
    out.index = pd.DatetimeIndex(dates)
    return out


def truncate_klines(src: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Kline rows that have CLOSED by the close of bar `cut`."""
    return src[src["open_time"] <= cut]


def truncate_snapshots(src: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Snapshots usable by the close of bar `cut`."""
    avail = src["create_time"] + pd.Timedelta(minutes=I.SNAPSHOT_DELAY_MIN)
    return src[avail <= cut + BAR]
