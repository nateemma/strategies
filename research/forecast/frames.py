"""038 candle frames (research R1-R3, data-model § Candle frame): per pair, gap-filled 15m candles with
log close, the N-candle targets, the next-open trade returns, trade eligibility, fold and half.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.frames [--pair SOL/USDT]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from research.forecast import common as C


def build(filled: pd.DataFrame, raw_dates: pd.Series, data_ok: pd.Series | None,
          fold_windows: dict, fwd_start: pd.Timestamp = C.FWD_START) -> pd.DataFrame:
    """The frame from gap-filled candles. `raw_dates` are the candles present in the raw file;
    `data_ok` (indexed by date) is the 037 warm-data flag, `has_data` where it has no value."""
    d = filled[["date", "open", "close", "volume"]].copy().reset_index(drop=True)
    d["has_data"] = d["date"].isin(set(pd.to_datetime(raw_dates))).astype(bool)
    if data_ok is not None:
        ok = d["date"].map(data_ok)
        d["data_ok"] = np.where(ok.isna(), d["has_data"] & (d["date"] >= fwd_start), ok.fillna(0) > 0)
    else:
        d["data_ok"] = d["has_data"]
    d["data_ok"] = d["data_ok"].astype(bool)
    d["lc"] = np.log(d["close"].to_numpy(float))
    lo = np.log(d["open"].to_numpy(float))
    traded = (d["volume"].to_numpy(float) > 0) & d["has_data"].to_numpy()
    n_rows = len(d)
    for n in C.ALL_HORIZONS:
        tgt = np.full(n_rows, np.nan)
        if n < n_rows:  # a horizon longer than the data has no target
            tgt[: n_rows - n] = d["lc"].to_numpy()[n:] - d["lc"].to_numpy()[: n_rows - n]
        d[f"target_{n}"] = tgt
        tr = np.full(n_rows, np.nan)
        el = np.zeros(n_rows, dtype=bool)
        m = n_rows - 1 - n
        if m > 0:
            tr[:m] = lo[1 + n: 1 + n + m] - lo[1: 1 + m]
            el[:m] = traded[1: 1 + m] & traded[1 + n: 1 + n + m]
        d[f"trade_ret_{n}"] = tr
        d[f"eligible_{n}"] = el
    fold = np.full(n_rows, "", dtype=object)
    for fid, (start, end) in fold_windows.items():
        fold[((d["date"] >= start) & (d["date"] < end)).to_numpy()] = fid
    fold[(d["date"] >= fwd_start).to_numpy()] = "fwd"
    d["fold"] = fold
    d["half"] = [C.half_of(f) for f in fold]
    return d


def load(pair: str, fill: bool) -> pd.DataFrame:
    from freqtrade.data import history

    return history.load_pair_history(pair=pair, timeframe="15m", datadir=C.DATADIR, fill_up_missing=fill,
                                     data_format="feather", candle_type="spot")


def cache_data_ok(pair: str) -> pd.Series | None:
    from research.indicator_screen import harvest as H

    try:
        c = H.read_cache(pair, ["meta:data_ok"])
    except FileNotFoundError:
        return None
    return pd.Series(c["meta:data_ok"].to_numpy(), index=pd.to_datetime(c["date"]))


def check_folds() -> list[str]:
    """The fold windows must be exactly the plan's 15 contiguous 90-day windows."""
    f = C.folds()
    bad = [] if list(f) == [f"f{i:02d}" for i in range(15)] else ["fold ids differ from f00-f14"]
    ends = [e for _, e in f.values()]
    starts = [s for s, _ in f.values()]
    bad += [f"{a} -> {b} not contiguous" for a, b in zip(ends[:-1], starts[1:]) if a != b]
    if ends[-1] != C.FWD_START:
        bad.append(f"last fold ends {ends[-1]}, forward window starts {C.FWD_START}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair")
    a = ap.parse_args(argv)
    bad = check_folds()
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    for pair in [a.pair] if a.pair else C.pairs():
        filled, raw = load(pair, True), load(pair, False)
        d = build(filled, raw["date"], cache_data_ok(pair), C.folds())
        C.write_feather(d, C.FRAMES / f"{C.fname(pair)}.feather")
        sc = d[d.fold != ""]
        print(f"{pair:10s} rows {len(d):7d} scored {len(sc):7d} has_data {sc.has_data.mean():.3f} "
              f"data_ok {sc.data_ok.mean():.3f} eligible_1 {sc.eligible_1.mean():.3f}")
    return 0


def read(pair: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_feather(C.FRAMES / f"{C.fname(pair)}.feather", columns=columns)


if __name__ == "__main__":
    raise SystemExit(main())
