"""035 T006: build the per-pair input cache (research R2, R5, R6).

    PYTHONPATH=. ../../.venv/bin/python -m research.new_inputs.build [--pairs ZEC/USDT,...]

Writes `user_data/data/new_inputs/<PAIR>.parquet` (date + every `ni_*` column) and
`build_meta.json`. Every value is computed at the bar's close from trailing data only; values are
in [-1, 1] or NaN (the framework fills NaN with 0). `build_all(bundle, cut)` is the single entry
point, used both for the cache and by the truncation test (causality.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from research.new_inputs import inputs as I  # noqa: E402
from research.new_inputs import sources as S  # noqa: E402
from utils import RegimeState as RS  # noqa: E402


DATA = STRAT.parent / "data"
OHLCV_DIR = DATA / "binanceus"
EXTERNAL = DATA / "external"
OUT_DIR = DATA / "new_inputs"
KLINE_USE = ["volume", "quote_volume", "trades", "taker_buy_base", "close"]
METRIC_USE = ["sum_open_interest", "sum_toptrader_long_short_ratio", "taker_roll"]


# ---------------------------------------------------------------- normalisation (R1, R6)

def z(x: pd.Series) -> pd.Series:
    x = x.replace([np.inf, -np.inf], np.nan)
    m = x.rolling(I.Z_WINDOW, min_periods=I.Z_MIN).mean()
    s = x.rolling(I.Z_WINDOW, min_periods=I.Z_MIN).std()
    return ((x - m) / s.where(s > 0)).clip(-I.Z_CLIP, I.Z_CLIP) / I.Z_CLIP


def c(share: pd.Series) -> pd.Series:
    return 2.0 * share - 1.0


def _sum(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).sum()


# ---------------------------------------------------------------- raw definitions (R6)

def taker_share(k: pd.DataFrame, n: int = 4) -> pd.Series:
    v = _sum(k["volume"], n)
    return (_sum(k["taker_buy_base"], n) / v.where(v > 0)).clip(0, 1)


def trade_count_ratio(k: pd.DataFrame, n: int = 4, med: int = 96) -> pd.Series:
    x = _sum(k["trades"], n)
    return np.log((x + 1) / (x.rolling(med, min_periods=med).median() + 1))


def trade_size_ratio(k: pd.DataFrame, n: int = 4, med: int = 96) -> pd.Series:
    t = _sum(k["trades"], n)
    size = _sum(k["quote_volume"], n) / t.where(t > 0)
    return np.log(size / size.rolling(med, min_periods=med).median())


def global_us_volume(g: pd.DataFrame, u: pd.DataFrame, n: int = 4) -> pd.Series:
    return np.log((_sum(g["quote_volume"], n) + 1) / (_sum(u["quote_volume"], n) + 1))


def _log_ret(close: pd.Series, n: int) -> pd.Series:
    return np.log(close / close.shift(n))


# ---------------------------------------------------------------- per pair

def _pair_frames(b: dict, pair: str, dates: pd.DatetimeIndex) -> dict:
    gap = pd.Series(b["ohlcv"][pair]["volume"].to_numpy() == 0, index=dates)
    g = S.align_kline(b["global"].get(pair, _empty_k()), dates, KLINE_USE)
    # Binance.US: no row = no trades on the venue in that bar (zero, not unknown) ...
    u = S.align_kline(b["us"].get(pair, _empty_k()), dates, KLINE_USE).fillna(0.0)
    us_start = b["us"][pair]["open_time"].min() if len(b["us"].get(pair, [])) else None
    if us_start is None:
        u[:] = np.nan
    else:
        u[dates < us_start] = np.nan
    met = b["metrics"].get(pair)
    if met is not None and len(met):
        met = S.dedupe(met, "create_time")
        roll = met["sum_taker_long_short_vol_ratio"].rolling(12, min_periods=12).mean()
        met = met.assign(taker_roll=roll)
        m = S.align_snapshots(met, dates, METRIC_USE)
    else:
        m = pd.DataFrame(np.nan, index=dates, columns=METRIC_USE)
    p = S.align_kline(b["premium"].get(pair, _empty_k()), dates, ["close"])["close"]
    return {"gap": gap, "g": g, "u": u, "m": m, "p": p}


def _empty_k() -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=float) for k in ["open_time", *KLINE_USE]}).astype(
        {"open_time": "datetime64[ns, UTC]"})


def _market(b: dict) -> dict:
    """Cross-pair series on the union of dates (026 functions on gap-free frames)."""
    clean = {p: f.loc[f["volume"].to_numpy() > 0, ["date", "close"]] for p, f in b["ohlcv"].items()}
    out = {"volreg": RS.volreg(clean, min_pairs=I.MIN_PAIRS),
           "breadth": RS.breadth(clean, min_pairs=I.MIN_PAIRS)}
    r96 = RS._wide(clean, lambda s: _log_ret(s, 96))
    out["ret_median"] = RS._cross(r96, "median", I.MIN_PAIRS)
    btc = b["btc"]
    bc = pd.Series(btc["close"].to_numpy(float), index=pd.DatetimeIndex(btc["date"]))
    out["btc_vol"] = np.log(bc).diff().rolling(96, min_periods=96).std()
    import talib

    adx = talib.ADX(btc["high"].to_numpy(float), btc["low"].to_numpy(float),
                    btc["close"].to_numpy(float), timeperiod=14)
    out["btc_adx"] = pd.Series(adx, index=bc.index) / 100.0
    return out


def build_pair(b: dict, pair: str, mk: dict) -> pd.DataFrame:
    o = b["ohlcv"][pair]
    dates = pd.DatetimeIndex(o["date"])
    f = _pair_frames(b, pair, dates)
    close = pd.Series(o["close"].to_numpy(float), index=dates)
    oi_chg = np.log(f["m"]["sum_open_interest"] / f["m"]["sum_open_interest"].shift(16))
    tt = np.log(f["m"]["sum_toptrader_long_short_ratio"])
    lv = f["p"].rolling(4, min_periods=4).mean()
    own_r96 = pd.Series(np.log(close / close.shift(96)).to_numpy(), index=dates)
    raw = {
        "OF_TBG": taker_share(f["g"]),
        "OF_TRG": trade_count_ratio(f["g"]),
        "OF_SZG": trade_size_ratio(f["g"]),
        "OF_GUS": global_us_volume(f["g"], f["u"]),
        "OF_TBU": taker_share(f["u"]).where(~f["gap"]),
        "OF_TRU": trade_count_ratio(f["u"]).where(~f["gap"]),
        "PO_OI": oi_chg,
        "PO_FL": np.maximum(0.0, -oi_chg) * np.maximum(0.0, -_log_ret(close, 16)),
        "PO_TT": tt - tt.shift(16),
        "PO_TK": np.log(f["m"]["taker_roll"]),
        "MS_VR": mk["volreg"].reindex(dates),
        "MS_BR": mk["breadth"].reindex(dates),
        "MS_BV": mk["btc_vol"].reindex(dates),
        "MS_BA": mk["btc_adx"].reindex(dates),
        "MS_RR": own_r96 - mk["ret_median"].reindex(dates),
        "BA_LV": lv,
        "BA_CH": lv - lv.shift(16),
    }
    out = {"date": dates}
    for inp in I.TABLE:
        x = raw[inp.tag].astype(float).replace([np.inf, -np.inf], np.nan)
        v = z(x) if inp.norm == "z" else c(x)
        out[I.column(inp.tag)] = v.to_numpy()
        if inp.missing_flag:
            out[I.column(inp.tag) + "_missing"] = v.isna().astype(float).to_numpy()
    return pd.DataFrame(out).reset_index(drop=True)


def truncate(b: dict, cut: pd.Timestamp) -> dict:
    """The bundle as it was at the close of bar `cut` (research R3)."""
    tk = S.truncate_klines
    return {"ohlcv": {p: f[f["date"] <= cut].reset_index(drop=True) for p, f in b["ohlcv"].items()},
            "btc": b["btc"][b["btc"]["date"] <= cut].reset_index(drop=True),
            "global": {p: tk(f, cut) for p, f in b["global"].items()},
            "us": {p: tk(f, cut) for p, f in b["us"].items()},
            "premium": {p: tk(f, cut) for p, f in b["premium"].items()},
            "metrics": {p: S.truncate_snapshots(f, cut) for p, f in b["metrics"].items()}}


def build_all(b: dict, cut: pd.Timestamp | None = None) -> dict:
    if cut is not None:
        b = truncate(b, cut)
    mk = _market(b)
    return {p: build_pair(b, p, mk) for p, f in b["ohlcv"].items() if len(f)}


# ---------------------------------------------------------------- disk

def _load_ohlcv(pair: str) -> pd.DataFrame:
    df = pd.read_feather(OHLCV_DIR / f"{I.file_stem(pair)}-15m.feather")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def _zips(source: str, pair: str) -> list[Path]:
    d = EXTERNAL / source / I.symbol(pair)
    return sorted(d.glob("*.zip")) if d.is_dir() else []


def _concat(frames: list[pd.DataFrame], key: str) -> pd.DataFrame:
    frames = [f for f in frames if len(f)]
    return S.dedupe(pd.concat(frames, ignore_index=True), key) if frames else pd.DataFrame()


def load_bundle(pairs=I.PAIRS) -> dict:
    """Every source for every pair, parsed from the verified local archives."""
    from research.new_inputs import fetch as FE

    ok = FE.verified_paths()
    b = {"ohlcv": {}, "global": {}, "us": {}, "metrics": {}, "premium": {}}
    for p in pairs:
        b["ohlcv"][p] = _load_ohlcv(p)
        b["global"][p] = _concat([S.read_klines_zip(z) for z in _zips("global_spot", p)
                                  if str(z) in ok], "open_time")
        b["premium"][p] = _concat([S.read_klines_zip(z) for z in _zips("um_premium", p)
                                   if str(z) in ok], "open_time")
        b["metrics"][p] = _concat([S.read_metrics_zip(z) for z in _zips("um_metrics", p)
                                   if str(z) in ok], "create_time")
        us = EXTERNAL / "us_spot" / I.symbol(p) / "klines.jsonl"
        b["us"][p] = S.dedupe(S.read_us_jsonl(us), "open_time") if us.is_file() else _empty_k()
    b["btc"] = _load_ohlcv(I.CONTEXT)
    return b


def code_sha() -> str:
    h = hashlib.sha256()
    for f in ("build.py", "sources.py", "inputs.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def main(argv=None) -> int:
    from research.new_inputs import fetch as FE

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="comma-separated; default all (market state needs all)")
    a = ap.parse_args(argv)
    pairs = tuple(a.pairs.split(",")) if a.pairs else I.PAIRS
    b = load_bundle(pairs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    built = build_all(b)
    for p, df in built.items():
        df.to_parquet(OUT_DIR / f"{I.file_stem(p)}.parquet", index=False)
        vals = [k for k in df.columns if k.startswith("ni_") and not k.endswith("_missing")]
        print(p, len(df), {k: round(float(df[k].notna().mean()), 3) for k in vals})
    meta = {"table_sha": I.TABLE_SHA, "code_sha": code_sha(), "manifest_sha": FE.manifest_sha(),
            "pairs": list(built), "built_at": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT_DIR / "build_meta.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
