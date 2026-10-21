"""Donchian ensemble screen (preregistration.json; lit review 2026-10-21 candidate 3).

Daily vectorised simulation on the 10 wf pairs: six Donchian lookbacks averaged into a [0, 1] signal,
inverse-vol weights over the eligible pairs, cash remainder, 15 bp per unit traded. Primary arm = 0.02
no-trade band; comparator = monthly-rebalanced equal weight. No fitting, CPU only.

    PYTHONPATH=. ../../.venv/bin/python -m research.donchian_screen.screen
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(STRAT))

from research.xsec_screen.screen import committed, load_15m  # noqa: E402

PREREG = HERE / "preregistration.json"


def daily_ohlc(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(m.index.floor("D"))
    return pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last()})


def ensemble(d: pd.DataFrame, lookbacks: list[int]) -> pd.Series:
    """Mean over lookbacks of the Donchian long state at each day's close (uses days before t for the channel)."""
    states = []
    for L in lookbacks:
        up = d["high"].shift(1).rolling(L, min_periods=L).max()
        dn = d["low"].shift(1).rolling(math.ceil(L / 2), min_periods=math.ceil(L / 2)).min()
        s, st = [], 0.0
        for c, u, lo in zip(d["close"].to_numpy(), up.to_numpy(), dn.to_numpy()):
            if not np.isnan(u) and c >= u:
                st = 1.0
            elif not np.isnan(lo) and c <= lo:
                st = 0.0
            s.append(st)
        states.append(pd.Series(s, index=d.index))
    return pd.concat(states, axis=1).mean(axis=1)


def inv_vol_targets(sig: pd.DataFrame, vol: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    iv = (1.0 / vol).where(eligible)
    return (sig.where(eligible) * iv).div(iv.sum(axis=1), axis=0).fillna(0.0)


def simulate(target: pd.DataFrame, ret: pd.DataFrame, cost: float, band: float = 0.0,
             trade_days: pd.Series | None = None) -> pd.DataFrame:
    """target row t fills at the open of t+1 and earns ret row t+1 (open t+1 -> open t+2).
    Held weights drift with returns; a pair trades only if |target - held| > band; trade_days (bool by
    day) limits which days trade at all. Returns per-day net, gross, traded, exposure."""
    cols = target.columns
    w = np.zeros(len(cols))
    out = []
    tgt, R = target.to_numpy(float), ret[cols].fillna(0.0).to_numpy(float)
    td = None if trade_days is None else trade_days.reindex(target.index).fillna(False).to_numpy(bool)
    for k in range(1, len(target)):
        traded = 0.0
        if td is None or td[k]:
            new = tgt[k - 1]
            move = np.abs(new - w) > band
            traded = float(np.abs(new - w)[move].sum())
            w = np.where(move, new, w)
        c = cost * traded
        gross = float(w @ R[k])
        net = (1 - c) * (1 + gross) - 1
        out.append((target.index[k], net, gross, traded, float(w.sum())))
        grown = w * (1 + R[k])
        w = grown / (1 + gross) if 1 + gross > 0 else np.zeros_like(w)
    return pd.DataFrame(out, columns=["date", "net", "gross", "traded", "exposure"]).set_index("date")


def metrics(x: pd.DataFrame) -> dict:
    n = len(x)
    yrs = n / 365.0
    eq = (1 + x["net"]).cumprod()
    sd = x["net"].std()
    return {"days": n, "net_cagr_pct": round((float(eq.iloc[-1]) ** (1 / yrs) - 1) * 100, 2),
            "gross_cagr_pct": round((float((1 + x["gross"]).prod()) ** (1 / yrs) - 1) * 100, 2),
            "sharpe": round(float(x["net"].mean() / sd * np.sqrt(365)), 3) if sd > 0 else None,
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 2),
            "turnover_per_yr": round(float(x["traded"].sum() / yrs), 2),
            "mean_exposure": round(float(x["exposure"].mean()), 3)}


def _sharpe(r: np.ndarray) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(365)) if s > 0 else 0.0


def block_bootstrap_p(a: np.ndarray, b: np.ndarray, mean_block: int = 20, n: int = 5000, seed: int = 0) -> float:
    """Paired stationary bootstrap: share of draws where Sharpe(a) - Sharpe(b) <= 0 (one-sided p)."""
    rng, T = np.random.default_rng(seed), len(a)
    t = np.arange(T)
    le = 0
    for _ in range(n):
        new = rng.random(T) < 1.0 / mean_block
        new[0] = True
        first = np.maximum.accumulate(np.where(new, t, 0))  # start position of each day's block
        idx = (rng.integers(T, size=T)[first] + t - first) % T
        le += _sharpe(a[idx]) - _sharpe(b[idx]) <= 0
    return float(le / n)


def main() -> int:
    if not committed(PREREG):
        print("REFUSED: preregistration.json is not committed or has changes")
        return 2
    pr = json.loads(PREREG.read_text())
    sg, cost = pr["signal"], 0.0015
    daily = {p: daily_ohlc(load_15m(p)) for p in pr["universe"]}
    idx = sorted(set().union(*(d.index for d in daily.values())))
    idx = pd.DatetimeIndex(idx)
    L = sg["lookbacks_days"]
    sig = pd.DataFrame({p: ensemble(d, L) for p, d in daily.items()}).reindex(idx)
    close = pd.DataFrame({p: d["close"] for p, d in daily.items()}).reindex(idx)
    opn = pd.DataFrame({p: d["open"] for p, d in daily.items()}).reindex(idx)
    vol = np.log(close).diff().rolling(30, min_periods=20).std()
    age = close.notna().cumsum()
    eligible = (age > max(L)) & vol.notna() & close.notna()
    ret = opn.shift(-1) / opn - 1  # row t: open t -> open t+1
    e0, e1 = (pd.Timestamp(x, tz="UTC") for x in pr["halves"]["early"])
    l0, l1 = (pd.Timestamp(x, tz="UTC") for x in pr["halves"]["late"])
    span = (idx >= e0 - pd.Timedelta("1D")) & (idx < l1)
    tgt = inv_vol_targets(sig, vol, eligible)[span]
    ew = eligible.astype(float).div(eligible.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)[span]
    month_start = pd.Series(tgt.index.day == 1, index=tgt.index)
    books = {"band": simulate(tgt, ret, cost, band=0.02), "noband": simulate(tgt, ret, cost),
             "EW": simulate(ew, ret, cost, trade_days=month_start | (tgt.index == tgt.index[1]))}
    n_elig = eligible.sum(axis=1)
    out = {"prereg": "3e3a7777", "halves": {}, "inconclusive_reasons": []}
    for h, (a, b) in {"early": (e0, e1), "late": (l0, l1)}.items():
        thin = float((n_elig[(idx >= a) & (idx < b)] < 6).mean())
        if thin > 0.10:
            out["inconclusive_reasons"].append(f"{h}: <6 eligible pairs on {thin:.0%} of days")
        out["halves"][h] = {"thin_day_share": round(thin, 4)} | {
            k: metrics(v[(v.index >= a) & (v.index < b)]) for k, v in books.items()}
    pooled = {k: v[(v.index >= e0) & (v.index < l1)]["net"].to_numpy() for k, v in books.items()}
    p = block_bootstrap_p(pooled["band"], pooled["EW"])
    out["pooled"] = {"sharpe_band": round(_sharpe(pooled["band"]), 3), "sharpe_EW": round(_sharpe(pooled["EW"]), 3),
                     "bootstrap_p_one_sided": round(p, 4),
                     "corr_band_EW": round(float(np.corrcoef(pooled["band"], pooled["EW"])[0, 1]), 3)}
    H = out["halves"]
    ok = all(H[h]["band"]["sharpe"] is not None and H[h]["band"]["sharpe"] > 0
             and H[h]["band"]["sharpe"] > (H[h]["EW"]["sharpe"] or -np.inf) for h in H) and p <= 0.05
    out["verdict"] = "INCONCLUSIVE" if out["inconclusive_reasons"] else ("PASS (PROVISIONAL)" if ok else "FAIL")
    (HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
