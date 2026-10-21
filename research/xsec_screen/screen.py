"""Cross-sectional rank-IC screen (preregistration.json; literature review 2026-10-21, candidates 1-2).

Daily signals per pair from 15m candles (causal: day t uses bars up to the end of day t), Spearman IC
across pairs vs the forward h-day log return, sampled every h days (non-overlapping), per half.
No training, CPU only.

    PYTHONPATH=. ../../.venv/bin/python -m research.xsec_screen.screen
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = STRAT.parent / "data" / "binanceus"  # config_wf's exchange
PREREG = HERE / "preregistration.json"


def load_15m(pair: str) -> pd.DataFrame:
    d = pd.read_feather(DATA / f"{pair.replace('/', '_')}-15m.feather")
    d["date"] = pd.to_datetime(d["date"], utc=True)
    return d.set_index("date").sort_index()


def daily_features(m: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day: close and the 15m-bar sums the signals need (volume == 0 bars excluded)."""
    r = np.log(m["close"]).diff()
    live = m["volume"] > 0
    r_ok = r.where(live)
    qv = (m["close"] * m["volume"]).where(live)
    day = m.index.floor("D")
    g = pd.DataFrame({
        "close": m["close"], "sq": r_ok ** 2, "sq_up": (r_ok.clip(lower=0)) ** 2,
        "sq_dn": (r_ok.clip(upper=0)) ** 2, "illiq": r_ok.abs() / qv.replace(0, np.nan), "n": r_ok.notna(),
    }).groupby(day)
    return pd.DataFrame({"close": g["close"].last(), "sq": g["sq"].sum(), "sq_up": g["sq_up"].sum(),
                         "sq_dn": g["sq_dn"].sum(), "illiq_sum": g["illiq"].sum(), "n": g["n"].sum()})


def signals(d: pd.DataFrame, btc_ret: pd.Series) -> pd.DataFrame:
    """Every pre-registered signal for one pair, indexed by day; value at t uses days <= t only."""
    lc = np.log(d["close"])
    r = lc.diff()
    b = btc_ret.reindex(d.index)
    cov = r.rolling(60, min_periods=40).cov(b)
    beta = cov / b.rolling(60, min_periods=40).var()
    res = r - beta * b
    rv = d["sq"].rolling(7, min_periods=6).sum()
    up = d["sq_up"].rolling(7, min_periods=6).sum()
    dn = d["sq_dn"].rolling(7, min_periods=6).sum()
    out = pd.DataFrame(index=d.index)
    for k in (1, 3, 7, 14, 90):
        out[f"mom_{k}d"] = lc - lc.shift(k)
    out["resmom_7d"] = res.rolling(7, min_periods=7).sum()
    out["resmom_14d"] = res.rolling(14, min_periods=14).sum()
    out["amihud_14d"] = d["illiq_sum"].rolling(14, min_periods=12).sum() / d["n"].rolling(14, min_periods=12).sum()
    out["rv_7d"] = rv
    out["rs_up_7d"] = up
    out["sj_7d"] = (up - dn) / rv
    return out


def forward(d: pd.DataFrame, h: int) -> pd.Series:
    lc = np.log(d["close"])
    return lc.shift(-h) - lc


def ic_series(sig: pd.DataFrame, fwd: pd.DataFrame, days: pd.DatetimeIndex, min_pairs: int = 6) -> pd.Series:
    vals = {}
    for t in days:
        s, f = sig.loc[t], fwd.loc[t]
        ok = s.notna() & f.notna()
        if ok.sum() >= min_pairs:
            vals[t] = s[ok].rank().corr(f[ok].rank())
    return pd.Series(vals, dtype=float)


def _t(x: pd.Series) -> float | None:
    x = x.dropna()
    return None if len(x) < 3 or x.std() == 0 else float(x.mean() / x.std() * np.sqrt(len(x)))


def top_book(sig: pd.DataFrame, fwd: pd.DataFrame, days, k: int = 3, cost_one_way: float = 0.0015) -> dict:
    """Long the top k (sig already sign-adjusted), equal weight, rebalanced on `days`; vs equal weight all."""
    gross, net, ew, prev = [], [], [], set()
    for t in days:
        s, f = sig.loc[t], fwd.loc[t]
        ok = s.notna() & f.notna()
        if ok.sum() < 6:
            continue
        top = set(s[ok].nlargest(k).index)
        ret = float(np.expm1(f[list(top)]).mean())
        turn = len(top - prev) / k if prev else 1.0
        gross.append(ret)
        net.append(ret - 2 * cost_one_way * turn)  # sell the leavers + buy the joiners
        ew.append(float(np.expm1(f[ok]).mean()))
        prev = top
    g, n, e = (float(np.prod(1 + np.array(x)) - 1) * 100 if x else None for x in (gross, net, ew))
    return {"gross_pct": g, "net_pct": n, "ew_pct": e,
            "net_minus_ew_pp": None if n is None else n - e, "periods": len(gross)}


def committed(path: Path) -> bool:
    rel = path.relative_to(STRAT)
    run = lambda *a: subprocess.run(["git", "-C", str(STRAT), *a], capture_output=True, text=True).stdout  # noqa: E731
    return bool(run("log", "-1", "--format=%H", "--", str(rel)).strip()) and not run("status", "--porcelain", "--", str(rel)).strip()


def main() -> int:
    if not committed(PREREG):
        print("REFUSED: preregistration.json is not committed or has changes")
        return 2
    pr = json.loads(PREREG.read_text())
    btc = daily_features(load_15m(pr["btc"]))
    btc_ret = np.log(btc["close"]).diff()
    per = {p: daily_features(load_15m(p)) for p in pr["pairs"]}
    sigs = {p: signals(d, btc_ret) for p, d in per.items()}
    halves = {h: (pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for h, (a, b) in
              (("early", pr["early"]), ("late", pr["late"]))}
    rule = pr["rule"]
    out = {}
    for name, spec in pr["signals"].items():
        sgn = 1.0 if spec["sign"] == "+" else -1.0
        S = pd.DataFrame({p: sigs[p][name] for p in pr["pairs"]}) * sgn
        for h in pr["horizons_days"]:
            F = pd.DataFrame({p: forward(per[p], h) for p in pr["pairs"]})
            row = {"signal": name, "h": h, "baseline": bool(spec.get("baseline"))}
            ics = []
            for half, (a, b) in halves.items():
                days = S.index[(S.index >= a) & (S.index < b)][::h]
                ic = ic_series(S, F, days)
                ics.append(ic)
                row[half] = {"mean_ic": float(ic.mean()) if len(ic) else None, "t": _t(ic), "n": len(ic),
                             "book": top_book(S, F, days)}
            pooled = pd.concat(ics)
            row["pooled"] = {"mean_ic": float(pooled.mean()), "t": _t(pooled), "n": len(pooled)}
            te, tl, tp = row["early"]["t"], row["late"]["t"], row["pooled"]["t"]
            row["pass"] = (None if row["baseline"] else bool(
                te is not None and tl is not None and tp is not None and te > 0 and tl > 0
                and tp >= 3.0 and te >= 1.5 and tl >= 1.5))
            out[f"{name}@{h}d"] = row
    (HERE / "results.json").write_text(json.dumps({"rule": rule, "results": out}, indent=1, default=str))
    fmt = lambda v: "   -  " if v is None else f"{v:+6.2f}"  # noqa: E731
    print(f"{'signal@h':16s} {'IC e':>7s} {'t e':>6s} {'IC l':>7s} {'t l':>6s} {'t all':>6s}  {'net-ew e':>8s} {'net-ew l':>8s}  pass")
    for k, r in out.items():
        print(f"{k:16s} {r['early']['mean_ic']:+7.3f} {fmt(r['early']['t'])} {r['late']['mean_ic']:+7.3f} "
              f"{fmt(r['late']['t'])} {fmt(r['pooled']['t'])}  {fmt(r['early']['book']['net_minus_ew_pp'])}   "
              f"{fmt(r['late']['book']['net_minus_ew_pp'])}  {'BASE' if r['baseline'] else r['pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
