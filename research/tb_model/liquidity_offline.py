"""Offline liquidity buckets (informal): rvol, qv / trailing 30-day median, within-pair qv tercile.
    PYTHONPATH=.:evaluation python research/tb_model/liquidity_offline.py cs_aro st_g24_mom"""
import sys
import numpy as np, pandas as pd
sys.path[:0] = [".", "evaluation"]
from research.exit_coupling import feasibility as F
from research.clfstudy.data import load_pair
EARLY = {f"f{i:02d}" for i in range(8)}
_D = {}
def sig(pair, ts):
    if pair not in _D:
        d = load_pair(pair); qv = pd.Series((d.volume * d.close).to_numpy(float), index=pd.DatetimeIndex(d.date))
        _D[pair] = pd.DataFrame({"rvol": (d.volume / d.volume.ewm(span=10, adjust=False).mean()).to_numpy(), "qv": qv,
                                 "rel": qv / qv.rolling(2880, min_periods=2880).median()}, index=qv.index)
    s = _D[pair]; return s.iloc[s.index.searchsorted(ts) - 1]
def line(name, x, n):
    turn = (x.stake * (2 + x.pr)).sum(); e, l = x[x.early], x[~x.early]
    s3 = np.minimum(3000 * x.w, x.qv / 10)
    return (f"{name:24s} {len(x):5d} ({len(x)/n:4.0%}) pnl {x.pa.sum():7.0f} (e {e.pa.sum():6.0f} / l {l.pa.sum():6.0f}) "
            f"mean pr {x.pr.mean()*100:6.3f}% BE/side {100*x.pa.sum()/turn:6.3f}% med cap {(x.qv/10).median():6.0f} L3000 {(s3*x.pr).sum():7.0f}")
for tag in sys.argv[1:]:
    rows = []
    for t in F.archive_trades(tag):
        s = sig(t["pair"], pd.Timestamp(t["open_date"]))
        tg = str(t["enter_tag"]); w = 4/3 if tg.endswith(":agree") else (2/3 if tg.endswith(":disagree") else 1.0)
        rows.append({"pair": t["pair"], "fold": t["fold"], "stake": float(t["stake_amount"]), "pr": float(t["profit_ratio"]),
                     "pa": float(t["profit_abs"]), "rvol": s.rvol, "qv": s.qv, "rel": s.rel, "w": w})
    d = pd.DataFrame(rows); d["early"] = d.fold.isin(EARLY); n = len(d)
    print(f"\n=== {tag}: {n} trades, median rel {d.rel.median():.2f}, median rvol {d.rvol.median():.2f}")
    print(line("all", d, n))
    for th in (1.5, 2.0): print(line(f"rvol >= {th}", d[d.rvol >= th], n))
    print("-- causal: qv / pair's trailing 30-day median")
    for th in (0.5, 1.0, 1.5, 2.0, 3.0): print(line(f"rel > {th}", d[d.rel > th], n))
    for i, (k, g) in enumerate(d.groupby(pd.qcut(d.rel.replace(np.inf, 1e9).fillna(-1).rank(method="first"), 4))): print(line(f"rel quartile {i+1} (med {g.rel.median():.1f})", g, n))
    d["t"] = d.groupby("pair").qv.transform(lambda s: pd.qcut(s, 3, labels=False))
    print("-- within-pair tercile (full sample, lookahead)")
    for k, g in d.groupby("t"): print(line(f"pair tercile {int(k)+1}", g, n))
