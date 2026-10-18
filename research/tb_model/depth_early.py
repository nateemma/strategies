"""Early, informal read of 036 depth (NOT the gated cost_model fit): walk the recorded book for the
stakes the backtest books would place at each snapshot. Cap = quote volume of the previous closed
15m candle / 10 (BaseStrategy._fillable_stake)."""
import glob, gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/philprice95/projects/freqtrade/user_data/strategies")
from research.live_pnl.depth import walk_cost_bps
D = "/Users/philprice95/projects/freqtrade/user_data/data/live_nnnc/live1/"
snaps = [json.loads(l) for f in sorted(glob.glob(D + "depth-*.jsonl.gz")) for l in gzip.open(f, "rt")]
qv = {}
for f in glob.glob(D + "signals-*.jsonl"):
    for l in open(f):
        s = json.loads(l); qv[(s["pair"], pd.Timestamp(s["candle"]))] = s["quote_volume"]
rows = []
for s in snaps:
    if s.get("depth_missing") or s["trigger"] != "schedule" or not s["bids"] or not s["asks"]:
        continue
    t = pd.Timestamp(s["time"]).floor("15min")
    q = qv.get((s["pair"], t - pd.Timedelta("15min")))
    if q is None:
        continue
    mid = s["mid"]
    r = {"pair": s["pair"], "t": t, "qv": q, "spread_bps": (s["asks"][0][0] / s["bids"][0][0] - 1) * 1e4,
         "depth20_ask": sum(p * a for p, a in s["asks"]), "depth20_bid": sum(p * a for p, a in s["bids"])}
    for name, stake in {"cap10": q / 10, "cap5": q / 20, "cs_aro": min(900, q / 10), "all3k": min(3000, q / 10),
                        "all3k_agree": min(4000, q / 10), "fixed300": 300, "fixed900": 900, "fixed3000": 3000}.items():
        if stake <= 0:
            continue
        b, xb = walk_cost_bps(s["asks"], mid, stake, "buy")
        sl, xs = walk_cost_bps(s["bids"], mid, stake, "sell")
        r[name] = (b + sl) / 2; r[name + "_stake"] = stake; r[name + "_x"] = xb or xs
    rows.append(r)
d = pd.DataFrame(rows)
print("snapshots", len(d), "pairs", d.pair.nunique(), d.t.min(), "->", d.t.max())
print("\nper pair medians (bps per side vs mid)")
cols = ["qv", "spread_bps", "depth20_ask", "cap10_stake", "cap10", "cap5", "cs_aro", "all3k", "fixed900", "fixed3000"]
print(d.groupby("pair")[cols].median().round(1).to_string())
print("\nall pairs: median / mean / p90 bps per side")
for c in ["spread_bps", "cap10", "cap5", "cs_aro", "all3k", "all3k_agree", "fixed300", "fixed900", "fixed3000"]:
    v = d[c] if c == "spread_bps" else d[c].dropna()
    x = "" if c == "spread_bps" else f"  extrapolated {d[c + '_x'].mean():.0%}"
    print(f"{c:12s} {v.median():7.1f} {v.mean():7.1f} {v.quantile(.9):7.1f}{x}")
print("\nshare of snapshots where cap binds for all3k (qv/10 < 3000):", round((d.qv / 10 < 3000).mean(), 3))

