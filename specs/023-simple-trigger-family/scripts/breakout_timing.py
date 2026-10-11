"""Step-1 diagnostic: forward outcome of a long entry vs bars since the most recent 20-bar breakout.
Causal conditioning only (everything known at bar t's close); outcomes are forward-looking BY DESIGN
(this is a screen, not a trigger). Test span 2023-01-01..2026-09-07; halves split at 2024-12-21."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from research.clfstudy.data import load_pair
PAIRS = ["ZEC","XRP","SOL","LINK","NEAR","AAVE","AVAX","LTC","BCH","DOT"]
LB, KMAX = 20, 16
H = {"4h": 16, "12h": 48, "24h": 96}
START, SPLIT, END = pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-12-21", tz="UTC"), pd.Timestamp("2026-09-07", tz="UTC")
rows = []
for p in PAIRS:
    df = load_pair(f"{p}/USDT")
    hi, lo, cl = (df[c].to_numpy(float) for c in ("high","low","close"))
    dt = df["date"]
    n = len(df)
    prior_high = pd.Series(hi).rolling(LB).max().shift(1).to_numpy()
    brk = cl > prior_high
    # contiguity: forward window must span exactly h bars of wall time
    tsec = ((dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).to_numpy().astype(np.int64)
    out = {}
    for name, h in H.items():
        f = np.full(n, np.nan); ok = np.zeros(n, bool)
        f[:-h] = cl[h:] / cl[:-h] - 1; ok[:-h] = (tsec[h:] - tsec[:-h]) == h * 900
        f[~ok] = np.nan; out[name] = f
    mae = pd.Series(lo[::-1]).rolling(26, min_periods=26).min().to_numpy()[::-1]   # min low over [t..t+25]
    mae = np.r_[mae[1:], np.nan] / cl - 1                                            # over t+1..t+26
    # bars since most recent breakout, the break level, and whether every close since held above it
    k = np.full(n, 10**6); held = np.zeros(n, bool); first = np.zeros(n, bool)
    last_b, level, minclose, last_any = -1, np.nan, np.inf, -10**6
    for t in range(n):
        if brk[t]:
            first[t] = (t - last_any) > LB          # first break after >= LB quiet bars = new episode
            last_any = t; last_b, level, minclose = t, prior_high[t], cl[t]
        if last_b >= 0:
            minclose = min(minclose, cl[t]); k[t] = t - last_b; held[t] = minclose > level
    d = pd.DataFrame({"pair": p, "date": dt, "k": k, "held": held, "first": first & brk,
                      "mae": mae, **out})
    rows.append(d[(d.date >= START) & (d.date < END)])
D = pd.concat(rows, ignore_index=True)
D["half"] = np.where(D.date < SPLIT, "early", "late")
base = D.groupby(["pair","half"])[list(H)+["mae"]].mean()

def table(sel, label):
    g = D[sel]
    m = g.groupby("half")[list(H)+["mae"]].mean()
    # excess over each pair's unconditional mean in the same half, averaged over bars
    ex = (g.set_index(["pair","half"])[list(H)+["mae"]] - base.reindex(g.set_index(["pair","half"]).index).to_numpy())
    ex = ex.reset_index()
    exh = ex.groupby("half")[list(H)+["mae"]].mean()*100
    # pair agreement on 12h excess sign
    pp = ex.groupby(["half","pair"])["12h"].mean()
    agree = {h: int((pp.loc[h] > 0).sum()) if h in pp.index.get_level_values(0) else 0 for h in ("early","late")}
    print(f"{label:28} n={len(g):7d} | excess 4h/12h/24h  early {exh.loc['early','4h']:+.3f} {exh.loc['early','12h']:+.3f} {exh.loc['early','24h']:+.3f}"
          f" | late {exh.loc['late','4h']:+.3f} {exh.loc['late','12h']:+.3f} {exh.loc['late','24h']:+.3f}"
          f" | MAE6.5h exc e/l {exh.loc['early','mae']:+.3f}/{exh.loc['late','mae']:+.3f} | pairs+ 12h e/l {agree['early']}/{agree['late']}")

print("excess = mean forward return minus the pair's unconditional mean (same half), in % ; pre-fee\n")
print("--- by bars since most recent breakout (k), held = every close since > broken level ---")
for kk in range(0, KMAX+1):
    table((D.k == kk), f"k={kk:2d} all")
    if kk > 0:
        table((D.k == kk) & D.held, f"k={kk:2d} held")
        table((D.k == kk) & ~D.held, f"k={kk:2d} failed")
table(D.k > 96, "no break in 24h")
print("\n--- k=0 split: first break of an episode vs a continuation break ---")
table(D["first"], "k=0 first break")
table((D.k == 0) & ~D["first"], "k=0 continuation")
