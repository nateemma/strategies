import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from research.clfstudy.data import load_pair
S = sys.argv[1]
K = {}
for p in ["ZEC","XRP","SOL","LINK","NEAR","AAVE","AVAX","LTC","BCH","DOT"]:
    df = load_pair(f"{p}/USDT"); hi = df.high.to_numpy(float); cl = df.close.to_numpy(float)
    ph = pd.Series(hi).rolling(20).max().shift(1).to_numpy(); brk = cl > ph
    k = np.full(len(df), 10**6); held = np.zeros(len(df), bool); lb, lvl, mc = -1, np.nan, np.inf
    for t in range(len(df)):
        if brk[t]: lb, lvl, mc = t, ph[t], cl[t]
        if lb >= 0: mc = min(mc, cl[t]); k[t] = t - lb; held[t] = mc > lvl
    K[f"{p}/USDT"] = pd.DataFrame({"k": k, "held": held}, index=df.date)
EARLY = {f"f{i:02d}" for i in range(8)}
for arm in sys.argv[2:]:
    tr = json.load(open(f"{S}/pooled_{arm}.json"))["trades"]
    rows = []
    for r in tr:
        sig = pd.Timestamp(r["open_date"]) - pd.Timedelta(minutes=15)   # signal bar = bar before entry
        x = K[r["pair"]].loc[sig]
        b = "k=0 (break bar)" if x.k == 0 else ("k1-8 held" if x.k <= 8 and x.held else ("k1-8 failed" if x.k <= 8 else "no break <=8 bars"))
        rows.append((b, r["fold_id"] in EARLY, r["profit_ratio"]))
    d = pd.DataFrame(rows, columns=["bucket", "early", "pr"])
    print(f"--- {arm}: {len(d)} trades")
    g = d.groupby("bucket").agg(n=("pr","size"), mean_pct=("pr", lambda s: s.mean()*100), sum_pct=("pr", lambda s: s.sum()*100))
    e = d[d.early].groupby("bucket").pr.mean()*100; l = d[~d.early].groupby("bucket").pr.mean()*100
    g["early_mean"] = e; g["late_mean"] = l; g["share"] = g.n / len(d)
    print(g.round(3).to_string()); print()
