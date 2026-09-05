"""Characterise the control model's OFF-TRIGGER entries.

TrigGate showed that forcing the label's trigger at inference costs 7.30pp: the
model enters on bars the trigger REJECTS and those entries are profitable. This
joins the control arm's trades back to the indicator data to describe them.

trigger = (guard_metric < -0.2) AND (bb_width > 0.035)   -- labels_gbb defaults
"""
import sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "user_data/strategies")
BT = Path("user_data/backtest_results")
DATA = Path("user_data/data/binanceus")
from utils.DataframePopulator import DataframePopulator, DatasetType

GUARD_T, BBW_T = -0.2, 0.035
WANT = {431: "W1", 763: "W2", 295: "W3"}
STRAT = "NNNC_MLX_Lbl02"

def load_arch(n):
    for p in sorted(BT.glob("*.zip"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            z = zipfile.ZipFile(p)
            main = [x for x in z.namelist() if x.endswith(".json") and "config" not in x and "market" not in x]
            d = json.loads(z.read(main[0]))
            if STRAT in d.get("strategy", {}) and len(d["strategy"][STRAT]["trades"]) == n:
                return d["strategy"][STRAT]["trades"]
        except Exception:
            continue
    return None

COLS = ["guard_metric","bb_width","rsi","adx","rvol","close_norm","bb_position",
        "atr_pct_roll","mfi_scaled","fisher_wr"]
cache = {}
def indicators(pair):
    if pair not in cache:
        f = DATA / f"{pair.replace('/','_')}-15m.feather"
        df = pd.read_feather(f)
        df = DataframePopulator().add_indicators(df, dataset_type=DatasetType.MINIMAL)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        cache[pair] = df.set_index("date")
    return cache[pair]

rows = []
for n, w in WANT.items():
    tr = load_arch(n)
    if tr is None:
        print(f"  {w}: archive with {n} trades not found"); continue
    for t in tr:
        try:
            df = indicators(t["pair"])
        except Exception:
            continue
        ts = pd.to_datetime(t["open_date"], utc=True)
        if ts not in df.index:
            idx = df.index.searchsorted(ts)
            if idx >= len(df): continue
            ts = df.index[idx]
        r = df.loc[ts]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        rec = {c: float(r.get(c, np.nan)) for c in COLS}
        rec.update(window=w, pair=t["pair"], profit=t["profit_abs"],
                   ratio=t["profit_ratio"], dur=t["trade_duration"],
                   exit=t["exit_reason"])
        rows.append(rec)

d = pd.DataFrame(rows)
d["trigger"] = (d["guard_metric"] < GUARD_T) & (d["bb_width"] > BBW_T)
d["win"] = d["profit"] > 0
print(f"joined {len(d)} trades across {d['window'].nunique()} windows\n")

print("=== ON- vs OFF-TRIGGER ===")
g = d.groupby("trigger")
print(f"{'group':<14}{'n':>6}{'win%':>8}{'mean P&L':>11}{'sum P&L':>11}{'mean hrs':>10}")
for k, s in g:
    lab = "ON-trigger" if k else "OFF-trigger"
    print(f"{lab:<14}{len(s):>6}{100*s['win'].mean():>7.1f}%{s['profit'].mean():>11.2f}"
          f"{s['profit'].sum():>11.1f}{s['dur'].mean()/60:>10.1f}")

off = d[~d["trigger"]]
print(f"\n=== WHY are the off-trigger bars rejected? (n={len(off)}) ===")
fail_g = (off["guard_metric"] >= GUARD_T)
fail_b = (off["bb_width"] <= BBW_T)
print(f"  guard_metric >= {GUARD_T} only : {(fail_g & ~fail_b).sum():>5}")
print(f"  bb_width <= {BBW_T} only       : {(~fail_g & fail_b).sum():>5}")
print(f"  BOTH fail                      : {(fail_g & fail_b).sum():>5}")

print(f"\n=== indicator profile at entry (median) ===")
print(f"{'indicator':<16}{'ON-trigger':>12}{'OFF-trigger':>13}{'OFF profitable':>16}")
offw = off[off["win"]]
for c in COLS:
    print(f"{c:<16}{d[d['trigger']][c].median():>12.3f}{off[c].median():>13.3f}{offw[c].median():>16.3f}")

print(f"\n=== exit reasons ===")
print(f"{'reason':<22}{'ON n':>7}{'ON meanP':>10}{'OFF n':>8}{'OFF meanP':>11}")
for r in sorted(set(d["exit"])):
    on = d[(d["trigger"]) & (d["exit"]==r)]; of = off[off["exit"]==r]
    print(f"{r:<22}{len(on):>7}{(on['profit'].mean() if len(on) else 0):>10.2f}"
          f"{len(of):>8}{(of['profit'].mean() if len(of) else 0):>11.2f}")
