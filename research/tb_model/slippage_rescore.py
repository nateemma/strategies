"""Post-hoc slippage (informal, not pre-registered; run: PYTHONPATH=.:evaluation python -m research.tb_model.slippage_rescore) on existing seed-1 records: cost = s x (entry notional + exit notional) per trade.
Flat: s fixed per side. Impact: s = k x sqrt(participation), participation = stake / candle quote volume."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
STRAT = Path("/Users/philprice95/projects/freqtrade/user_data/strategies")
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]
from evaluation.pool import pool
from research.nnnc_switches import judge as J
from research.tb_model.lev2 import _cap

pr = json.loads((STRAT / "research/tb_model/preregistration_all.json").read_text())
E, L = set(pr["early"]), set(pr["late"])
BOOKS = ["cs_aro", "cs_aro_hp", "cs_aro_lev2", "cs_aro_hpany", "cs_aro_hpany_lev2", "cs_aro_all2k", "cs_aro_all3k", "cs_aro_all3k_p5"]
data = {}
for b in BOOKS:
    p = pool(J._plan(b))
    rows = []
    for t in p["trades"]:
        fills = t["entry_fills"] or [(None, float(t["stake_amount"]))]
        ent = sum(s for _, s in fills)
        ext = float(t["stake_amount"]) * (1 + float(t["profit_ratio"] or 0)) if t["entry_fills"] is None else ent + float(t["profit_abs"])
        cap = _cap(t["pair"], pd.Timestamp(t["open_date"]))
        part = [s / (cap * 10) for _, s in fills]  # cap = quote volume / 10
        rows.append({"f": t["fold_id"], "sc": t["scale_pp_per_abs"], "ent": ent, "ext": ext,
                     "imp_ent": sum(s * np.sqrt(q) for (_, s), q in zip(fills, part)),
                     "imp_ext": ext * np.sqrt(max(part))})
    d = pd.DataFrame(rows)
    base = {f["fold_id"]: f["return_pct"] or 0.0 for f in p["per_fold"]}
    data[b] = (d, base, float(np.median([q for r in rows for q in [r["ent"]]])), )
    print(b, "trades", len(d), "median participation",
          round(float(np.median([(r["imp_ent"] / r["ent"]) ** 2 for r in rows])), 4), file=sys.stderr)

def score(b, s=0.0, k=0.0):
    d, base, _ = data[b]
    cost = (s * (d.ent + d.ext) + k * (d.imp_ent + d.imp_ext)) * d.sc * 100 / 100  # sc: pp per abs
    c = cost.groupby(d.f).sum()
    adj = {f: base[f] - c.get(f, 0.0) for f in base}
    return (sum(v for f, v in adj.items() if f in E), sum(v for f, v in adj.items() if f in L))

print("sanity cs_aro_all3k s=0:", [round(x, 2) for x in score("cs_aro_all3k")])
print("\nFLAT per-side slippage -> total (early / late)")
print("book".ljust(20), *[f"{s*100:.2f}%".rjust(20) for s in (0, .0005, .001, .002, .003)])
for b in BOOKS:
    print(b.ljust(20), *[f"{sum(score(b, s)):7.2f} ({score(b,s)[0]:6.2f}/{score(b,s)[1]:6.2f})".rjust(20) for s in (0, .0005, .001, .002, .003)])
for b in BOOKS:
    lo, hi = 0, 0.05
    for _ in range(50):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if sum(score(b, m)) > 0 else (lo, m)
    print(f"break-even per side {b}: {lo*100:.3f}%")
lo, hi = 0, 0.05
for _ in range(50):
    m = (lo + hi) / 2
    lo, hi = (m, hi) if sum(score("cs_aro_all3k", m)) > sum(score("cs_aro", m)) else (lo, m)
print(f"all3k beats cs_aro up to flat {lo*100:.3f}% per side")
print("\nIMPACT s = k*sqrt(participation) per side")
for k in (0.01, 0.02, 0.05):
    print(f"k={k}", {b: round(sum(score(b, 0, k)), 2) for b in BOOKS})
