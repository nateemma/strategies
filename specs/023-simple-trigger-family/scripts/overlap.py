"""Price gbb/breakout_gbb overlap entries under reversion (ARM_D) vs momentum (no TP + 24h) exits."""
import json, sys, random
S = sys.argv[1]
def load(t):
    d = json.load(open(f"{S}/pooled_{t}.json"))
    return {(r["fold_id"], r["pair"], r["open_date"]): r for r in d["trades"]}
A = {t: load(t) for t in ("st_gbb", "st_bgo", "st_gbb_mx", "st_bgo_mx")}
EARLY = {f"f{i:02d}" for i in range(8)}

def summarise(name, keys, rev, mom):
    keys = sorted(k for k in keys if k in rev and k in mom)
    if not keys: print(name, "no keys"); return
    d = [(k[0], rev[k]["profit_ratio"], mom[k]["profit_ratio"]) for k in keys]
    def stats(rows):
        n = len(rows); r = sum(x[1] for x in rows) / n; m = sum(x[2] for x in rows) / n
        return n, r, m, m - r
    # fold-blocked bootstrap of the mean paired difference (momentum - reversion)
    folds = sorted({x[0] for x in d}); byf = {f: [x for x in d if x[0] == f] for f in folds}
    rng = random.Random(0); boots = []
    for _ in range(5000):
        rows = [x for f in rng.choices(folds, k=len(folds)) for x in byf[f]]
        boots.append(stats(rows)[3])
    boots.sort(); lo, hi = boots[124], boots[4874]
    p = 2 * min(sum(b <= 0 for b in boots), sum(b >= 0 for b in boots)) / len(boots)
    n, r, m, diff = stats(d)
    print(f"{name}: n={n}  reversion exits {r*100:+.3f}%/trade  momentum exits {m*100:+.3f}%/trade  "
          f"diff {diff*100:+.3f}pp/trade  95% CI [{lo*100:+.3f}, {hi*100:+.3f}]  p={p:.3f}")
    for lab, sel in (("early", lambda x: x[0] in EARLY), ("late", lambda x: x[0] not in EARLY)):
        rows = [x for x in d if sel(x)]
        if rows:
            n2, r2, m2, d2 = stats(rows)
            print(f"   {lab:5} n={n2:4d}  rev {r2*100:+.3f}  mom {m2*100:+.3f}  diff {d2*100:+.3f}")
    wr = sum(x[1] > 0 for x in d) / n; wm = sum(x[2] > 0 for x in d) / n
    print(f"   win rate rev {wr:.1%}  mom {wm:.1%};  sum over trades rev {sum(x[1] for x in d)*100:+.1f}%  mom {sum(x[2] for x in d)*100:+.1f}%")

ov_rev = set(A["st_gbb"]) & set(A["st_bgo"])          # both triggers fired, ARM_D exits
ov_mom = set(A["st_gbb_mx"]) & set(A["st_bgo_mx"])    # both triggers fired, momentum exits
print(f"overlap entries: under ARM_D exits {len(ov_rev)}, under momentum exits {len(ov_mom)}, in both {len(ov_rev & ov_mom)}")
print(f"share of gbb trades: {len(ov_rev)/len(A['st_gbb']):.1%};  of breakout_gbb trades: {len(ov_rev)/len(A['st_bgo']):.1%}")
# sanity: the same entry under the same exits should give the same per-trade return in both arms
same = [k for k in ov_rev if abs(A['st_gbb'][k]['profit_ratio'] - A['st_bgo'][k]['profit_ratio']) > 1e-9]
print(f"sanity: overlap trades with differing profit between st_gbb and st_bgo: {len(same)}")
print()
summarise("OVERLAP (priced via gbb arms)", ov_rev | ov_mom, A["st_gbb"], A["st_gbb_mx"])
summarise("OVERLAP (priced via bgo arms)", ov_rev | ov_mom, A["st_bgo"], A["st_bgo_mx"])
print()
summarise("ALL gbb entries (context)", set(A["st_gbb"]), A["st_gbb"], A["st_gbb_mx"])
summarise("ALL breakout_gbb entries (context)", set(A["st_bgo"]), A["st_bgo"], A["st_bgo_mx"])
